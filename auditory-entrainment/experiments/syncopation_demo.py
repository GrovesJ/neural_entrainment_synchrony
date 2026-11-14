"""
- Simulates the medium entrainment network
- Simulates a set of rhythms in order of syncopation
- Scores each and saves summary statistics and figures

By default the run uses five eighth note patterns. 
Use --sixteenth to use the 16-step patterns instead.
Use --no-background to run without a background meter.

Background meters (--meter):
  beat      on-beat pulse at steps 0,4,8,12 (default)
  downbeat  single onset at step 0 (sparsest metrical anchor)
  beats13   onsets at steps 0 and 8
  accent    metrical accent on all 16 steps (weighted; needs > 56 Hz)

Sparse meters are preferred when studying syncopation: a dense meter is an
onset on every step, so the acoustic onset union no longer distinguishes the
foreground rhythm. Syncopation is always scored on the foreground rhythm
against the fixed metrical hierarchy, since the meter supplies the grid
rather than being a measured stimulus. The CSV still records the onset union
as a factual description of what was presented.
"""


from __future__ import annotations

# Standard library imports
import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (_ROOT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Internal imports
from src import model, stimuli
from experiments.measures import syncopation
from experiments.sweep import (
    _score,
    MEDIUM_ENTRAINMENT_PARAMS,
    _thal_count,
    stimulus_frequencies,
    combined_frequencies,
    beat_reference_is_informative,
    _nan_uninformative_beat,
)


RESULTS_DIR = _ROOT / "results"
FIGURES_DIR = _ROOT / "figures"


DEFAULT_PATTERNS = (1,2,3,4,5)
DEFAULT_PATTERNS_16 = tuple(stimuli.PATTERNS_16)
MEDIUM_STIMULUS = {
    "step_duration": 0.25,
    "rate_on": 40.0,
    "rate_off": 0.0,
    "jitter_ms": 1.0,
}

# Background options for the 16-step grid. Sparse meters keep the rhythm's
# onsets distinguishable in ``rhythm | meter``; a dense meter (``accent``) is an
# onset on every step and therefore erases the foreground rhythm's syncopation.
METERS = ("beat", "downbeat", "beats13", "accent")

_SPARSE_METERS = {
    "downbeat": stimuli.DOWNBEAT_16,
    "beats13": stimuli.BEATS_1_3_16,
}


def build_meter(
    meter: str = "beat",
    step_duration: float = 0.125,
    rate_on: float = 40.0,
) -> stimuli.Rhythm | None:
    """Return the requested background meter, or None when disabled."""
    if meter == "none":
        return None
    if meter == "beat":
        pattern = (
            stimuli.ONBEAT_BACKGROUND_16
            if step_duration == 0.125
            else stimuli.ONBEAT_BACKGROUND
        )
        return stimuli.Rhythm(
            tuple(pattern), step_duration=step_duration, rate_on=rate_on, name="beat"
        )
    if meter in _SPARSE_METERS:
        if step_duration != 0.125:
            raise ValueError(
                f"the {meter!r} meter is defined on the 16-step grid; "
                "use --sixteenth"
            )
        return stimuli.Rhythm(
            tuple(_SPARSE_METERS[meter]),
            step_duration=step_duration,
            rate_on=rate_on,
            name=meter,
        )
    if meter == "accent":
        if not stimuli.accent_is_representable(rate_on, step_duration):
            minimum = stimuli.minimum_accent_rate(step_duration)
            raise ValueError(
                f"the accent meter needs rate_on > {minimum:g} Hz at "
                f"step_duration={step_duration:g} s "
                f"({stimuli.required_accent_spikes()} spikes per cycle), "
                f"got {rate_on:g} Hz; pass --meter-rate-on to raise it"
            )
        return stimuli.make_metrical_accent(
            step_duration=step_duration,
            rate_on=rate_on,
            name="meter",
        )
    raise ValueError(f"unknown meter {meter!r}; choose from {METERS} or 'none'")


def resolve_accent_rates(
    rate_on: float | None,
    meter_rate_on: float | None,
    meter: str,
    step_duration: float = 0.125,
) -> tuple[float, float, bool]:
    """Pick stimulus and meter rates, keeping the accent representable.

    Drive matching between the accent and a four-onset pattern holds at any
    rate, so the accent only needs a representable rate. When the caller left
    both rates at the default, select the accent's recommended rate for *both*
    channels so the comparison stays drive-matched. Returns
    ``(stimulus_rate, meter_rate, auto_selected)``.
    """
    default = float(MEDIUM_STIMULUS["rate_on"])
    stimulus = default if rate_on is None else float(rate_on)
    meter_rate = stimulus if meter_rate_on is None else float(meter_rate_on)

    if meter != "accent" or stimuli.accent_is_representable(meter_rate, step_duration):
        return stimulus, meter_rate, False

    if meter_rate_on is None and rate_on is None:
        chosen = stimuli.METRICAL_ACCENT_DEFAULT_RATE_ON
        return chosen, chosen, True

    raise ValueError(
        f"the accent meter needs rate_on > {stimuli.minimum_accent_rate(step_duration):g} Hz "
        f"at step_duration={step_duration:g} s, got {meter_rate:g} Hz"
    )


def _describe_drive(rhythm, background) -> str:
    """One-line summary of stimulus vs meter drive and their ratio."""
    stimulus = rhythm.spikes_per_cycle
    meter = background.spikes_per_cycle if background is not None else 0
    if meter == 0:
        return f"drive: rhythm {stimulus}/cycle, no meter"
    ratio = meter / stimulus if stimulus else float("inf")
    note = "  <- meter stronger than rhythm" if ratio > 1.0 else ""
    return (
        f"drive: rhythm {stimulus}/cycle, meter {meter}/cycle "
        f"(meter/rhythm {ratio:.2f}x), total {stimulus + meter}{note}"
    )


def _pattern_tables(sixteenth: bool) -> tuple[dict, dict, tuple, float]:
    """
    Return (patterns, names, background_pattern, step_duration) for eighth or sixteenth note patterns
    """
    if sixteenth:
        return (
            stimuli.PATTERNS_16,
            stimuli.PATTERN_NAMES_16,
            stimuli.ONBEAT_BACKGROUND_16,
            0.125,
        )
    return (
        stimuli.PATTERNS,
        stimuli.PATTERN_NAMES,
        stimuli.ONBEAT_BACKGROUND,
        0.25,
    )


def _pattern_specs_from_table(
    patterns,
    pattern_table,
    name_table,
    make_fn,
    stimulus_kwargs,
) -> list[dict]:
    """Build runtime specs from the built-in integer-keyed pattern tables."""
    specs = []
    for pid in patterns:
        specs.append({
            "id": pid,
            "name": name_table[pid],
            "pattern": tuple(pattern_table[pid]),
            "lhl": syncopation.lhl(pattern_table[pid]),
            "offbeat_count": syncopation.offbeat_count(pattern_table[pid]),
            "wnbd": syncopation.wnbd(pattern_table[pid]),
            "rhythm": make_fn(pid, **stimulus_kwargs),
        })
    return specs


def _simulate_specs(
    specs: list[dict],
    *,
    duration_ms: float,
    dt: float,
    bg_noise: float,
    n_seeds: int,
    base_seed: int,
    use_background: bool,
    meter: str,
    rate_on: float | None,
    meter_rate_on: float | None,
    step_duration: float,
    condition: str = "combined",
) -> list[dict]:
    """Simulate a list of rhythm specs and return one row per (spec, seed)."""
    freqs_of = combined_frequencies
    beat_informative = beat_reference_is_informative
    rate_on, meter_rate_on, auto_selected = resolve_accent_rates(
        rate_on, meter_rate_on, meter if use_background else "none", step_duration
    )
    if auto_selected:
        print(
            f"  accent meter: selected {rate_on:g} Hz for stimulus and meter "
            "(drive-matched, best realised hierarchy)"
        )
    background = (
        build_meter(meter, step_duration, meter_rate_on)
        if use_background
        else None
    )
    rows: list[dict] = []
    for spec_index, spec in enumerate(specs):
        rhythm = spec["rhythm"]
        combined = stimuli.combined_pattern(rhythm, background)
        freqs = freqs_of(rhythm, background)
        if spec_index == 0:
            print(f"  {_describe_drive(rhythm, background)}")
            print(
                "  reference frequencies: "
                + ", ".join(f"{name}={value:g} Hz" for name, value in freqs.items())
            )
        for k in range(n_seeds):
            params = model.Params(
                **MEDIUM_ENTRAINMENT_PARAMS,
                n_thal=_thal_count(MEDIUM_ENTRAINMENT_PARAMS),
                duration=duration_ms,
                dt=dt,
                seed=base_seed + k,
                bg_noise=bg_noise,
            )
            result = model.simulate(params, stimulus=rhythm, background=background)
            scored = _nan_uninformative_beat(
                _score(result, freqs),
                beat_informative(rhythm, background),
            )
            row = {
                "condition": condition,
                "pattern": spec["id"],
                "name": spec["name"],
                "lhl": spec["lhl"],
                "offbeat_count": spec["offbeat_count"],
                "wnbd": spec["wnbd"],
                "freq": rhythm.pulse_frequency,
                "seed": base_seed + k,
                "background": int(use_background),
                "meter": meter if use_background else "none",
                "stimulus_spikes_per_cycle": rhythm.spikes_per_cycle,
                "meter_spikes_per_cycle": (
                    background.spikes_per_cycle if background is not None else 0
                ),
                "meter_rate_on_hz": meter_rate_on if use_background else 0.0,
                "combined_onsets": "".join(str(value) for value in combined),
                "combined_n_onsets": int(sum(combined)),
                **{f"ref_{name}_hz": value for name, value in freqs.items()},
                "beat_reference_informative": int(beat_informative(rhythm, background)),
                **scored,
            }
            rows.append(row)
            onset_note = (
                f"  plv_onset={row['plv_exc_onset']:.3f}/{row['plv_inh_onset']:.3f}"
                if "plv_exc_onset" in row
                else ""
            )
            beat_note = (
                "" if row["beat_reference_informative"] else "  [beat ref not meaningful]"
            )
            print(
                f"  {spec['id']:>3} {row['name']:<10} seed={base_seed + k}  "
                f"plv_bar={row['plv_exc_bar']:.3f}/{row['plv_inh_bar']:.3f}  "
                f"{onset_note}{beat_note}  "
                f"r_e={row['rate_exc_hz']:.2f} Hz  r_i={row['rate_inh_hz']:.2f} Hz"
            )
    return rows


def simulate_gradient(
    patterns=DEFAULT_PATTERNS,
    duration_ms: float = 10000.0,
    dt: float = 0.1,
    bg_noise: float = 3.4,
    n_seeds: int = 3,
    base_seed: int = 42,
    sixteenth: bool = False,
    use_background: bool = True,
    meter: str = "beat",
    rate_on: float | None = None,
    meter_rate_on: float | None = None,
) -> list[dict]:
    """Simulate each built-in rhythm and return one row per (pattern, seed)."""
    pattern_table, name_table, _bg_pattern, step_duration = _pattern_tables(sixteenth)
    make_fn = stimuli.make_rhythm_16 if sixteenth else stimuli.make_rhythm
    if rate_on is None:
        rate_on = float(MEDIUM_STIMULUS["rate_on"])
    if meter_rate_on is None:
        meter_rate_on = rate_on
    stimulus_kwargs = {
        **MEDIUM_STIMULUS,
        "step_duration": step_duration,
        "rate_on": rate_on,
    }
    specs = _pattern_specs_from_table(
        patterns, pattern_table, name_table, make_fn, stimulus_kwargs
    )
    return _simulate_specs(
        specs,
        duration_ms=duration_ms,
        dt=dt,
        bg_noise=bg_noise,
        n_seeds=n_seeds,
        base_seed=base_seed,
        use_background=use_background,
        meter=meter,
        rate_on=rate_on,
        meter_rate_on=meter_rate_on,
        step_duration=step_duration,
    )


def simulate_lhl_gradient(
    *,
    levels: Sequence[int] | None = None,
    duration_ms: float = 10000.0,
    dt: float = 0.1,
    bg_noise: float = 3.4,
    n_seeds: int = 3,
    base_seed: int = 42,
    use_background: bool = True,
    meter: str = "beat",
    rate_on: float | None = None,
    meter_rate_on: float | None = None,
    step_duration: float = 0.125,
    condition: str = "combined",
) -> list[dict]:
    """Simulate a generated monotone LHL gradient (equal density + period).

    Patterns come from ``experiments.rhythm_generator``: one representative
    per LHL level (default 0-7), each with four onsets and a 16-step shortest
    period. This provides the controlled syncopation ladder that the built-in
    ``PATTERNS_16`` (LHL 0,1,1,5,11) do not.
    """
    from experiments.rhythm_generator import lhl_gradient, pattern_string_to_tuple

    if condition not in {"meter", "rhythm", "combined"}:
        raise ValueError(
            f"unknown condition {condition!r}; choose meter, rhythm, or combined"
        )
    generated = lhl_gradient(levels=levels)
    if rate_on is None:
        rate_on = float(MEDIUM_STIMULUS["rate_on"])
    if meter_rate_on is None:
        meter_rate_on = rate_on
    if condition == "rhythm":
        use_background = False
    elif condition == "meter":
        use_background = True
        meter_rate_on = rate_on
    specs = []
    for row in generated:
        pattern = pattern_string_to_tuple(row["pattern"])
        rhythm = stimuli.Rhythm(
            pattern if condition != "meter" else (0,) * len(pattern),
            step_duration=step_duration,
            rate_on=rate_on,
            rate_off=0.0,
            jitter_ms=MEDIUM_STIMULUS["jitter_ms"],
            name=f"lhl-{row['lhl']}",
        )
        specs.append({
            "id": int(row["lhl"]),
            "name": f"lhl-{row['lhl']}",
            "pattern": pattern,
            "lhl": int(row["lhl"]),
            "offbeat_count": int(row["offbeat_count"]),
            "wnbd": float(row["wnbd"]),
            "rhythm": rhythm,
        })
    return _simulate_specs(
        specs,
        duration_ms=duration_ms,
        dt=dt,
        bg_noise=bg_noise,
        n_seeds=n_seeds,
        base_seed=base_seed,
        use_background=use_background,
        meter=meter,
        rate_on=rate_on,
        meter_rate_on=meter_rate_on,
        step_duration=step_duration,
        condition=condition,
    )

def _pattern_order(rows: list[dict], sort_key: str | None = None) -> list[tuple[int, str, int]]:
    """Patterns in first-seen order, optionally sorted by a syncopation metric."""
    seen: dict[int, tuple[str, int]] = {}
    for r in rows:
        seen.setdefault(r["pattern"], (r["name"], r["lhl"]))
    order = [(pid, name, lhl) for pid, (name, lhl) in seen.items()]
    if sort_key is not None:
        values = {r["pattern"]: r[sort_key] for r in rows}
        order.sort(key=lambda item: values[item[0]])
    return order


def _series(rows: list[dict], key: str, order=None) -> tuple[np.ndarray, np.ndarray]:
    if order is None:
        order = [pid for pid, _, _ in _pattern_order(rows)]
    means, sems = [], []
    for pid in order:
        vals = np.asarray(
            [r.get(key, float("nan")) for r in rows if r["pattern"] == pid],
            dtype=float,
        )
        finite = vals[np.isfinite(vals)]
        means.append(float(finite.mean()) if finite.size else float("nan"))
        sem = (
            float(finite.std(ddof=1) / np.sqrt(finite.size))
            if finite.size > 1
            else 0.0
        )
        sems.append(sem)
    return np.asarray(means), np.asarray(sems)


def _pattern_value(rows: list[dict], key: str, order=None) -> np.ndarray:
    """Per pattern value for each metric for each pattern, in order"""
    if order is None:
        order = [pid for pid, _, _ in _pattern_order(rows)]
    return np.asarray(
        [next(r[key] for r in rows if r["pattern"] == pid) for pid in order],
        dtype=float,
    )


def _rank(values: np.ndarray) -> np.ndarray:
    """Ranks with tied values sharing their mean rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    sorted_values = values[order]
    i = 0
    while i < sorted_values.size:
        j = i
        while j + 1 < sorted_values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def _spearman(a, b) -> float:
    """Spearman rank correlation"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or b.size != a.size:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])

# (excitatory key, inhibitory key, title, y-axis lower bound or None for auto)
NEURAL_PANELS = [
    ("plv_exc_bar", "plv_inh_bar", "PLV to bar (0.5 Hz)", 0.0),
    ("vs_exc_bar", "vs_inh_bar", "Vector strength to bar (0.5 Hz)", 0.0),
    ("corr_exc", "corr_inh", "Mean pairwise corr", None),
    ("rate_exc_hz", "rate_inh_hz", "Firing rate (Hz)", None),
]

# Syncopation measures
SYNCOPATION_MEASURES = [
    ("lhl", "LHL"),
    ("offbeat_count", "Offbeat count"),
    ("wnbd", "WNBD (weighted note to beat distance)"),
]


# Rows of the  bar chart
# Plotting and saving results
def plot_summary(rows: list[dict], outdir: Path, stem: str = "syncopation_demo",
                 subtitle: str = "") -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    n_patterns = len({r["pattern"] for r in rows})
    n_seeds = len(rows) // n_patterns
    panel_width = max(4.6, 0.34 * n_patterns)

    fig, axes = plt.subplots(
        1, len(NEURAL_PANELS),
        figsize=(panel_width * len(NEURAL_PANELS), 5.2),
        squeeze=False,
    )
    # Sort by LHL and include the other syncopation measures in each label.
    order = _pattern_order(rows, sort_key="lhl")
    pids = [pid for pid, _, _ in order]
    labels = []
    for pid, name, _ in order:
        row = next(r for r in rows if r["pattern"] == pid)
        labels.append(
            f"{name}\n"
            f"LHL={row['lhl']:g}\n"
            f"offbeat={row['offbeat_count']:g}\n"
            f"WNBD={row['wnbd']:.3f}"
        )
    x = np.arange(len(order))
    for col_i, (e_key, i_key, title, ymin) in enumerate(NEURAL_PANELS):
        ax = axes[0, col_i]
        e_mean, e_sem = _series(rows, e_key, order=pids)
        i_mean, i_sem = _series(rows, i_key, order=pids)
        width = 0.35
        ax.bar(x - width / 2, e_mean, width, yerr=e_sem, capsize=3,
               color="tab:blue", label="excitatory")
        ax.bar(x + width / 2, i_mean, width, yerr=i_sem, capsize=3,
               color="tab:red", label="inhibitory")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(title)
        if ymin is not None:
            ax.set_ylim(ymin, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("ordered by LHL")
    axes[0, 0].legend()
    fig.suptitle(
        f"Syncopation gradient (mean \u00b1 SEM, {n_seeds} seeds)"
        + (f"\n{subtitle}" if subtitle else ""),
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    path = outdir / f"{stem}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_measures(rows: list[dict], outdir: Path, stem: str = "syncopation_demo",
                  subtitle: str = "") -> Path | None:
    """    Neural sycnchrony metrics vs each syncopation measure    """
    measures = [(key, label) for key, label in SYNCOPATION_MEASURES if key in rows[0]]
    if not measures:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    order = [pid for pid, _, _ in _pattern_order(rows)]
    lhl = _pattern_value(rows, "lhl")
    n_seeds = int(np.max([
        sum(1 for r in rows if r["pattern"] == pid) for pid in order
    ]))

    fig, axes = plt.subplots(
        len(measures), len(NEURAL_PANELS),
        figsize=(4.3 * len(NEURAL_PANELS), 3.6 * len(measures)),
        squeeze=False,
    )
    for row_i, (measure_key, measure_label) in enumerate(measures):
        xvals = _pattern_value(rows, measure_key)
        unique_x = np.unique(xvals)
        # Spread patterns that tie on the measure so their markers stay visible.
        span = unique_x.max() - unique_x.min()
        nudge = 0.012 * span if span > 0 else 0.02
        if len(unique_x) == 1:
            offsets = np.linspace(-0.05, 0.05, len(xvals))
        else:
            counted: dict[float, int] = {}
            offsets = np.empty(len(xvals))
            for i, xv in enumerate(xvals):
                seen = counted.get(float(xv), 0)
                counted[float(xv)] = seen + 1
                offsets[i] = seen * nudge
            for xv in unique_x:
                members = np.flatnonzero(xvals == xv)
                offsets[members] -= offsets[members].mean()

        for col_i, (e_key, i_key, title, ymin) in enumerate(NEURAL_PANELS):
            ax = axes[row_i, col_i]
            e_mean, e_sem = _series(rows, e_key)
            i_mean, i_sem = _series(rows, i_key)
            ax.errorbar(xvals + offsets, e_mean, yerr=e_sem, fmt="o", ms=5,
                        color="tab:blue", ecolor="tab:blue", elinewidth=1.0,
                        capsize=2, label="excitatory")
            ax.errorbar(xvals + offsets, i_mean, yerr=i_sem, fmt="s", ms=5,
                        color="tab:red", ecolor="tab:red", elinewidth=1.0,
                        capsize=2, label="inhibitory")
            for xv, yv, lv in zip(xvals + offsets, e_mean, lhl):
                ax.annotate(f"{int(lv)}", (xv, yv), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color="tab:blue")
            rho = _spearman(xvals, e_mean)
            rho_txt = "n/a" if np.isnan(rho) else f"{rho:+.2f}"
            ax.set_title(f"{title}\nvs {measure_key} (exc \u03c1={rho_txt})",
                         fontsize=9)
            ax.set_xlabel(measure_label, fontsize=8)
            if ymin is not None:
                ax.set_ylim(ymin, 1.0)
            ax.grid(alpha=0.3)
            if row_i == len(measures) - 1 and col_i == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        "Neural metrics vs syncopation measures "
        f"(pattern means \u00b1 SEM, up to {n_seeds} seeds; "
        "labels are LHL levels)"
        + (f"\n{subtitle}" if subtitle else ""),
        fontsize=12,
    )
    fig.tight_layout()
    path = outdir / f"{stem}_vs_syncopation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _save_csv(rows: list[dict], filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    # Rows can carry different metric columns when their reference frequencies
    # differ (e.g. a pattern whose onset rate duplicates the beat rate), so
    # build the schema from the union of keys in first-seen order.
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Simulate a syncopation gradient (eighth note patterns or "
                    "sixteenth note LHL levels) and save a figure"
    )
    parser.add_argument(
        "--patterns", default=None,
        help="Comma separated pattern ids: eighth-note ids 1-5, or LHL levels "
             "0-18 with --sixteenth (default: every pattern in the resolution).",
    )
    parser.add_argument("--sixteenth", action="store_true",
                        help="Use the sixteenth note LHL-level patterns instead "
                             "of the eighth note patterns.")
    parser.add_argument(
        "--lhl-gradient", action="store_true",
        help="Simulate a generated monotone LHL gradient (equal onset count "
             "and full period) instead of the built-in patterns. "
             "Overrides --patterns/--sixteenth.",
    )
    parser.add_argument(
        "--lhl-levels", default=None,
        help="Comma separated LHL levels to include when --lhl-gradient is set "
             "(default: all available levels, 0-7).",
    )
    parser.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated generated-gradient conditions: meter,rhythm,combined.",
    )
    parser.add_argument("--duration-ms", type=float, default=10000.0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--bg-noise", type=float, default=3.4)
    parser.add_argument("--no-background", action="store_true",
                        help="Remove background beat (control)")
    parser.add_argument("--meter", choices=(*METERS, "none"), default="beat",
                        help="Background meter: on-beat pulse (beat), downbeat, "
                             "beats 1+3 (beats13), metrical accent, or none. "
                             "default: beat.")
    parser.add_argument("--rate-on", type=float, default=None,
                        help="Thalamic onset rate in Hz for the foreground rhythm "
                             "(default: 40).")
    parser.add_argument(
        "--combined-rate-on", type=float, default=None,
        help="Total rate for the combined condition; meter and rhythm each use "
             "half this rate (default: --rate-on).",
    )
    parser.add_argument("--meter-rate-on", type=float, default=None,
                        help="Separate onset rate for the background meter "
                             "(default: same as --rate-on). Lower it to keep the "
                             "meter weaker than the rhythm.")
    parser.add_argument("--csv", default="syncopation_demo.csv")
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    args = parser.parse_args(argv)
    conditions = (
        tuple(value.strip() for value in args.conditions.split(",") if value.strip())
        if args.conditions
        else None
    )
    if conditions is not None:
        if not args.lhl_gradient:
            parser.error("--conditions requires --lhl-gradient")
        invalid = [value for value in conditions
                   if value not in {"meter", "rhythm", "combined"}]
        if invalid:
            parser.error(
                f"unknown condition(s): {', '.join(invalid)}; "
                "choose meter, rhythm, or combined"
            )
    if args.combined_rate_on is not None and args.combined_rate_on <= 0:
        parser.error("--combined-rate-on must be greater than zero")

    if args.lhl_gradient:
        patterns = None
        lhl_levels = (
            tuple(int(level) for level in args.lhl_levels.split(","))
            if args.lhl_levels
            else None
        )
        resolution = (
            "generated LHL gradient "
            f"(levels {','.join(map(str, lhl_levels)) if lhl_levels else '0-7'})"
        )
    else:
        if args.lhl_levels:
            parser.error("--lhl-levels requires --lhl-gradient")
        if args.patterns:
            patterns = tuple(int(p) for p in args.patterns.split(","))
        elif args.sixteenth:
            patterns = DEFAULT_PATTERNS_16
        else:
            patterns = DEFAULT_PATTERNS
        resolution = "sixteenth-note LHL" if args.sixteenth else "eighth-note"
    meter = "none" if args.no_background else args.meter
    context = "no background" if meter == "none" else f"{meter} meter"
    print(
        f"Simulating {resolution} "
        f"({args.duration_ms:.0f} ms, bg_noise={args.bg_noise} mV, "
        f"{args.n_seeds} seeds, {context})..."
    )
    try:
        if args.lhl_gradient:
            if conditions is None:
                rows = simulate_lhl_gradient(
                    levels=lhl_levels,
                    duration_ms=args.duration_ms,
                    bg_noise=args.bg_noise,
                    n_seeds=args.n_seeds,
                    base_seed=args.base_seed,
                    use_background=meter != "none",
                    meter=meter,
                    rate_on=args.rate_on,
                    meter_rate_on=args.meter_rate_on,
                )
            else:
                base_rate = (
                    float(args.rate_on)
                    if args.rate_on is not None
                    else float(MEDIUM_STIMULUS["rate_on"])
                )
                combined_rate = (
                    base_rate
                    if args.combined_rate_on is None
                    else float(args.combined_rate_on)
                )
                rows = []
                for condition in conditions:
                    component_rate = (
                        combined_rate / 2.0
                        if condition == "combined"
                        else base_rate
                    )
                    rows.extend(simulate_lhl_gradient(
                        levels=lhl_levels,
                        duration_ms=args.duration_ms,
                        bg_noise=args.bg_noise,
                        n_seeds=args.n_seeds,
                        base_seed=args.base_seed,
                        use_background=condition != "rhythm" and meter != "none",
                        meter=meter,
                        rate_on=component_rate,
                        meter_rate_on=(
                            component_rate
                            if condition == "combined"
                            else args.meter_rate_on
                        ),
                        condition=condition,
                    ))
        else:
            rows = simulate_gradient(
                patterns=patterns,
                duration_ms=args.duration_ms,
                bg_noise=args.bg_noise,
                n_seeds=args.n_seeds,
                base_seed=args.base_seed,
                sixteenth=args.sixteenth,
                use_background=meter != "none",
                meter=meter,
                rate_on=args.rate_on,
                meter_rate_on=args.meter_rate_on,
            )
    except ValueError as exc:
        parser.error(str(exc))
    csv_path = _save_csv(rows, args.csv)
    print(f"Saved results to {csv_path}")
    if conditions is None:
        fig_path = plot_summary(
            rows,
            Path(args.figures_dir),
            stem=Path(args.csv).stem,
            subtitle=context,
        )
        print(f"Saved figure to {fig_path}")
        measures_path = plot_measures(
            rows,
            Path(args.figures_dir),
            stem=Path(args.csv).stem,
            subtitle=context,
        )
        if measures_path is not None:
            print(f"Saved syncopation measure figure to {measures_path}")
    else:
        for condition in conditions:
            condition_rows = [
                row for row in rows if row["condition"] == condition
            ]
            condition_stem = f"{Path(args.csv).stem}_{condition}"
            fig_path = plot_summary(
                condition_rows,
                Path(args.figures_dir),
                stem=condition_stem,
                subtitle=f"{context}; {condition}",
            )
            print(f"Saved figure to {fig_path}")
            measures_path = plot_measures(
                condition_rows,
                Path(args.figures_dir),
                stem=condition_stem,
                subtitle=f"{context}; {condition}",
            )
            if measures_path is not None:
                print(f"Saved syncopation measure figure to {measures_path}")


if __name__ == "__main__":
    main()
