"""
- Simulates the medium entrainment network
- Simulates a set of rhythms in order of syncopation
- Scores each and saves summary statistics and figures

By default the run uses five eighth note patterns. 
Use --sixteenth to use 19 sixteenth note LHL level patterns instead
Use --no-background to run without background rhythm (this is the control)
"""


from __future__ import annotations

# Standard library imports
import argparse
import csv
import sys
from pathlib import Path
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
from experiments.sweep import _score, MEDIUM_ENTRAINMENT_PARAMS, _thal_count


RESULTS_DIR = _ROOT / "results"
FIGURES_DIR = _ROOT / "figures"


DEFAULT_PATTERNS = (1,2,3,4,5)
DEFAULT_PATTERNS_16 = stimuli.LHL_LEVELS_16
MEDIUM_STIMULUS = {
    "step_duration": 0.25,
    "rate_on": 40.0,
    "rate_off": 0.0,
    "jitter_ms": 1.0,
}


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


def simulate_gradient(
    patterns=DEFAULT_PATTERNS,
    duration_ms: float = 10000.0,
    dt: float = 0.1,
    bg_noise: float = 3.4,
    n_seeds: int = 3,
    base_seed: int = 42,
    sixteenth: bool = False,
    use_background: bool = True,
) -> list[dict]:
    """Simulate each rhythm and return one row per (pattern, seed) with stats    """
    pattern_table, name_table, bg_pattern, step_duration = _pattern_tables(sixteenth)
    make_fn = stimuli.make_rhythm_16 if sixteenth else stimuli.make_rhythm
    rows: list[dict] = []
    background = None
    if use_background:
        background = stimuli.Rhythm(
            bg_pattern,
            step_duration=step_duration,
            rate_on=MEDIUM_STIMULUS["rate_on"],
            name="beat",
        )
    for pid in patterns:
        rhythm = make_fn(
            pid,
            **{**MEDIUM_STIMULUS, "step_duration": step_duration},
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
            row = {
                "pattern": pid,
                "name": name_table[pid],
                "lhl": syncopation.lhl(pattern_table[pid]),
                "offbeat_count": syncopation.offbeat_count(pattern_table[pid]),
                "wnbd": syncopation.wnbd(pattern_table[pid]),
                "freq": rhythm.pulse_frequency,
                "seed": base_seed + k,
                "background": int(use_background),
                **_score(result),
            }
            rows.append(row)
            print(
                f"  {pid:>2} {row['name']:<10} seed={base_seed + k}  "
                f"plv_beat={row['plv_exc_beat']:.3f}/{row['plv_inh_beat']:.3f}  "
                f"vs_beat={row['vs_exc_beat']:.3f}/{row['vs_inh_beat']:.3f}  "
                f"r_e={row['rate_exc_hz']:.2f} Hz  r_i={row['rate_inh_hz']:.2f} Hz"
            )
    return rows

def _pattern_order(rows: list[dict]) -> list[tuple[int, str, int]]:
    seen: dict[int, tuple[str, int]] = {}
    for r in rows:
        seen.setdefault(r["pattern"], (r["name"], r["lhl"]))
    return [(pid, name, lhl) for pid, (name, lhl) in seen.items()]


def _series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    order = [pid for pid, _, _ in _pattern_order(rows)]
    means, sems = [], []
    for pid in order:
        vals = np.asarray([r[key] for r in rows if r["pattern"] == pid], dtype=float)
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0)
    return np.asarray(means), np.asarray(sems)


def _pattern_value(rows: list[dict], key: str) -> np.ndarray:
    """Per pattern value for each metric for each pattern, in order"""
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
    ("plv_exc_beat", "plv_inh_beat", "PLV to beat (2 Hz)", 0.0),
    ("vs_exc_beat", "vs_inh_beat", "Vector strength to beat (2 Hz)", 0.0),
    ("corr_exc", "corr_inh", "Mean pairwise corr", None),
    ("rate_exc_hz", "rate_inh_hz", "Firing rate (Hz)", None),
]

# Syncopation measures carried in each row, as (column, axis label).
SYNCOPATION_MEASURES = [
    ("wnbd", "WNBD (normalised note-to-beat distance)"),
    ("offbeat_count", "Offbeat count"),
]


# Plotting and saving results
def plot_summary(rows: list[dict], outdir: Path, stem: str = "syncopation_demo",
                 subtitle: str = "") -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    order = _pattern_order(rows)
    x = np.arange(len(order))
    labels = [
        f"LHL={lhl}" if name == f"lhl{lhl}" else f"{name}\nLHL={lhl}"
        for _, name, lhl in order
    ]
    n_seeds = len(rows) // len(order)
    panel_width = max(4.6, 0.34 * len(order))

    panels = NEURAL_PANELS
    fig, axes = plt.subplots(
        1, len(panels), figsize=(panel_width * len(panels), 4.2), squeeze=True
    )
    for ax, (e_key, i_key, title, ymin) in zip(axes, panels):
        e_mean, e_sem = _series(rows, e_key)
        i_mean, i_sem = _series(rows, i_key)
        width = 0.35
        ax.bar(x - width / 2, e_mean, width, yerr=e_sem, capsize=3,
               color="tab:blue", label="excitatory")
        ax.bar(x + width / 2, i_mean, width, yerr=i_sem, capsize=3,
               color="tab:red", label="inhibitory")
        ax.set_xticks(x)
        ax.set_xticklabels(
            labels,
            rotation=90 if len(order) > 8 else 0,
            fontsize=8 if len(order) > 8 else None,
        )
        ax.set_title(title)
        if ymin is not None:
            ax.set_ylim(ymin, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.suptitle(
        f"Syncopation gradient (mean \u00b1 SEM, {n_seeds} seeds)"
        + (f"\n{subtitle}" if subtitle else ""),
        fontsize=13,
    )
    fig.tight_layout()
    path = outdir / f"{stem}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_measures(rows: list[dict], outdir: Path, stem: str = "syncopation_demo",
                  subtitle: str = "") -> Path | None:
    """Neural metrics against each syncopation measure.

    One row per syncopation measure in "SYNCOPATION_MEASURES" and one column
    per metric in "NEURAL_PANELS", so the same metric can be compared across
    ordinations. Points are pattern means (seed SEM bars) labelled with their
    LHL level; repeated x values are the patterns that tie on that measure.
    """
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
    fieldnames = list(rows[0])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    parser.add_argument("--duration-ms", type=float, default=10000.0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--bg-noise", type=float, default=3.4)
    parser.add_argument("--no-background", action="store_true",
                        help="Remove background beat (control)")
    parser.add_argument("--csv", default="syncopation_demo.csv")
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    args = parser.parse_args(argv)

    if args.patterns:
        patterns = tuple(int(p) for p in args.patterns.split(","))
    elif args.sixteenth:
        patterns = DEFAULT_PATTERNS_16
    else:
        patterns = DEFAULT_PATTERNS
    resolution = "sixteenth-note LHL" if args.sixteenth else "eighth-note"
    context = "no background beat" if args.no_background else "isochronous beat"
    print(
        f"Simulating {len(patterns)} {resolution} patterns {patterns} "
        f"({args.duration_ms:.0f} ms, bg_noise={args.bg_noise} mV, "
        f"{args.n_seeds} seeds, {context})..."
    )
    rows = simulate_gradient(
        patterns=patterns,
        duration_ms=args.duration_ms,
        bg_noise=args.bg_noise,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        sixteenth=args.sixteenth,
        use_background=not args.no_background,
    )
    csv_path = _save_csv(rows, args.csv)
    print(f"Saved results to {csv_path}")
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


if __name__ == "__main__":
    main()
