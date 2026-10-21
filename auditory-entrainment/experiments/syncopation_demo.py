"""Simulate a low/mid/high syncopation gradient and save summary figures.

Simulates a small set of rhythms spanning the syncopation gradient
(eighths -> tresillo -> offbeat by default), scores each with the same
medium entrainment parameter set, writes one row per (pattern, seed) to
``results/syncopation_demo.csv``, and renders a summary bar chart (PLV,
vector strength, firing rate) with seed-averaged error bars under
``figures/``.

The simulation is the slow part, so run this once on the faster machine
and keep the CSV and PNG for later use.

Usage::

    python experiments/syncopation_demo.py [--patterns 3,4,5] \
        [--duration-ms 8000] [--n-seeds 3] [--bg-noise 3.0]
"""

from __future__ import annotations

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

from src import model, stimuli
from experiments.measures import syncopation
from experiments.sweep import _score

RESULTS_DIR = _ROOT / "results"
FIGURES_DIR = _ROOT / "figures"

# Low, mid, high syncopation: eighths (no syncopation), tresillo, offbeat.
DEFAULT_PATTERNS = (3, 4, 5)

# "Medium Entrainment" from notebooks/parameter_sets.json, translated to
# model.Params names. The preset's show_background setting is represented by
# the on-beat background passed to model.simulate below.
MEDIUM_ENTRAINMENT_PARAMS = {
    "I_bias": 125.0,
    "I_bias_std": 15.0,
    "weight_lognormal_sigma": 0.4,
    "w_thal": 0.08,
    "w_ee": 1.5,
    "w_ei": 0.85,
    "w_ie": 3.0,
    "w_ii": 4.0,
    "p_ee": 0.02,
    "p_ei": 0.4,
    "p_ie": 0.5,
    "p_ii": 0.5,
    "p_thal": 0.4,
    "delay_ei": 1.5,
    "delay_ie": 1.5,
    "delay_ei_std": 0.5,
    "delay_ie_std": 0.5,
    "n_exc": 6400,
    "n_inh": 1600,
    "synaptic_normalization": "sqrt",
    "n_exc_ref": 400,
    "n_inh_ref": 100,
    "n_thal_ref": 20,
}
MEDIUM_STIMULUS = {
    "step_duration": 0.25,
    "rate_on": 40.0,
    "rate_off": 0.0,
    "jitter_ms": 1.0,
}


def _thal_count(pset: dict) -> int:
    """Thalamic population size, mirroring the notebook's size-invariant rule.

    With synaptic normalization on, the thalamic population scales with
    ``n_exc`` so the feedforward:recurrent ratio stays size-invariant;
    with normalization off, the base reference size is used unchanged.
    """
    if pset["synaptic_normalization"] != "none":
        return max(1, round(pset["n_thal_ref"] * pset["n_exc"] / pset["n_exc_ref"]))
    return int(pset["n_thal_ref"])


def simulate_gradient(
    patterns=DEFAULT_PATTERNS,
    duration_ms: float = 10000.0,
    dt: float = 0.1,
    bg_noise: float = 3.4,
    n_seeds: int = 3,
    base_seed: int = 42,
) -> list[dict]:
    """Simulate each rhythm and return one row per (pattern, seed)."""
    rows: list[dict] = []
    background = stimuli.Rhythm(
        stimuli.ONBEAT_BACKGROUND,
        step_duration=MEDIUM_STIMULUS["step_duration"],
        rate_on=MEDIUM_STIMULUS["rate_on"],
        name="beat",
    )
    for pid in patterns:
        rhythm = stimuli.make_rhythm(
            pid,
            **MEDIUM_STIMULUS,
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
                "name": stimuli.PATTERN_NAMES[pid],
                "lhl": syncopation.lhl(stimuli.PATTERNS[pid]),
                "offbeat_count": syncopation.offbeat_count(stimuli.PATTERNS[pid]),
                "wnbd": syncopation.wnbd(stimuli.PATTERNS[pid]),
                "freq": rhythm.pulse_frequency,
                "seed": base_seed + k,
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


def _save_csv(rows: list[dict], filename: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    fieldnames = list(rows[0])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _pattern_order(rows: list[dict]) -> list[tuple[int, str, int]]:
    """Unique ``(pattern, name, lhl)`` triples in first-seen order."""
    seen: dict[int, tuple[str, int]] = {}
    for r in rows:
        seen.setdefault(r["pattern"], (r["name"], r["lhl"]))
    return [(pid, name, lhl) for pid, (name, lhl) in seen.items()]


def _series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-pattern mean and SEM (across seeds) of ``key``, in pattern order."""
    order = [pid for pid, _, _ in _pattern_order(rows)]
    means, sems = [], []
    for pid in order:
        vals = np.asarray([r[key] for r in rows if r["pattern"] == pid], dtype=float)
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0)
    return np.asarray(means), np.asarray(sems)


def plot_summary(rows: list[dict], outdir: Path) -> Path:
    """Bar chart of PLV, vector strength, pairwise correlation, and firing rate."""
    outdir.mkdir(parents=True, exist_ok=True)
    order = _pattern_order(rows)
    x = np.arange(len(order))
    labels = [f"{name}\nLHL={lhl}" for _, name, lhl in order]
    n_seeds = len(rows) // len(order)

    # (excitatory key, inhibitory key, title, y-axis lower bound or None for auto)
    panels = [
        ("plv_exc_beat", "plv_inh_beat", "PLV to beat (2 Hz)", 0.0),
        ("vs_exc_beat", "vs_inh_beat", "Vector strength to beat (2 Hz)", 0.0),
        ("corr_exc", "corr_inh", "Mean pairwise corr", None),
        ("rate_exc_hz", "rate_inh_hz", "Firing rate (Hz)", None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), squeeze=True)
    for ax, (e_key, i_key, title, ymin) in zip(axes, panels):
        e_mean, e_sem = _series(rows, e_key)
        i_mean, i_sem = _series(rows, i_key)
        width = 0.35
        ax.bar(x - width / 2, e_mean, width, yerr=e_sem, capsize=3,
               color="tab:blue", label="excitatory")
        ax.bar(x + width / 2, i_mean, width, yerr=i_sem, capsize=3,
               color="tab:red", label="inhibitory")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        if ymin is not None:
            ax.set_ylim(ymin, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    fig.suptitle(
        f"Syncopation gradient (mean \u00b1 SEM, {n_seeds} seeds)", fontsize=13
    )
    fig.tight_layout()
    path = outdir / "syncopation_demo.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Simulate low/mid/high syncopation and save a summary figure."
    )
    parser.add_argument("--patterns", default=",".join(map(str, DEFAULT_PATTERNS)),
                        help="Comma-separated pattern ids (default 3,4,5).")
    parser.add_argument("--duration-ms", type=float, default=10000.0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--bg-noise", type=float, default=3.4)
    parser.add_argument("--csv", default="syncopation_demo.csv")
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    args = parser.parse_args(argv)

    patterns = tuple(int(p) for p in args.patterns.split(","))
    print(
        f"Simulating patterns {patterns} ({args.duration_ms:.0f} ms, "
        f"bg_noise={args.bg_noise} mV, {args.n_seeds} seeds)..."
    )
    rows = simulate_gradient(
        patterns=patterns,
        duration_ms=args.duration_ms,
        bg_noise=args.bg_noise,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
    )
    csv_path = _save_csv(rows, args.csv)
    print(f"Saved results to {csv_path}")
    fig_path = plot_summary(rows, Path(args.figures_dir))
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
