"""Plot synchrony-demo CSV results with comparable axes.

Example:

    python experiments/plotting.py results/syncopation_drive_matched_120hz.csv \
        --outdir figures/drive_matched_120hz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PANELS = (
    ("plv_exc_bar", "plv_inh_bar", "PLV to bar (0.5 Hz)", 0.0, 1.0),
    ("vs_exc_bar", "vs_inh_bar", "Vector strength to bar (0.5 Hz)", 0.0, 1.0),
    ("corr_exc", "corr_inh", "Mean pairwise correlation", 0.0, None),
    ("rate_exc_hz", "rate_inh_hz", "Firing rate (Hz)", 0.0, None),
)


def _series(
    rows: pd.DataFrame, key: str, levels: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    means = []
    sems = []
    for level in levels:
        values = pd.to_numeric(
            rows.loc[rows["lhl"] == level, key], errors="coerce"
        ).dropna()
        means.append(float(values.mean()) if not values.empty else np.nan)
        sems.append(
            float(values.std(ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else 0.0
        )
    return np.asarray(means), np.asarray(sems)


def _shared_limits(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for exc_key, inh_key, _title, lower, fixed_upper in PANELS:
        if fixed_upper is not None:
            limits[exc_key] = (lower, fixed_upper)
            continue
        values = pd.concat(
            [pd.to_numeric(df[exc_key], errors="coerce"),
             pd.to_numeric(df[inh_key], errors="coerce")]
        ).dropna()
        upper = max(float(values.max()) * 1.1, 1.0) if not values.empty else 1.0
        limits[exc_key] = (lower, upper)
    return limits


def plot_condition(
    rows: pd.DataFrame,
    outdir: Path,
    stem: str,
    subtitle: str,
    limits: dict[str, tuple[float, float]],
) -> Path:
    levels = sorted(rows["lhl"].dropna().astype(int).unique())
    fig, axes = plt.subplots(
        1, len(PANELS), figsize=(max(18.0, 2.8 * len(levels)), 5.2),
        squeeze=False,
    )
    x = np.arange(len(levels))
    labels = [str(level) for level in levels]
    for index, (exc_key, inh_key, title, _lower, _upper) in enumerate(PANELS):
        ax = axes[0, index]
        exc_mean, exc_sem = _series(rows, exc_key, levels)
        inh_mean, inh_sem = _series(rows, inh_key, levels)
        width = 0.36
        ax.bar(x - width / 2, exc_mean, width, yerr=exc_sem, capsize=3,
               color="tab:blue", label="excitatory")
        ax.bar(x + width / 2, inh_mean, width, yerr=inh_sem, capsize=3,
               color="tab:red", label="inhibitory")
        ax.set_xticks(x, labels)
        ax.set_xlabel("LHL")
        ax.set_title(title)
        lower, upper = limits[exc_key]
        ax.set_ylim(lower, upper)
        ax.grid(axis="y", alpha=0.3)
    axes[0, 0].set_ylabel("mean +/- SEM")
    axes[0, 0].legend()
    fig.suptitle(f"{subtitle} (shared axes across conditions)", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    path = outdir / f"{stem}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_csv(csv_path: Path, outdir: Path) -> list[Path]:
    df = pd.read_csv(csv_path)
    required = {"lhl", "condition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
        )
    outdir.mkdir(parents=True, exist_ok=True)
    limits = _shared_limits(df)
    paths = []
    for condition in df["condition"].dropna().astype(str).unique():
        rows = df[df["condition"].astype(str) == condition]
        paths.append(plot_condition(
            rows,
            outdir,
            f"{csv_path.stem}_{condition}",
            f"{csv_path.stem}: {condition}",
            limits,
        ))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("figures") / "replotted")
    args = parser.parse_args()
    for path in plot_csv(args.csv, args.outdir):
        print(f"Saved figure to {path}")


if __name__ == "__main__":
    main()
