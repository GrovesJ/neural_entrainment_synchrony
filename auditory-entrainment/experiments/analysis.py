"""
Analysis and plotting tools for the entrainment sweep.

- Loads "results/sweep.csv" 
- computes several statistics for each synchrony metric across syncopation levels
- makes plots of each metric vs syncopation 
"""

from __future__ import annotations

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = _ROOT / "results" / "sweep.csv"
FIGURES_DIR = _ROOT / "figures"

AXES = ["lhl", "offbeat_count", "wnbd"]
METRICS = [
    "plv_exc_bar", "plv_inh_bar", "plv_exc_beat", "plv_inh_beat",
    "vs_exc_bar", "vs_inh_bar", "vs_exc_beat", "vs_inh_beat",
    "corr_exc", "corr_inh",
    "rate_exc_hz", "rate_inh_hz",
]
PRIMARY_METRICS = ["plv_exc_beat", "vs_exc_beat", "corr_exc", "rate_exc_hz"]


def load_sweep(path: str | Path = RESULTS_CSV) -> pd.DataFrame:
    """Read a sweep CSV into a DataFrame"""
    return pd.read_csv(path)


def gradient(df: pd.DataFrame) -> pd.DataFrame:
    """The accent gradient rows"""
    return df[df["pattern"] != 0].copy()


def baseline(df: pd.DataFrame) -> pd.DataFrame:
    """The simple beat baseline"""
    return df[df["pattern"] == 0].copy()


def describe(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Mean / std / min / max of each metric, one row per metric."""
    metrics = metrics or METRICS
    cols = [m for m in metrics if m in df.columns]
    return df[cols].agg(["mean", "std", "min", "max"]).T


def group_means(df: pd.DataFrame, axis: str, metrics=None) -> pd.DataFrame:
    """Mean of each metric grouped by a syncopation axis"""
    metrics = metrics or METRICS
    cols = [m for m in metrics if m in df.columns]
    return df.groupby(axis)[cols].mean()


def _corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    """Pearson or Spearman correlation coefficient of two 1D arrays"""
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Least square fit"""
    slope, intercept, r, p, _se = stats.linregress(x, y)
    return float(slope), float(intercept), float(r), float(p)  # type: ignore


def correlation_table(df: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    """Correlation of each metric row with each syncopation axis"""
    rows = []
    for m in METRICS:
        if m not in df.columns:
            continue
        row: dict[str, object] = {"metric": m}
        for a in AXES:
            if a not in df.columns or df[a].nunique() < 2:
                row[a] = np.nan
            else:
                row[a] = _corr(df[a].to_numpy(float), df[m].to_numpy(float), method)
        rows.append(row)
    return pd.DataFrame(rows).set_index("metric")


def trend_table(df: pd.DataFrame) -> pd.DataFrame:
    """Linear regression of each metric on each axis"""
    rows = []
    for m in METRICS:
        if m not in df.columns:
            continue
        for a in AXES:
            if a not in df.columns:
                continue
            x = df[a].to_numpy(float)
            y = df[m].to_numpy(float)
            if np.unique(x).size < 2:
                continue
            slope, _intercept, r, p = _linfit(x, y)
            rows.append({
                "metric": m, "axis": a, "slope": slope,
                "r": r, "r2": r ** 2, "p": p, "n": x.size,
            })
    return pd.DataFrame(rows)


def plot_metric_vs_axis(df, metric: str, axis: str, ax=None, annotate: bool = True):
    ax = ax if ax is not None else plt.gca()
    x = df[axis].to_numpy(float)
    y = df[metric].to_numpy(float)
    ax.scatter(x, y, s=40, zorder=3)
    title = metric
    if np.unique(x).size >= 2:
        slope, intercept, r, p = _linfit(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, intercept + slope * xs, color="C1", lw=1.4, zorder=2)
        if annotate:
            title = f"{metric}\nr={r:.2f}, p={p:.3f}"
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(axis)
    ax.set_ylabel(metric)
    return ax


def figure_metric_panels(df, outdir, metrics=None, axes=None) -> list[Path]:
    metrics = metrics or PRIMARY_METRICS
    axes = axes or AXES
    outdir = Path(outdir)
    paths = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        fig, axs = plt.subplots(1, len(axes), figsize=(4.0 * len(axes), 3.2), squeeze=False)
        for ax, a in zip(axs[0], axes):
            plot_metric_vs_axis(df, metric, a, ax=ax)
        fig.suptitle(f"{metric} vs syncopation axes")
        fig.tight_layout()
        p = outdir / f"metric_{metric}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def figure_offbeat_groups(df, outdir, metrics=None) -> Path:
    metrics = metrics or PRIMARY_METRICS
    metrics = [m for m in metrics if m in df.columns]
    fig, axs = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 3.2), squeeze=False)
    for ax, m in zip(axs[0], metrics):
        g = df.groupby("offbeat_count")[m]
        mean = g.mean()
        sem = g.std() / np.sqrt(g.count().clip(lower=1))
        ax.bar(mean.index, mean, yerr=sem, capsize=3, color="C0")
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("off-beat accents")
    fig.suptitle("Metrics by off-beat accent count")
    fig.tight_layout()
    p = Path(outdir) / "offbeat_groups.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_CSV
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else FIGURES_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_sweep(csv)
    grad = gradient(df)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    print(f"Loaded {len(df)} rows from {csv} "
          f"({len(grad)} gradient + {len(df) - len(grad)} baseline)")

    print("\n== Gradient metrics (mean/std/min/max) ==")
    print(describe(grad))

    print("\n== Metric means by off-beat count ==")
    print(group_means(grad, "offbeat_count"))

    print("\n== Spearman correlations (metric x axis) ==")
    print(correlation_table(grad, "spearman"))

    print("\n== Linear trends (metric ~ axis) ==")
    print(trend_table(grad)[["metric", "axis", "slope", "r", "r2", "p"]])

    if not baseline(df).empty:
        print("\n== beat-alone baseline ==")
        print(describe(baseline(df)))

    print("\nFigures:")
    for p in figure_metric_panels(grad, outdir):
        print(" ", p)
    print(" ", figure_offbeat_groups(grad, outdir))


if __name__ == "__main__":
    main()



