"""Persistence experiments with piecewise rhythmic stimulation.

Examples
--------
Run patterns 1, 3, and 4 with a silent interval of one stimulus duration
before, between, and after the stimuli:

    python experiments/persistence.py --patterns 1 3 4

Run the same sequence without the continuous on-beat background:

    python experiments/persistence.py --patterns 1 3 4 --no-background

Run three sections for each generated LHL level 1, 3, and 7:

    python experiments/persistence.py --lhl-levels 1 3 7 --repeats 3

The public ``run_persistence`` function accepts built-in pattern IDs or
``stimuli.Rhythm`` objects, per-section durations, gaps, and parameter sets
for reuse from notebooks.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _path in (_ROOT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src import model, stimuli
from experiments.measures import synchrony
from experiments.sweep import (
    MEDIUM_ENTRAINMENT_PARAMS,
    _thal_count,
    _build_meter,
    combined_fundamental_hz,
)

RESULTS_DIR = _ROOT / "results"
FIGURES_DIR = _ROOT / "figures"
MEDIUM_ENTRAINMENT = MEDIUM_ENTRAINMENT_PARAMS
DEFAULT_STEP_DURATION = 0.125
DEFAULT_SECTION_DURATION = 8.0
DEFAULT_RATE_ON = 40.0
DEFAULT_JITTER_MS = 1.0
DEFAULT_BG_NOISE = 3.4
DEFAULT_GAP_SECONDS = None
PatternInput = int | stimuli.Rhythm | None


@dataclass(frozen=True)
class ScheduledRhythm:
    """A rhythm made of time-local sections; ``None`` means silence."""

    sections: tuple[tuple[float, float, stimuli.Rhythm | None], ...]
    name: str = "scheduled stimulus"

    def spike_generator_group(self, duration: float, n_neurons: int = 100, seed: int = 0):
        """Create one Brian2 input group containing all scheduled sections."""
        times_parts: list[np.ndarray] = []
        index_parts: list[np.ndarray] = []
        for section_index, (start, stop, rhythm) in enumerate(self.sections):
            if rhythm is None or start >= duration or stop <= start:
                continue
            local_duration = min(stop, duration) - start
            times, indices = rhythm.spike_trains(
                local_duration,
                n_neurons=n_neurons,
                seed=seed + section_index,
            )
            times_parts.append(times + start)
            index_parts.append(indices)

        if not times_parts:
            return _empty_spike_group(n_neurons)

        times = np.concatenate(times_parts)
        indices = np.concatenate(index_parts).astype(np.int32, copy=False)
        keep = (times >= 0.0) & (times < duration)
        return _b2.SpikeGeneratorGroup(
            n_neurons, indices[keep], times[keep] * _b2.second
        )


try:
    import brian2 as _b2
except ImportError:  # pragma: no cover - model import reports this in normal use.
    _b2 = None


def _empty_spike_group(n_neurons: int):
    if _b2 is None:
        raise ImportError("Brian2 is required to build scheduled input groups")
    return _b2.SpikeGeneratorGroup(
        n_neurons, np.empty(0, dtype=np.int32), np.empty(0) * _b2.second
    )


def make_rhythm(
    pattern_id: int,
    *,
    step_duration: float = DEFAULT_STEP_DURATION,
    rate_on: float = DEFAULT_RATE_ON,
    rate_off: float = 0.0,
    jitter_ms: float = DEFAULT_JITTER_MS,
) -> stimuli.Rhythm:
    """Build one sixteenth-note rhythm from the repository pattern table."""
    if pattern_id not in stimuli.PATTERNS_16:
        raise KeyError(
            f"Unknown sixteenth-note pattern {pattern_id}; "
            f"available patterns: {list(stimuli.PATTERNS_16)}"
        )
    return stimuli.make_rhythm_16(
        pattern_id,
        step_duration=step_duration,
        rate_on=rate_on,
        rate_off=rate_off,
        jitter_ms=jitter_ms,
    )


def make_background(
    *,
    meter: str = "beat",
    step_duration: float = DEFAULT_STEP_DURATION,
    rate_on: float = DEFAULT_RATE_ON,
) -> stimuli.Rhythm:
    """Build the continuous metrical background drive.

    ``meter`` selects the metrical context: ``beat`` (default on-beat pulse),
    ``downbeat``, ``beats13``, or ``accent`` (a metrically weighted drive on all
    16 steps, which requires ``rate_on > 56 Hz``).
    """
    return _build_meter(meter, step_duration, rate_on)


def schedule_background(
    schedule: ScheduledRhythm,
    *,
    meter: str = "beat",
    step_duration: float = DEFAULT_STEP_DURATION,
    rate_on: float = DEFAULT_RATE_ON,
) -> ScheduledRhythm:
    """Keep the metrical background active throughout the schedule."""
    background = make_background(
        meter=meter, step_duration=step_duration, rate_on=rate_on
    )
    sections = tuple(
        (start, stop, background)
        for start, stop, _ in schedule.sections
    )
    return ScheduledRhythm(sections, name=f"scheduled {meter} background")


def build_schedule(
    patterns: Sequence[PatternInput],
    *,
    durations: float | Sequence[float] | None = None,
    gaps: float | Sequence[float] | None = DEFAULT_GAP_SECONDS,
    step_duration: float = DEFAULT_STEP_DURATION,
    rate_on: float = DEFAULT_RATE_ON,
    rate_off: float = 0.0,
    jitter_ms: float = DEFAULT_JITTER_MS,
) -> tuple[ScheduledRhythm, list[dict]]:
    """Build a continuous schedule from arbitrary patterns and gaps.

    ``None`` in ``patterns`` is an explicit silent section.  If ``gaps`` is
    ``None``, every automatic gap has the duration of the adjacent stimulus
    section.  A scalar applies before, between, and after all sections.  A
    sequence may provide ``len(patterns) + 1`` values in that same order.
    Explicit ``None`` sections still receive their requested duration.
    """
    pattern_ids = tuple(patterns)
    if not pattern_ids:
        raise ValueError("patterns must contain at least one pattern or None")

    section_durations = _expand_values(
        durations, len(pattern_ids), "durations",
        default=DEFAULT_SECTION_DURATION,
    )
    if gaps is None:
        gap_values: list[float | None] = [None] * (len(pattern_ids) + 1)
    elif np.isscalar(gaps):
        gap_values = [float(gaps)] * (len(pattern_ids) + 1)
    else:
        gap_values = [float(value) for value in gaps]
        if len(gap_values) != len(pattern_ids) + 1:
            raise ValueError(
                f"gaps must have {len(pattern_ids) + 1} values "
                "(leading, between, and trailing)"
            )
        if any(value < 0 for value in gap_values):
            raise ValueError("gaps must be non-negative")

    sections: list[tuple[float, float, stimuli.Rhythm | None]] = []
    metadata: list[dict] = []
    current = 0.0
    leading_gap = gap_values[0]
    if leading_gap is None:
        leading_gap = section_durations[0]
    if leading_gap > 0:
        gap_stop = current + leading_gap
        sections.append((current, gap_stop, None))
        metadata.append({
            "section": len(metadata),
            "kind": "gap",
            "pattern": None,
            "name": "gap",
            "start_s": current,
            "stop_s": gap_stop,
            "duration_s": leading_gap,
            "frequency_hz": np.nan,
        })
        current = gap_stop

    for index, (pattern_input, section_duration) in enumerate(
        zip(pattern_ids, section_durations)
    ):
        if isinstance(pattern_input, stimuli.Rhythm):
            rhythm = pattern_input
            pattern_id = rhythm.name
            label = rhythm.name
            phase_kind = "stimulus"
        elif pattern_input is not None:
            pattern_id = pattern_input
            rhythm = make_rhythm(
                pattern_id,
                step_duration=step_duration,
                rate_on=rate_on,
                rate_off=rate_off,
                jitter_ms=jitter_ms,
            )
            label = stimuli.PATTERN_NAMES_16[pattern_id]
            phase_kind = "stimulus"
        else:
            pattern_id = None
            rhythm = None
            label = "silence"
            phase_kind = "silence"

        stop = current + section_duration
        sections.append((current, stop, rhythm))
        metadata.append({
            "section": index,
            "kind": phase_kind,
            "pattern": pattern_id,
            "name": label,
            "start_s": current,
            "stop_s": stop,
            "duration_s": section_duration,
            "frequency_hz": rhythm.pulse_frequency if rhythm else np.nan,
        })
        current = stop

        gap = gap_values[index + 1]
        if gap is None:
            gap = section_duration
        if gap < 0:
            raise ValueError("gaps must be non-negative")
        if gap > 0:
            gap_stop = current + gap
            sections.append((current, gap_stop, None))
            metadata.append({
                "section": len(metadata),
                "kind": "gap",
                "pattern": None,
                "name": "gap",
                "start_s": current,
                "stop_s": gap_stop,
                "duration_s": gap,
                "frequency_hz": np.nan,
            })
            current = gap_stop

    return ScheduledRhythm(tuple(sections)), metadata


def _expand_values(
    values: float | Sequence[float] | None,
    count: int,
    name: str,
    *,
    default: float,
) -> list[float | None]:
    if values is None:
        return [default] * count
    if np.isscalar(values):
        result = [float(values)] * count
    else:
        result = [float(value) for value in values]
        if len(result) != count:
            raise ValueError(f"{name} must have {count} values")
    if any(value < 0 for value in result if value is not None):
        raise ValueError(f"{name} must be non-negative")
    return result


def _phase_metrics(
    result: model.Result,
    start: float,
    stop: float,
    frequency: float,
) -> dict[str, float]:
    """Measure PLV, vector strength, and pairwise XCorr within one section."""
    rate_mask = (result.rate_t >= start) & (result.rate_t < stop)
    if rate_mask.sum() < 2:
        return {key: float("nan") for key in (
            "plv_exc", "plv_inh", "vs_exc", "vs_inh", "xcorr_exc", "xcorr_inh",
        )}

    fs = 1.0 / float(np.mean(np.diff(result.rate_t[rate_mask])))
    spike_mask = (result.spike_times >= start) & (result.spike_times < stop)
    times = result.spike_times[spike_mask]
    indices = result.spike_indices[spike_mask]
    exc_mask = indices < result.n_exc
    exc_times = times[exc_mask] - start
    exc_indices = indices[exc_mask]
    inh_times = times[~exc_mask] - start
    inh_indices = indices[~exc_mask] - result.n_exc
    duration = stop - start

    if not np.isfinite(frequency) or frequency <= 0:
        plv_exc = plv_inh = vs_exc = vs_inh = float("nan")
    else:
        plv_exc = synchrony.plv(
            result.rate_exc[rate_mask], fs, frequency, t0=start
        )
        plv_inh = synchrony.plv(
            result.rate_inh[rate_mask], fs, frequency, t0=start
        )
        vs_exc = synchrony.mean_vector_strength(
            exc_times, exc_indices, result.n_exc, frequency, t0=start
        )
        vs_inh = synchrony.mean_vector_strength(
            inh_times, inh_indices, result.n_inh, frequency, t0=start
        )

    return {
        "plv_exc": plv_exc,
        "plv_inh": plv_inh,
        "vs_exc": vs_exc,
        "vs_inh": vs_inh,
        "xcorr_exc": synchrony.spike_pairwise_corr(
            exc_times, exc_indices, result.n_exc, duration
        ),
        "xcorr_inh": synchrony.spike_pairwise_corr(
            inh_times, inh_indices, result.n_inh, duration
        ),
    }


def _phase_rows(
    result: model.Result,
    metadata: Sequence[dict],
    *,
    background: bool,
    seed: int,
) -> list[dict]:
    rows = []
    for phase_index, section in enumerate(metadata):
        frequency = section.get("frequency_hz", float("nan"))
        row = dict(section)
        row.update({
            "phase": phase_index,
            "background": int(background),
            "seed": seed,
            "rate_exc_hz": float(result.rate_exc[
                (result.rate_t >= section["start_s"])
                & (result.rate_t < section["stop_s"])
            ].mean()),
            "rate_inh_hz": float(result.rate_inh[
                (result.rate_t >= section["start_s"])
                & (result.rate_t < section["stop_s"])
            ].mean()),
        })
        row.update(_phase_metrics(
            result,
            section["start_s"],
            section["stop_s"],
            frequency,
        ))
        row["reference_frequency_hz"] = frequency
        rows.append(row)
    return rows


def _simulate(
    schedule: ScheduledRhythm,
    *,
    parameter_set: Mapping[str, float],
    background: bool,
    meter: str,
    meter_rate_on: float,
    step_duration: float,
    duration_s: float,
    dt: float,
    bg_noise: float,
    seed: int,
) -> model.Result:
    parameter_values = dict(parameter_set)
    parameter_values.update({
        "n_thal": _thal_count(parameter_values),
        "duration": duration_s * 1000.0,
        "dt": dt,
        "seed": seed,
        "bg_noise": bg_noise,
    })
    params = model.Params(**parameter_values)
    background_rhythm = (
        schedule_background(
            schedule,
            meter=meter,
            step_duration=step_duration,
            rate_on=meter_rate_on,
        )
        if background
        else None
    )
    return model.simulate(params, stimulus=schedule, background=background_rhythm)


def run_persistence(
    patterns: Sequence[PatternInput],
    *,
    parameter_set: Mapping[str, float] = MEDIUM_ENTRAINMENT,
    durations: float | Sequence[float] | None = None,
    gaps: float | Sequence[float] | None = DEFAULT_GAP_SECONDS,
    background: bool = True,
    meter: str = "beat",
    meter_rate_on: float | None = None,
    step_duration: float = DEFAULT_STEP_DURATION,
    rate_on: float = DEFAULT_RATE_ON,
    rate_off: float = 0.0,
    jitter_ms: float = DEFAULT_JITTER_MS,
    dt: float = 0.1,
    bg_noise: float = DEFAULT_BG_NOISE,
    seed: int = 42,
    save: bool = True,
    label: str = "persistence",
) -> tuple[model.Result, list[dict]]:
    """Run one continuous experiment and return per-section metrics.

    ``meter`` selects the background meter (``beat``, ``downbeat``,
    ``beats13``, ``accent``). When ``meter == "accent"`` and no explicit
    ``meter_rate_on`` is given, the stimulus and meter both move to the accent's
    recommended rate so the comparison stays drive-matched.

    Phase metrics are scored against the fundamental frequency of the stimulus
    actually presented in each section (foreground rhythm plus the metrical
    meter), matching the frequency-aware scoring in sweep/syncopation_demo.
    """
    if background and meter == "accent" and meter_rate_on is None:
        if not stimuli.accent_is_representable(rate_on, step_duration):
            chosen = stimuli.METRICAL_ACCENT_DEFAULT_RATE_ON
            print(
                f"  accent meter: selected {chosen:g} Hz for stimulus and meter "
                "(drive-matched, best realised hierarchy)"
            )
            rate_on = chosen
            meter_rate_on = chosen
    if meter_rate_on is None:
        meter_rate_on = rate_on

    schedule, metadata = build_schedule(
        patterns,
        durations=durations,
        gaps=gaps,
        step_duration=step_duration,
        rate_on=rate_on,
        rate_off=rate_off,
        jitter_ms=jitter_ms,
    )

    # Score each section against the frequency of the stimulus actually
    # presented in that section (foreground rhythm plus the metrical meter),
    # mirroring the frequency-aware scoring in sweep/syncopation_demo.
    meter_rhythm = (
        make_background(
            meter=meter, step_duration=step_duration, rate_on=meter_rate_on
        )
        if background
        else None
    )
    for section, (_, _, rhythm) in zip(metadata, schedule.sections):
        if rhythm is None:
            section["frequency_hz"] = float("nan")
        else:
            section["frequency_hz"] = combined_fundamental_hz(rhythm, meter_rhythm)

    duration_s = metadata[-1]["stop_s"]
    result = _simulate(
        schedule,
        parameter_set=parameter_set,
        background=background,
        meter=meter,
        meter_rate_on=meter_rate_on,
        step_duration=step_duration,
        duration_s=duration_s,
        dt=dt,
        bg_noise=bg_noise,
        seed=seed,
    )
    rows = _phase_rows(
        result,
        metadata,
        background=background,
        seed=seed,
    )
    for row in rows:
        row["meter"] = meter if background else "none"
        row["meter_rate_on_hz"] = meter_rate_on if background else 0.0
    if save:
        RESULTS_DIR.mkdir(exist_ok=True)
        FIGURES_DIR.mkdir(exist_ok=True)
        _write_csv(rows, RESULTS_DIR / f"{label}_metrics.csv")
        plot_result(
            result, schedule, metadata, label, FIGURES_DIR / f"{label}.png",
            background=background,
            meter=meter,
            meter_rate_on=meter_rate_on,
            step_duration=step_duration,
        )
        plot_section_metrics(
            rows,
            label,
            FIGURES_DIR / f"{label}_metrics.png",
        )
        plot_section_diagnostics(
            result,
            metadata,
            label,
            FIGURES_DIR / f"{label}_sections.png",
        )
    return result, rows


def _write_csv(rows: Sequence[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _binned_spike_trains(
    spike_times: np.ndarray,
    spike_indices: np.ndarray,
    n_units: int,
    duration: float,
    bin_size: float,
) -> np.ndarray:
    """Return one binned spike train per neuron."""
    n_bins = max(1, int(np.ceil(duration / bin_size)))
    counts = np.zeros((n_units, n_bins), dtype=float)
    if spike_times.size:
        bins = np.clip((spike_times / bin_size).astype(np.int64), 0, n_bins - 1)
        keep = (spike_indices >= 0) & (spike_indices < n_units)
        np.add.at(counts, (spike_indices[keep], bins[keep]), 1.0)
    return counts - counts.mean(axis=1, keepdims=True)


def _average_cross_correlation(
    trains_a: np.ndarray,
    trains_b: np.ndarray | None = None,
    bin_size: float = 0.001,
    max_lag: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Average within- or between-population binned spike correlations."""
    max_lag_bins = int(round(max_lag / bin_size))
    lags = np.arange(-max_lag_bins, max_lag_bins + 1) * bin_size
    a = np.asarray(trains_a, dtype=float)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        return lags, np.zeros(lags.size)
    n_bins = a.shape[1]
    if trains_b is None:
        if a.shape[0] < 2:
            return lags, np.zeros(lags.size)
        aggregate = np.correlate(a.sum(axis=0), a.sum(axis=0), mode="full")
        correlation = aggregate[
            n_bins - 1 - max_lag_bins:n_bins + max_lag_bins
        ].copy()
        correlation[max_lag_bins] -= np.einsum("ij,ij->", a, a)
        for lag in range(1, max_lag_bins + 1):
            value = np.einsum("ij,ij->", a[:, :-lag], a[:, lag:])
            correlation[max_lag_bins + lag] -= value
            correlation[max_lag_bins - lag] -= value
        correlation /= a.shape[0] * (a.shape[0] - 1)
        norm = n_bins * float(a.std(axis=1).mean()) ** 2
    else:
        b = np.asarray(trains_b, dtype=float)
        if b.ndim != 2 or b.shape[0] == 0 or b.shape[1] != n_bins:
            return lags, np.zeros(lags.size)
        correlation = np.correlate(
            a.sum(axis=0), b.sum(axis=0), mode="full"
        )[n_bins - 1 - max_lag_bins:n_bins + max_lag_bins]
        correlation /= a.shape[0] * b.shape[0]
        norm = n_bins * float(a.std(axis=1).mean()) * float(b.std(axis=1).mean())
    if norm > 0:
        correlation /= norm
    return lags, correlation


def _section_cross_correlations(
    result: model.Result,
    start: float,
    stop: float,
    bin_size: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return E-E, I-I, and E-I correlograms within one section."""
    in_section = (result.spike_times >= start) & (result.spike_times < stop)
    times = result.spike_times[in_section] - start
    indices = result.spike_indices[in_section]
    exc = indices < result.n_exc
    exc_trains = _binned_spike_trains(
        times[exc], indices[exc], result.n_exc, stop - start, bin_size
    )
    inh_trains = _binned_spike_trains(
        times[~exc], indices[~exc] - result.n_exc,
        result.n_inh, stop - start, bin_size,
    )
    lags, ee = _average_cross_correlation(exc_trains, bin_size=bin_size)
    _, ii = _average_cross_correlation(inh_trains, bin_size=bin_size)
    _, ei = _average_cross_correlation(
        exc_trains, inh_trains, bin_size=bin_size
    )
    return lags, ee, ii, ei


def plot_result(
    result: model.Result,
    schedule: ScheduledRhythm,
    metadata: Sequence[dict],
    label: str = "persistence",
    path: Path | None = None,
    background: bool = False,
    meter: str = "beat",
    meter_rate_on: float = DEFAULT_RATE_ON,
    step_duration: float = DEFAULT_STEP_DURATION,
) -> None:
    """Save continuous input and population-rate diagnostics."""
    fig = plt.figure(figsize=(14, 14))
    grid = fig.add_gridspec(2, 1, hspace=0.5)
    t_rate = result.rate_t
    axes = [fig.add_subplot(grid[index]) for index in range(2)]

    drive = np.zeros_like(t_rate)
    background_drive = np.zeros_like(t_rate)
    if background:
        background_rhythm = schedule_background(
            schedule,
            meter=meter,
            step_duration=step_duration,
            rate_on=meter_rate_on,
        )
        for start, stop, rhythm in background_rhythm.sections:
            if rhythm is not None:
                mask = (t_rate >= start) & (t_rate < stop)
                background_drive[mask] = rhythm.rate(t_rate[mask])
    section_handles = []
    section_colors = plt.get_cmap("tab20").colors
    for section_index, (start, stop, rhythm) in enumerate(schedule.sections):
        if rhythm is not None:
            mask = (t_rate >= start) & (t_rate < stop)
            drive[mask] = rhythm.rate(t_rate[mask])
            color = section_colors[section_index % len(section_colors)]
        else:
            color = "0.75"
        section_label = metadata[section_index]["name"]
        if metadata[section_index]["kind"] == "gap":
            section_label = "gap"
        else:
            section_label = f"{section_index}: {section_label}"
        section_handles.append(Patch(
            facecolor=color,
            edgecolor="none",
            alpha=0.3,
            label=section_label,
        ))
        for axis in axes:
            axis.axvspan(
                start, stop,
                color=color,
                alpha=0.12 if rhythm is not None else 0.18,
            )
    axes[0].plot(t_rate, drive, color="tab:orange", label="stimulus rate")
    if background:
        axes[0].plot(
            t_rate, background_drive, color="tab:green",
            linestyle="--", label="background rate",
        )
    axes[0].set_ylabel("input (Hz)")
    axes[0].set_title("Continuous input drive")
    axes[0].legend(
        handles=[
            *axes[0].get_legend_handles_labels()[0],
            *section_handles,
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
    )

    axes[1].plot(t_rate, result.rate_exc, color="tab:blue", label="excitatory rate")
    axes[1].plot(t_rate, result.rate_inh, color="tab:red", label="inhibitory rate")
    axes[1].set_ylabel("rate (Hz)")
    axes[1].set_title("Excitatory and inhibitory population rates")
    axes[1].legend(loc="upper right")
    time_ticks = np.arange(
        0.0,
        np.ceil(result.params.duration / 1000.0 / 2.0) * 2.0 + 2.0,
        2.0,
    )
    for axis in axes[:2]:
        axis.set_xticks(time_ticks)
        axis.set_xlim(0.0, result.params.duration / 1000.0)

    fig.suptitle(label)
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_section_diagnostics(
    result: model.Result,
    metadata: Sequence[dict],
    label: str = "persistence",
    path: Path | None = None,
) -> None:
    """Plot a phase histogram and E-I correlogram for each stimulus section."""
    stimulus_rows = [row for row in metadata if row["kind"] == "stimulus"]
    if not stimulus_rows:
        return

    fig = plt.figure(
        figsize=(22, max(3.5, 3.8 * len(stimulus_rows)))
    )
    grid = fig.add_gridspec(len(stimulus_rows), 5)
    axes = []
    for row_index in range(len(stimulus_rows)):
        axes.append([
            fig.add_subplot(grid[row_index, column], projection="polar")
            if column < 2 else fig.add_subplot(grid[row_index, column])
            for column in range(5)
        ])
    for row_index, row in enumerate(stimulus_rows):
        start, stop = row["start_s"], row["stop_s"]
        in_section = (result.spike_times >= start) & (result.spike_times < stop)
        exc = in_section & (result.spike_indices < result.n_exc)
        inh = in_section & (result.spike_indices >= result.n_exc)
        frequency = row.get("frequency_hz", float("nan"))
        if not np.isfinite(frequency) or frequency <= 0:
            frequency = float("nan")
        exc_phase = (
            2.0 * np.pi * frequency * (result.spike_times[exc] - start)
        ) % (2 * np.pi)
        inh_phase = (
            2.0 * np.pi * frequency * (result.spike_times[inh] - start)
        ) % (2 * np.pi)
        phase_exc_axis, phase_inh_axis, ee_axis, ii_axis, ei_axis = axes[row_index]

        if exc_phase.size:
            phase_exc_axis.hist(
                exc_phase, bins=24, density=True,
                color="tab:blue", alpha=0.75,
            )
            z = np.mean(np.exp(1j * exc_phase))
            phase_exc_axis.annotate(
                "", xy=(np.angle(z), np.abs(z)), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=2.0),
            )
        if inh_phase.size:
            phase_inh_axis.hist(
                inh_phase, bins=24, density=True,
                color="tab:red", alpha=0.75,
            )
            z = np.mean(np.exp(1j * inh_phase))
            phase_inh_axis.annotate(
                "", xy=(np.angle(z), np.abs(z)), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=2.0),
            )
        for axis, title, color in (
            (phase_exc_axis, "E spike phase", "tab:blue"),
            (phase_inh_axis, "I spike phase", "tab:red"),
        ):
            axis.set_theta_zero_location("N")
            axis.set_theta_direction(-1)
            axis.set_title(title)
            if not axis.patches:
                axis.bar([0.0], [0.0], width=0.1, color=color, alpha=0.0)
                axis.text(
                    0.5, 0.5, "no spikes",
                    transform=axis.transAxes, ha="center", va="center",
                )
        lags, ee, ii, ei = _section_cross_correlations(
            result, start=start, stop=stop
        )
        for axis, correlation, title, color in (
            (ee_axis, ee, "E-E", "tab:blue"),
            (ii_axis, ii, "I-I", "tab:red"),
            (ei_axis, ei, "E-I", "tab:purple"),
        ):
            axis.plot(lags * 1000.0, correlation, color=color)
            axis.axhline(0.0, color="black", linewidth=0.6)
            axis.set_title(
                f"Section {row['section']}: {row['name']} {title} cross-correlogram"
            )
            axis.set_xlabel("lag (ms)")
            axis.set_ylabel("correlation")
        phase_exc_axis.set_title(
            f"Section {row['section']}: {row['name']} E phase\n"
            f"({start:.2f}-{stop:.2f} s; {frequency:.2f} Hz)"
        )
        phase_inh_axis.set_title("I phase")

    fig.suptitle(f"{label}: diagnostics by stimulus section")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_section_metrics(
    rows: Sequence[dict],
    label: str = "persistence",
    path: Path | None = None,
) -> None:
    """Plot PLV, VS, and XCorr separately for each stimulus section."""
    stimulus_rows = [row for row in rows if row["kind"] == "stimulus"]
    if not stimulus_rows:
        return

    fig, axes = plt.subplots(
        len(stimulus_rows),
        1,
        figsize=(10, max(3.0, 3.0 * len(stimulus_rows))),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    metric_names = ("plv", "vs", "xcorr")
    metric_labels = ("PLV", "VS", "XCorr")
    x = np.arange(len(metric_names))
    width = 0.36
    all_values = [
        value
        for row in stimulus_rows
        for metric in metric_names
        for value in (row[f"{metric}_exc"], row[f"{metric}_inh"])
        if np.isfinite(value)
    ]
    if all_values:
        lower = min(0.0, min(all_values))
        upper = max(1.0, max(all_values))
        if lower == upper:
            upper = lower + 1.0
        for axis in axes[:, 0]:
            axis.set_ylim(lower, upper)

    for axis, row in zip(axes[:, 0], stimulus_rows):
        exc_values = [row[f"{metric}_exc"] for metric in metric_names]
        inh_values = [row[f"{metric}_inh"] for metric in metric_names]
        axis.bar(
            x - width / 2,
            exc_values,
            width,
            color="tab:blue",
            label="excitatory",
        )
        axis.bar(
            x + width / 2,
            inh_values,
            width,
            color="tab:red",
            label="inhibitory",
        )
        frequency = row.get("reference_frequency_hz", float("nan"))
        freq_txt = (
            f"{frequency:.2f} Hz" if np.isfinite(frequency) else "no reference"
        )
        section_label = (
            f"Section {row['section']}: {row['name']} "
            f"({row['start_s']:.2f}-{row['stop_s']:.2f} s; {freq_txt})"
        )
        axis.set_title(section_label, loc="left")
        axis.set_ylabel("value")
        axis.set_xticks(x, metric_labels)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="upper right")

    axes[-1, 0].set_xlabel("section metrics")
    fig.suptitle(
        f"{label}: metrics by stimulus section "
        "(phase metrics referenced to each section's stimulus frequency)"
    )
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    pattern_group = parser.add_mutually_exclusive_group(required=True)
    pattern_group.add_argument("--patterns", nargs="+", type=int,
                               help="Built-in sixteenth-note pattern IDs.")
    pattern_group.add_argument(
        "--lhl-levels",
        nargs="+",
        type=int,
        metavar="LEVEL",
        help="Generated full-period LHL levels; each is repeated --repeats times.",
    )
    parser.add_argument("--repeats", type=int, default=3,
                        help="Sections per generated LHL level (default: 3).")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stimulus duration in seconds; defaults to 8 seconds.")
    parser.add_argument("--gap", type=float, default=None,
                        help="Gap in seconds; defaults to each section duration.")
    parser.add_argument("--no-background", action="store_true")
    parser.add_argument("--meter", choices=("beat", "downbeat", "beats13", "accent"),
                        default="beat",
                        help="Background meter used throughout the schedule "
                             "(default: beat).")
    parser.add_argument("--meter-rate-on", type=float, default=None,
                        help="Onset rate for the background meter; defaults to "
                             "the stimulus rate. The accent meter needs > 56 Hz.")
    parser.add_argument("--rate-on", type=float, default=DEFAULT_RATE_ON,
                        help="Onset rate for foreground rhythms "
                             "(default: 40 Hz).")
    parser.add_argument("--noise", type=float, default=DEFAULT_BG_NOISE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="persistence")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.lhl_levels is not None:
        from experiments.rhythm_generator import lhl_gradient, pattern_string_to_tuple

        generated = lhl_gradient(levels=args.lhl_levels)
        patterns = [
            stimuli.Rhythm(
                pattern_string_to_tuple(str(row["pattern"])),
                step_duration=DEFAULT_STEP_DURATION,
                rate_on=args.rate_on,
                rate_off=0.0,
                jitter_ms=DEFAULT_JITTER_MS,
                name=f"lhl-{int(row['lhl'])}",
            )
            for row in generated
            for _ in range(args.repeats)
        ]
    else:
        patterns = args.patterns
    run_persistence(
        patterns,
        durations=args.duration,
        gaps=args.gap,
        background=not args.no_background,
        meter=args.meter,
        rate_on=args.rate_on,
        meter_rate_on=args.meter_rate_on,
        bg_noise=args.noise,
        seed=args.seed,
        label=args.label,
    )


if __name__ == "__main__":
    main()
