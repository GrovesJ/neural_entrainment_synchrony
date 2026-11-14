"""Run rhythm, parameter, and jitter sweeps."""

from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (_ROOT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src import model, stimuli
from experiments.measures import syncopation, synchrony

RESULTS_DIR = _ROOT / "results"

_BAR_FREQ = 0.5
_BEAT_FREQ = 2.0

# Phase metrics are only identifiable when the recording spans at least this
# many cycles of the reference frequency; shorter windows return NaN.
_MIN_CYCLES = 2.0

_METRIC_KEYS = (
    "plv_exc_bar", "plv_inh_bar", "plv_exc_beat", "plv_inh_beat",
    "vs_exc_bar", "vs_inh_bar", "vs_exc_beat", "vs_inh_beat",
    "corr_exc", "corr_inh",
    "rate_exc_hz", "rate_inh_hz",
)

# Beat-frequency phase metrics that are only meaningful when the presented
# stimulus actually repeats at the metrical beat rate.
_BEAT_METRIC_KEYS = (
    "plv_exc_beat", "plv_inh_beat", "vs_exc_beat", "vs_inh_beat",
)


def _nan_uninformative_beat(metrics: dict, informative: bool) -> dict:
    """Null beat-frequency metrics when the beat reference is absent.

    The beat column is kept (it is the correct metric for isochronous pulses),
    but when ``informative`` is False the stimulus has no 2 Hz periodicity and
    the computed value is a placeholder. Replace it with NaN so a downstream
    plot or table cannot silently read a meaningless number.
    """
    if informative:
        return metrics
    result = dict(metrics)
    for key in _BEAT_METRIC_KEYS:
        if key not in result:
            continue
        values = result[key]
        if isinstance(values, list):
            result[key] = [float("nan")] * len(values)
        else:
            result[key] = float("nan")
    return result


def stimulus_frequencies(rhythm) -> dict[str, float]:
    """Reference frequencies implied by a stimulus.

    ``bar`` and ``beat`` come from the metrical grid, ``onset`` is the event
    rate actually presented, and ``pattern`` is the rhythm's own fundamental.
    A metrical accent has an onset on every step, so its 2 Hz beat reference
    carries no stimulus energy and must not be read as entrainment; score it
    against ``bar``, ``onset``, and ``pattern`` instead.
    """
    freqs = {
        "bar": _BAR_FREQ,
        "beat": _BEAT_FREQ,
        "onset": float(rhythm.onset_rate),
        "pattern": float(rhythm.pulse_frequency),
    }
    # Duplicate frequencies would silently overwrite each other, so keep only
    # the first label for each distinct value.
    unique: dict[str, float] = {}
    seen: dict[float, str] = {}
    for name, value in freqs.items():
        if value <= 0:
            continue
        key = round(value, 6)
        if key in seen:
            continue
        seen[key] = name
        unique[name] = value
    return unique


def combined_frequencies(rhythm, background=None) -> dict[str, float]:
    """Reference frequencies of the stimulus actually presented.

    Unlike ``stimulus_frequencies``, this accounts for the background meter, so
    a dense meter's onset rate and slow envelope are both visible. Frequencies
    are derived from the summed per-step thalamic spikes, which is what the
    network receives.
    """
    if rhythm.n_steps != (background.n_steps if background is not None else rhythm.n_steps):
        raise ValueError("rhythm and background must share a step count")
    if background is None:
        return stimulus_frequencies(rhythm)

    n_steps = rhythm.n_steps
    on_steps = sum(stimuli.combined_pattern(rhythm, background))
    cycle_duration = n_steps * rhythm.step_duration
    candidate = {
        "bar": _BAR_FREQ,
        "beat": _BEAT_FREQ,
        "onset": on_steps / cycle_duration,
        "pattern": combined_fundamental_hz(rhythm, background),
    }

    unique: dict[str, float] = {}
    seen: set[float] = set()
    for name, value in candidate.items():
        if value <= 0:
            continue
        key = round(value, 6)
        if key in seen:
            continue
        seen.add(key)
        unique[name] = value
    return unique


def combined_fundamental_hz(rhythm, background=None) -> float:
    """Fundamental repetition rate of the summed per-step thalamic drive."""
    if background is None:
        return float(rhythm.pulse_frequency)
    spikes = [a + b for a, b in zip(rhythm.step_spikes, background.step_spikes)]
    period = stimuli.sequence_period(spikes)
    return 1.0 / (period * rhythm.step_duration)


def beat_reference_is_informative(rhythm, background=None) -> bool:
    """Whether the presented stimulus genuinely repeats at the beat rate.

    The 2 Hz reference is only meaningful if the stimulus' own fundamental is
    the beat rate. A metrical accent repeats at the bar, so phase locking to
    2 Hz there reflects the dense onset grid rather than beat entrainment.
    """
    try:
        fundamental = combined_fundamental_hz(rhythm, background)
    except ValueError:
        return False
    return abs(float(fundamental) - _BEAT_FREQ) < 1e-6


def _score(result, freqs: dict[str, float] | None = None) -> dict:
    """Return synchrony metrics for one simulated result.

    Phase metrics need enough cycles to be identifiable: at 2.5 Hz a 0.5 s
    window holds barely one cycle, so the estimate is degenerate. Frequencies
    with fewer than ``_MIN_CYCLES`` cycles over the recording return NaN rather
    than a spuriously high value.
    """
    fs = 1.0 / float(np.mean(np.diff(result.rate_t)))
    exc_mask = result.spike_indices < result.n_exc
    inh_mask = ~exc_mask
    exc_times = result.spike_times[exc_mask]
    exc_idx = result.spike_indices[exc_mask]
    inh_times = result.spike_times[inh_mask]
    inh_idx = result.spike_indices[inh_mask] - result.n_exc
    duration = result.params.duration / 1000.0

    def measurable(freq: float) -> bool:
        return freq > 0 and duration * freq >= _MIN_CYCLES

    def phase_metric(rate_exc, rate_inh, freq):
        if not measurable(freq):
            return float("nan"), float("nan")
        return (
            synchrony.plv(rate_exc, fs, freq),
            synchrony.plv(rate_inh, fs, freq),
        )

    def spike_metric(freq):
        if not measurable(freq):
            return float("nan"), float("nan")
        return (
            synchrony.mean_vector_strength(exc_times, exc_idx, result.n_exc, freq),
            synchrony.mean_vector_strength(inh_times, inh_idx, result.n_inh, freq),
        )

    scores = {}
    for name, freq in (("bar", _BAR_FREQ), ("beat", _BEAT_FREQ)):
        exc, inh = phase_metric(result.rate_exc, result.rate_inh, freq)
        scores[f"plv_exc_{name}"] = exc
        scores[f"plv_inh_{name}"] = inh
        exc, inh = spike_metric(freq)
        scores[f"vs_exc_{name}"] = exc
        scores[f"vs_inh_{name}"] = inh
    scores["corr_exc"] = synchrony.spike_pairwise_corr(
        exc_times, exc_idx, result.n_exc, duration
    )
    scores["corr_inh"] = synchrony.spike_pairwise_corr(
        inh_times, inh_idx, result.n_inh, duration
    )
    scores["rate_exc_hz"] = float(result.rate_exc.mean())
    scores["rate_inh_hz"] = float(result.rate_inh.mean())

    for name, freq in (freqs or {}).items():
        if name in ("bar", "beat"):
            continue
        exc, inh = phase_metric(result.rate_exc, result.rate_inh, freq)
        scores[f"plv_exc_{name}"] = exc
        scores[f"plv_inh_{name}"] = inh
        exc, inh = spike_metric(freq)
        scores[f"vs_exc_{name}"] = exc
        scores[f"vs_inh_{name}"] = inh
    return scores


def run_rhythm_sweep(
    duration_ms: float = 8000.0,
    dt: float = 0.1,
    bg_noise: float = 3.0,
    n_seeds: int = 3,
    base_seed: int = 42,
    background: bool = False,
    beat_rate_on: float = 20.0,
) -> list[dict]:
    """Simulate each rhythm and return one row of scores per rhythm."""
    rows = []


    # Baseline: no input, averaged over seeds
    base_rates_e, base_rates_i = [], []
    for k in range(n_seeds):
        params = model.Params(
            duration=duration_ms, dt=dt, seed=base_seed + k, bg_noise=bg_noise
        )
        result = model.simulate(params, stimulus=None)
        base_rates_e.append(float(result.rate_exc.mean()))
        base_rates_i.append(float(result.rate_inh.mean()))
    print(
        f"baseline (no input): r_e={float(np.mean(base_rates_e)):.2f}±{float(np.std(base_rates_e)):.2f} Hz, "
        f"r_i={float(np.mean(base_rates_i)):.2f}±{float(np.std(base_rates_i)):.2f} Hz"
    )

    # Baseline: steady on-beat background (beat alone), as a full row
    if background:
        base = stimuli.Rhythm(stimuli.ONBEAT_BACKGROUND, rate_on=beat_rate_on, name="beat-alone")
        base_scores = {key: [] for key in _METRIC_KEYS}
        base_freqs = stimulus_frequencies(base)
        for k in range(n_seeds):
            params = model.Params(
                duration=duration_ms, dt=dt, seed=base_seed + k, bg_noise=bg_noise
            )
            result = model.simulate(params, stimulus=base)
            for key, value in _score(result, base_freqs).items():
                base_scores.setdefault(key, []).append(value)
        rows.append({
            "pattern": 0,
            "name": "beat-alone",
            "lhl": syncopation.lhl(stimuli.ONBEAT_BACKGROUND),
            "offbeat_count": syncopation.offbeat_count(stimuli.ONBEAT_BACKGROUND),
            "wnbd": syncopation.wnbd(stimuli.ONBEAT_BACKGROUND),
            "freq": _BAR_FREQ,
            **{key: float(np.mean(base_scores[key])) for key in base_scores},
        })

    # Iterate through the off-beat gradient
    for pid in stimuli.OFFBEAT_GRADIENT:
        accent = stimuli.make_rhythm(pid)
        beat = (
            stimuli.Rhythm(stimuli.ONBEAT_BACKGROUND, rate_on=beat_rate_on, name="beat")
            if background
            else None
        )
        name = stimuli.PATTERN_NAMES[pid] + ("+beat" if background else "")
        scores = {key: [] for key in _METRIC_KEYS}
        freqs = stimulus_frequencies(accent)
        for k in range(n_seeds):
            params = model.Params(
                duration=duration_ms, dt=dt, seed=base_seed + k, bg_noise=bg_noise
            )
            result = model.simulate(params, stimulus=accent, background=beat)
            for key, value in _score(result, freqs).items():
                scores.setdefault(key, []).append(value)

        row = {
            "pattern": pid,
            "name": name,
            "lhl": syncopation.lhl(stimuli.PATTERNS[pid]),
            "offbeat_count": syncopation.offbeat_count(stimuli.PATTERNS[pid]),
            "wnbd": syncopation.wnbd(stimuli.PATTERNS[pid]),
            "freq": _BAR_FREQ,
            **{key: float(np.mean(scores[key])) for key in scores},
        }
        rows.append(row)
        print(
            f"  {row['pattern']:>2}  {row['name']:<18} "
            f"lhl={row['lhl']:>2}  off={row['offbeat_count']}  wnbd={row['wnbd']:.2f}  "
            f"f={row['freq']:.2f} Hz  "
            f"plv_bar={row['plv_exc_bar']:.3f}/{row['plv_inh_bar']:.3f}  "
            f"plv_beat={row['plv_exc_beat']:.3f}/{row['plv_inh_beat']:.3f}  "
            f"vs_bar={row['vs_exc_bar']:.3f}/{row['vs_inh_bar']:.3f}  "
            f"vs_beat={row['vs_exc_beat']:.3f}/{row['vs_inh_beat']:.3f}  "
            f"corr={row['corr_exc']:.3f}  "
            f"r_e={row['rate_exc_hz']:.2f} Hz"
        )
    return rows


# Backward compatible name used by earlier notebooks and scripts.
run_sweep = run_rhythm_sweep


# "Medium Entrainment" regime ("Level 2 - Medium" in
# notebooks/parameter_sets.json), translated to model.Params names. The
# preset's show_background setting is represented by the on-beat background
# passed to model.simulate. This is the operating point used by the
# sixteenth-note rhythm sweep and by experiments/syncopation_demo.py.
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

MEDIUM_16_STEP_DURATION = 0.125  # one sixteenth note at 120 BPM


def _thal_count(pset: dict) -> int:
    """
    With normalization on, the thalamic population scales with "n_exc" so the feedforward:recurrent ratio stays size invariant
    with normalization off, the reference size (n_thal_ref) is used 
    """
    if pset["synaptic_normalization"] != "none":
        return max(1, round(pset["n_thal_ref"] * pset["n_exc"] / pset["n_exc_ref"]))
    return int(pset["n_thal_ref"])


METERS = ("beat", "downbeat", "beats13", "accent")

_SPARSE_METERS = {
    "downbeat": stimuli.DOWNBEAT_16,
    "beats13": stimuli.BEATS_1_3_16,
}


def _describe_drive(meter: stimuli.Rhythm | None, stimulus_rate_on: float) -> str:
    """One-line summary of four-onset stimulus drive vs meter drive."""
    reference = stimuli.Rhythm(
        tuple(stimuli.PATTERNS_16[0]),
        step_duration=MEDIUM_16_STEP_DURATION,
        rate_on=stimulus_rate_on,
    )
    stimulus = reference.spikes_per_cycle
    if meter is None:
        return f"drive: stimulus {stimulus}/cycle, no meter"
    ratio = meter.spikes_per_cycle / stimulus if stimulus else float("inf")
    note = "  <- meter stronger than stimulus" if ratio > 1.0 else ""
    return (
        f"drive: stimulus {stimulus}/cycle, meter {meter.spikes_per_cycle}/cycle "
        f"(meter/stimulus {ratio:.2f}x), total {stimulus + meter.spikes_per_cycle}{note}"
    )


def _build_meter(
    meter: str = "beat",
    step_duration: float = MEDIUM_16_STEP_DURATION,
    rate_on: float = 40.0,
) -> stimuli.Rhythm | None:
    """Return the requested background meter, or None when disabled."""
    if meter == "none":
        return None
    if meter == "beat":
        return stimuli.Rhythm(
            tuple(stimuli.ONBEAT_BACKGROUND_16),
            step_duration=step_duration,
            rate_on=rate_on,
            name="beat",
        )
    if meter in _SPARSE_METERS:
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
                f"step_duration={step_duration:g} s, got {rate_on:g} Hz"
            )
        return stimuli.make_metrical_accent(
            step_duration=step_duration,
            rate_on=rate_on,
            name="meter",
        )
    raise ValueError(f"unknown meter {meter!r}; choose from {METERS} or 'none'")


def run_rhythm_sweep_16(
    duration_ms: float = 20000.0,
    dt: float = 0.1,
    bg_noise: float = 3.4,
    n_seeds: int = 3,
    base_seed: int = 42,
    background: bool = True,
    beat_rate_on: float | None = None,
    levels: tuple[int, ...] | None = None,
    meter: str = "beat",
    stimulus_rate_on: float = 40.0,
) -> list[dict]:
    """
    Simulate the 16th note rhythms under medium entrainment.

    ``beat_rate_on`` defaults to ``stimulus_rate_on``. When the accent meter is
    requested without an explicit meter rate, both channels move to the accent's
    recommended rate so the comparison stays drive-matched.
    """
    if levels is None:
        levels = tuple(stimuli.PATTERNS_16)
    levels = tuple(levels)
    unknown = [level for level in levels if level not in stimuli.PATTERNS_16]
    if unknown:
        raise KeyError(
            f"Unknown pattern ID(s) {unknown}. "
            f"Available: {list(stimuli.PATTERNS_16)}"
        )

    if (
        background
        and meter == "accent"
        and beat_rate_on is None
        and not stimuli.accent_is_representable(
            stimulus_rate_on, MEDIUM_16_STEP_DURATION
        )
    ):
        chosen = stimuli.METRICAL_ACCENT_DEFAULT_RATE_ON
        print(
            f"  accent meter: selected {chosen:g} Hz for stimulus and meter "
            "(drive-matched, best realised hierarchy)"
        )
        stimulus_rate_on = chosen
        beat_rate_on = chosen
    if beat_rate_on is None:
        beat_rate_on = stimulus_rate_on

    rows = []
    beat = _build_meter(
        meter, MEDIUM_16_STEP_DURATION, beat_rate_on
    ) if background else None
    print(f"  {_describe_drive(beat, stimulus_rate_on)}")
    for level in levels:
        pattern = stimuli.PATTERNS_16[level]
        name = stimuli.PATTERN_NAMES_16[level]
        accent = stimuli.Rhythm(
            tuple(pattern),
            step_duration=MEDIUM_16_STEP_DURATION,
            rate_on=stimulus_rate_on,
            rate_off=0.0,
            jitter_ms=1.0,
            name=name,
        )
        scores = {key: [] for key in _METRIC_KEYS}
        freqs = combined_frequencies(accent, beat)
        informative = beat_reference_is_informative(accent, beat)
        for k in range(n_seeds):
            params = model.Params(
                **MEDIUM_ENTRAINMENT_PARAMS,
                n_thal=_thal_count(MEDIUM_ENTRAINMENT_PARAMS),
                duration=duration_ms,
                dt=dt,
                seed=base_seed + k,
                bg_noise=bg_noise,
            )
            result = model.simulate(params, stimulus=accent, background=beat)
            for key, value in _score(result, freqs).items():
                scores.setdefault(key, []).append(value)
        scores = _nan_uninformative_beat(scores, informative)
        combined = stimuli.combined_pattern(accent, beat)
        row = {
            "pattern": level,
            "name": name,
            "lhl": syncopation.lhl(pattern),
            "offbeat_count": syncopation.offbeat_count(pattern),
            "wnbd": syncopation.wnbd(pattern),
            "freq": _BAR_FREQ,
            "meter": meter if background else "none",
            "stimulus_spikes_per_cycle": accent.spikes_per_cycle,
            "meter_spikes_per_cycle": (
                beat.spikes_per_cycle if beat is not None else 0
            ),
            "combined_onsets": "".join(str(value) for value in combined),
            "combined_n_onsets": int(sum(combined)),
            # Record the reference frequencies so each phase column is
            # self-describing; a 2 Hz beat reference is uninformative when the
            # stimulus has no 2 Hz energy (e.g. the metrical accent).
            **{f"ref_{name}_hz": value for name, value in freqs.items()},
            "beat_reference_informative": int(informative),
            **{key: float(np.mean(scores[key])) for key in scores},
        }
        rows.append(row)
        onset_note = (
            f"  plv_onset={row['plv_exc_onset']:.3f}"
            if "plv_exc_onset" in row
            else ""
        )
        beat_note = "" if informative else "  [beat ref not meaningful]"
        print(
            f"  {row['name']:<20} "
            f"lhl={row['lhl']:>2}  off={row['offbeat_count']}  wnbd={row['wnbd']:.2f}  "
            f"f={row['freq']:.2f} Hz  "
            f"plv_bar={row['plv_exc_bar']:.3f}/{row['plv_inh_bar']:.3f}  "
            f"plv_beat={row['plv_exc_beat']:.3f}/{row['plv_inh_beat']:.3f}"
            f"{onset_note}{beat_note}  "
            f"corr={row['corr_exc']:.3f}  "
            f"r_e={row['rate_exc_hz']:.2f} Hz"
        )
    return rows


# --------------------------------------------------------------------
# Parameter sweeps: stimulus weight x noise, and accent timing jitter

# Log spaced grids of  two axes that define the phase diagram.
PHASE_STIM_W = (0.02, 0.05, 0.1, 0.2, 0.4)
PHASE_NOISE = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)

# Accent timing jitter, and the operating points (stim_w, noise) at
# which it is swept: high, moderate and (nominally) no entrainment.
JITTER_MS_VALUES = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
JITTER_POINTS = ((0.2, 1.0), (0.1, 4.0), (0.05, 12.0))

PHASE_CSV = "phase_diagram_stim_w_noise.csv"
JITTER_CSV = "jitter_sweep.csv"
RHYTHM_CSV = "rhythm_sweep.csv"
RHYTHM_16_CSV = "rhythm_sweep_16.csv"

# The parameter sweep uses a simple 1 Hz accent modulation against a
# stable 2 Hz meter at the default 0.25 s step duration. Both frequencies
# are derived from the patterns below rather than hardcoded in result rows.
PHASE_ACCENT_PATTERN = 2
PHASE_METER_PATTERN = tuple(stimuli.ONBEAT_BACKGROUND)
PHASE_ACCENT_RATE_ON = 40.0
PHASE_METER_RATE_ON = 20.0

# Everything the parameter sweeps hold fixed. "stim_w" sets "w_thal"
# for the accent channel and "noise" sets "bg_noise"
# Every parameter accepted by "model.Params" is listed here explicitly
# except "w_thal" and "bg_noise" (the two sweep axes, set per cell in
# "_phase_params"). No "Params" class defaults are relied on
# implicitly, so the values used to produce "PHASE_CSV" are fully
# visible and reproducible.
PHASE_REFERENCE = dict(
    # Simulation
    duration=20000.0, dt=0.1, v_dt=1.0, seed=42,
    # Population sizes and synaptic-normalization reference sizes
    n_exc=3200, n_inh=800, n_thal=160,
    synaptic_normalization="sqrt",
    n_exc_ref=400, n_inh_ref=100, n_thal_ref=20,
    # Neuron dynamics
    C_m=200.0, tau_exc=20.0, tau_inh=10.0,
    E_l=-65.0, E_e=0.0, E_i=-80.0,
    v_thresh=-50.0, v_reset=-65.0,
    tau_ref_exc=2.0, tau_ref_inh=1.0,
    # Synaptic time constants and delays
    tau_e=5.0, tau_i=10.0,
    delay_ei=1.0, delay_ie=1.0,
    delay_ei_std=0.0, delay_ie_std=0.0,
    # Recurrent weights
    w_ee=1.5, w_ei=1.0, w_ie=3.0, w_ii=4.0,
    # Recurrent connection probabilities
    p_ee=0.02, p_ei=0.4, p_ie=0.5, p_ii=0.5,
    # Spike-frequency adaptation
    tau_adapt=150.0, w_adapt=100.0, delta_adapt=0.1,
    # Thalamic projection (weight and noise are the sweep axes)
    p_thal=0.4,
    # Tonic drive and shared weight spread
    I_bias=100.0, I_bias_std=7.5, weight_lognormal_sigma=0.0,
)

PHASE_STEP_DURATION = 0.25


def parameter_rhythms(
    jitter_ms: float = 0.0,
    step_duration: float = PHASE_STEP_DURATION,
    accent_rate_on: float = PHASE_ACCENT_RATE_ON,
    meter_rate_on: float = PHASE_METER_RATE_ON,
) -> tuple[stimuli.Rhythm, stimuli.Rhythm]:
    """Build the accent and meter rhythms used by parameter sweeps."""
    accent = stimuli.make_rhythm(
        PHASE_ACCENT_PATTERN,
        step_duration=step_duration,
        rate_on=accent_rate_on,
        rate_off=0.0,
        jitter_ms=jitter_ms,
    )
    meter = stimuli.Rhythm(
        PHASE_METER_PATTERN,
        step_duration=step_duration,
        rate_on=meter_rate_on,
        rate_off=0.0,
        name="meter",
    )
    return accent, meter


def _parameter_frequencies(
    accent: stimuli.Rhythm, meter: stimuli.Rhythm
) -> tuple[float, float]:
    """Return (accent_hz, meter_hz) derived from the actual patterns."""
    accent_hz = 1.0 / (
        stimuli.pattern_period(accent.pattern) * accent.step_duration
    )
    meter_hz = 1.0 / (
        stimuli.pattern_period(meter.pattern) * meter.step_duration
    )
    return accent_hz, meter_hz


def stimulus_rhythm(
    jitter_ms: float = 0.0,
    step_duration: float = PHASE_STEP_DURATION,
    rate_on: float = PHASE_ACCENT_RATE_ON,
) -> stimuli.Rhythm:
    """Backward compatible helper returning the parameter sweep accent."""
    accent, _meter = parameter_rhythms(jitter_ms, step_duration, rate_on)
    return accent


def _phase_params(
    stim_w: float,
    noise: float,
    reference: dict | None = None,
    duration_ms: float | None = None,
) -> model.Params:
    """Fixed reference parameters with the two sweep axes set"""
    values = dict(PHASE_REFERENCE if reference is None else reference)
    if duration_ms is not None:
        values["duration"] = duration_ms
    # Params has an explicit keyword signature, while this reference
    # dictionary intentionally supports overriding any fixed parameter.
    return model.Params(**cast(Any, values), w_thal=stim_w, bg_noise=noise)


def run_phase_diagram(
    stim_w_values=PHASE_STIM_W,
    noise_values=PHASE_NOISE,
    step_duration: float = PHASE_STEP_DURATION,
    accent_rate_on: float = PHASE_ACCENT_RATE_ON,
    meter_rate_on: float = PHASE_METER_RATE_ON,
    duration_ms: float | None = None,
    reference: dict | None = None,
) -> list[dict]:
    """Simulate the stimulus weight and noise grid"""
    stim_w_values = tuple(stim_w_values)
    noise_values = tuple(noise_values)
    accent, meter = parameter_rhythms(
        jitter_ms=0.0,
        step_duration=step_duration,
        accent_rate_on=accent_rate_on,
        meter_rate_on=meter_rate_on,
    )
    accent_hz, meter_hz = _parameter_frequencies(accent, meter)
    freqs = {"stim": accent_hz, "accent": accent_hz, "meter": meter_hz}
    rows = []
    total = len(stim_w_values) * len(noise_values)
    for n, (stim_w, noise) in enumerate(itertools.product(stim_w_values, noise_values), 1):
        params = _phase_params(stim_w, noise, reference, duration_ms)
        row = {
            "stim_w": stim_w,
            "noise": noise,
            "jitter_ms": 0.0,
            "stim_hz": accent_hz,
            "accent_hz": accent_hz,
            "meter_hz": meter_hz,
            **_score(
                model.simulate(params, stimulus=accent, background=meter),
                freqs,
            ),
        }
        rows.append(row)
        print(
            f"{n:>3}/{total}  stim_w={stim_w:<6g} noise={noise:<5g} "
            f"meter={row['plv_exc_meter']:.3f}  accent={row['plv_exc_accent']:.3f}  "
            f"corr={row['corr_exc']:.3f}  "
            f"r_e={row['rate_exc_hz']:.2f} Hz  r_i={row['rate_inh_hz']:.2f} Hz"
        )
    return rows


def run_jitter_sweep(
    points=JITTER_POINTS,
    jitter_values=JITTER_MS_VALUES,
    step_duration: float = PHASE_STEP_DURATION,
    accent_rate_on: float = PHASE_ACCENT_RATE_ON,
    meter_rate_on: float = PHASE_METER_RATE_ON,
    duration_ms: float | None = None,
    reference: dict | None = None,
) -> list[dict]:
    """Sweep accent jitter at each point of stimulus weight and noise"""
    points = tuple((float(w), float(n)) for w, n in points)
    jitter_values = tuple(jitter_values)
    rows = []
    total = len(points) * len(jitter_values)
    for group, (stim_w, noise) in enumerate(points, 1):
        print(f"\noperating point {group}/{len(points)}: stim_w={stim_w:g}, noise={noise:g}")
        for jitter_ms in jitter_values:
            accent, meter = parameter_rhythms(
                jitter_ms=jitter_ms,
                step_duration=step_duration,
                accent_rate_on=accent_rate_on,
                meter_rate_on=meter_rate_on,
            )
            accent_hz, meter_hz = _parameter_frequencies(accent, meter)
            params = _phase_params(stim_w, noise, reference, duration_ms)
            row = {
                "stim_w": stim_w,
                "noise": noise,
                "jitter_ms": jitter_ms,
                "stim_hz": accent_hz,
                "accent_hz": accent_hz,
                "meter_hz": meter_hz,
                **_score(
                    model.simulate(params, stimulus=accent, background=meter),
                    {"stim": accent_hz, "accent": accent_hz, "meter": meter_hz},
                ),
            }
            rows.append(row)
            n = len(rows)
            print(
                f"{n:>3}/{total}  jitter={jitter_ms:<5g} ms  "
                f"meter={row['plv_exc_meter']:.3f}  accent={row['plv_exc_accent']:.3f}  "
                f"corr={row['corr_exc']:.3f}  "
                f"r_e={row['rate_exc_hz']:.2f} Hz  r_i={row['rate_inh_hz']:.2f} Hz"
            )
    return rows


def run_parameter_sweep(
    stim_w_values=PHASE_STIM_W,
    noise_values=PHASE_NOISE,
    jitter_points=JITTER_POINTS,
    jitter_values=JITTER_MS_VALUES,
    step_duration: float = PHASE_STEP_DURATION,
    accent_rate_on: float = PHASE_ACCENT_RATE_ON,
    meter_rate_on: float = PHASE_METER_RATE_ON,
    duration_ms: float | None = None,
    reference: dict | None = None,
) -> dict[str, list[dict]]:
    """Run the phase diagram and jitter parameter sweeps"""
    return {
        "phase": run_phase_diagram(
            stim_w_values=stim_w_values,
            noise_values=noise_values,
            step_duration=step_duration,
            accent_rate_on=accent_rate_on,
            meter_rate_on=meter_rate_on,
            duration_ms=duration_ms,
            reference=reference,
        ),
        "jitter": run_jitter_sweep(
            points=jitter_points,
            jitter_values=jitter_values,
            step_duration=step_duration,
            accent_rate_on=accent_rate_on,
            meter_rate_on=meter_rate_on,
            duration_ms=duration_ms,
            reference=reference,
        ),
    }


def _save_csv(rows: list[dict], filename: str = "sweep.csv") -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    # Rows can carry different metric columns when their reference frequencies
    # differ, so build the schema from the union of keys in first-seen order.
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


def _run_rhythm(argv: list[str], filename: str = RHYTHM_CSV) -> None:
    duration_ms = float(argv[0]) if len(argv) > 0 else 8000.0
    bg_noise = float(argv[1]) if len(argv) > 1 else 3.0
    n_seeds = int(argv[2]) if len(argv) > 2 else 3
    background = bool(int(argv[3])) if len(argv) > 3 else False
    beat_rate_on = float(argv[4]) if len(argv) > 4 else 20.0
    print(
        f"Sweeping {len(stimuli.OFFBEAT_GRADIENT)} rhythms over {duration_ms:.0f} ms "
        f"(bg_noise={bg_noise} mV, {n_seeds} seeds, background={background}, "
        f"beat_rate_on={beat_rate_on} Hz) each..."
    )
    rows = run_rhythm_sweep(
        duration_ms=duration_ms, bg_noise=bg_noise, n_seeds=n_seeds,
        background=background, beat_rate_on=beat_rate_on,
    )
    path = _save_csv(rows, filename)
    print(f"\nSaved results to {path}")


def _run_rhythm_16(argv: list[str], filename: str = RHYTHM_16_CSV) -> None:
    """Run the sixteenth note sweep over LHL levels"""
    duration_ms = float(argv[0]) if len(argv) > 0 else 20000.0
    bg_noise = float(argv[1]) if len(argv) > 1 else 3.4
    n_seeds = int(argv[2]) if len(argv) > 2 else 3
    background = bool(int(argv[3])) if len(argv) > 3 else True
    if len(argv) > 4 and argv[4].lower() == "auto":
        beat_rate_on = None
    else:
        beat_rate_on = float(argv[4]) if len(argv) > 4 else None
    levels = (
        tuple(int(level) for level in argv[5].split(",")) if len(argv) > 5 else None
    )
    meter = argv[6] if len(argv) > 6 else "beat"
    stimulus_rate_on = float(argv[7]) if len(argv) > 7 else 40.0
    n_levels = len(levels) if levels is not None else len(stimuli.PATTERNS_16)
    meter_rate_label = "auto" if beat_rate_on is None else f"{beat_rate_on:g} Hz"
    print(
        f"Sweeping {n_levels} sixteenth-note LHL rhythms over {duration_ms:.0f} ms "
        f"(medium entrainment, bg_noise={bg_noise} mV, {n_seeds} seeds, "
        f"background={background}, meter={meter}, beat_rate_on={meter_rate_label}, "
        f"stimulus_rate_on={stimulus_rate_on} Hz) each..."
    )
    rows = run_rhythm_sweep_16(
        duration_ms=duration_ms, bg_noise=bg_noise, n_seeds=n_seeds,
        background=background, beat_rate_on=beat_rate_on, levels=levels,
        meter=meter, stimulus_rate_on=stimulus_rate_on,
    )
    path = _save_csv(rows, filename)
    print(f"\nSaved results to {path}")


def _run_parameters() -> None:
    n_cells = len(PHASE_STIM_W) * len(PHASE_NOISE)
    n_jitter = len(JITTER_POINTS) * len(JITTER_MS_VALUES)
    print(
        f"Parameter sweep: {n_cells} stim_w/noise runs plus "
        f"{n_jitter} accent-jitter runs; "
        f"accent pattern={PHASE_ACCENT_PATTERN} at {PHASE_ACCENT_RATE_ON:g} Hz, "
        f"meter={PHASE_METER_RATE_ON:g} Hz..."
    )
    tables = run_parameter_sweep()
    print(f"\nSaved results to {_save_csv(tables['phase'], PHASE_CSV)}")
    print(f"Saved results to {_save_csv(tables['jitter'], JITTER_CSV)}")


_SWEEP_MODES = ("rhythms", "rhythms16", "parameters", "offbeat", "phase", "jitter", "all")


def main() -> None:
    argv = sys.argv[1:]
    mode = argv[0].lower() if argv and argv[0].lower() in _SWEEP_MODES else "rhythms"
    if argv and argv[0].lower() in _SWEEP_MODES:
        argv = argv[1:]

    if mode in ("rhythms", "offbeat"):
        # Keep the old no-mode output filename for existing analysis scripts.
        filename = "sweep.csv" if mode == "offbeat" else RHYTHM_CSV
        _run_rhythm(argv, filename)
        return

    if mode == "rhythms16":
        _run_rhythm_16(argv)
        return

    if mode == "parameters":
        _run_parameters()
        return

    if mode == "phase":
        print(f"Saved results to {_save_csv(run_phase_diagram(), PHASE_CSV)}")
        return

    if mode == "jitter":
        print(f"Saved results to {_save_csv(run_jitter_sweep(), JITTER_CSV)}")
        return

    if mode == "all":
        _run_rhythm([], RHYTHM_CSV)
        _run_parameters()


if __name__ == "__main__":
    main()
