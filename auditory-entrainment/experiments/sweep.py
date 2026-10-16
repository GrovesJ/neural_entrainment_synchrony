"""Sweep every rhythm and score syncopation and synchrony.

For each rhythm in ``stimuli.OFFBEAT_GRADIENT`` (three onsets per bar,
ordered by how many accents fall off the beat) the network is simulated
with Gaussian background noise and the scores are averaged over seeds.

All phase measures are computed at two fixed reference frequencies: the
bar rate (0.5 Hz) and the beat rate (2 Hz, the quarter-note tactus),
where on- vs off-beat effects appear. With ``background=True`` a steady
on-beat pulse is presented on a separate input channel, and a beat-alone
baseline row is added.

- syncopation: Longuet-Higgins & Lee score, off-beat accent count, and
  Weighted Note-to-Beat Distance (WNBD) of the onset pattern;
- synchrony: phase-locking value (PLV) and vector strength (VS) at 0.5
  and 2 Hz, the Golomb-Hansel chi measure, and the mean pairwise
  spike-count correlation in 5 ms bins.

The module exposes two independent experiment families:

- ``run_rhythm_sweep``: compare the different rhythm patterns while
  holding the network parameters fixed;
- ``run_parameter_sweep``: run the stimulus-weight/noise phase diagram
  and the separate jitter sweep while holding the rhythm fixed.

Usage:

    python experiments/sweep.py rhythms [duration_ms] [noise] [n_seeds] [background] [beat_rate_on]
    python experiments/sweep.py parameters
    python experiments/sweep.py all

The legacy ``offbeat`` mode and positional invocation are retained as
aliases for the rhythm sweep.
"""

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

_METRIC_KEYS = (
    "plv_exc_bar", "plv_inh_bar", "plv_exc_beat", "plv_inh_beat",
    "vs_exc_bar", "vs_inh_bar", "vs_exc_beat", "vs_inh_beat",
    "chi_exc", "chi_inh",
    "corr_exc", "corr_inh",
    "rate_exc_hz", "rate_inh_hz",
)


def _score(result, freqs: dict[str, float] | None = None) -> dict:
    """All synchrony metrics for one simulated result, as a dict.

    ``freqs`` optionally adds an extra PLV/VS pair per reference
    frequency, keyed by name: ``{"stim": 1.0}`` adds ``plv_exc_stim``,
    ``plv_inh_stim``, ``vs_exc_stim`` and ``vs_inh_stim``. The fixed bar
    and beat rates are always scored, so sweeps whose drive frequency is
    neither can still measure locking to the stimulus.
    """
    fs = 1.0 / float(np.mean(np.diff(result.rate_t)))
    exc_mask = result.spike_indices < result.n_exc
    inh_mask = ~exc_mask
    exc_times = result.spike_times[exc_mask]
    exc_idx = result.spike_indices[exc_mask]
    inh_times = result.spike_times[inh_mask]
    inh_idx = result.spike_indices[inh_mask] - result.n_exc
    duration = result.params.duration / 1000.0
    scores = {
        "plv_exc_bar": synchrony.plv(result.rate_exc, fs, _BAR_FREQ),
        "plv_inh_bar": synchrony.plv(result.rate_inh, fs, _BAR_FREQ),
        "plv_exc_beat": synchrony.plv(result.rate_exc, fs, _BEAT_FREQ),
        "plv_inh_beat": synchrony.plv(result.rate_inh, fs, _BEAT_FREQ),
        "vs_exc_bar": synchrony.mean_vector_strength(exc_times, exc_idx, result.n_exc, _BAR_FREQ),
        "vs_inh_bar": synchrony.mean_vector_strength(inh_times, inh_idx, result.n_inh, _BAR_FREQ),
        "vs_exc_beat": synchrony.mean_vector_strength(exc_times, exc_idx, result.n_exc, _BEAT_FREQ),
        "vs_inh_beat": synchrony.mean_vector_strength(inh_times, inh_idx, result.n_inh, _BEAT_FREQ),
        "chi_exc": synchrony.golomb_hansel(result.v_exc),
        "chi_inh": synchrony.golomb_hansel(result.v_inh),
        "corr_exc": synchrony.spike_pairwise_corr(exc_times, exc_idx, result.n_exc, duration),
        "corr_inh": synchrony.spike_pairwise_corr(inh_times, inh_idx, result.n_inh, duration),
        "rate_exc_hz": float(result.rate_exc.mean()),
        "rate_inh_hz": float(result.rate_inh.mean()),
    }
    for name, freq in (freqs or {}).items():
        scores[f"plv_exc_{name}"] = synchrony.plv(result.rate_exc, fs, freq)
        scores[f"plv_inh_{name}"] = synchrony.plv(result.rate_inh, fs, freq)
        scores[f"vs_exc_{name}"] = synchrony.mean_vector_strength(
            exc_times, exc_idx, result.n_exc, freq
        )
        scores[f"vs_inh_{name}"] = synchrony.mean_vector_strength(
            inh_times, inh_idx, result.n_inh, freq
        )
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
    """Simulate every rhythm and return one row of scores per rhythm.

    ``duration_ms`` is the simulation length per rhythm in milliseconds.
    ``bg_noise`` is the Gaussian noise level in mV (0 = deterministic).
    ``n_seeds`` independent seeds are averaged to stabilise the scores.
    ``background`` presents an on-beat quarter-note pulse on a separate
    input channel so syncopated accents are heard against a steady
    metrical beat, and adds a beat-alone baseline row for comparison.
    ``beat_rate_on`` is the firing rate (Hz) of the background beat
    channel during an onset step, so the beat can be made subtler than
    the accents (which use ``make_rhythm``'s 40 Hz).
    """
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
        for k in range(n_seeds):
            params = model.Params(
                duration=duration_ms, dt=dt, seed=base_seed + k, bg_noise=bg_noise
            )
            result = model.simulate(params, stimulus=base)
            for key, value in _score(result).items():
                base_scores[key].append(value)
        rows.append({
            "pattern": 0,
            "name": "beat-alone",
            "lhl": syncopation.lhl(stimuli.ONBEAT_BACKGROUND),
            "offbeat_count": syncopation.offbeat_count(stimuli.ONBEAT_BACKGROUND),
            "wnbd": syncopation.wnbd(stimuli.ONBEAT_BACKGROUND),
            "freq": _BAR_FREQ,
            **{key: float(np.mean(base_scores[key])) for key in _METRIC_KEYS},
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
        for k in range(n_seeds):
            params = model.Params(
                duration=duration_ms, dt=dt, seed=base_seed + k, bg_noise=bg_noise
            )
            result = model.simulate(params, stimulus=accent, background=beat)
            for key, value in _score(result).items():
                scores[key].append(value)

        row = {
            "pattern": pid,
            "name": name,
            "lhl": syncopation.lhl(stimuli.PATTERNS[pid]),
            "offbeat_count": syncopation.offbeat_count(stimuli.PATTERNS[pid]),
            "wnbd": syncopation.wnbd(stimuli.PATTERNS[pid]),
            "freq": _BAR_FREQ,
            **{key: float(np.mean(scores[key])) for key in _METRIC_KEYS},
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
            f"chi={row['chi_exc']:.3f}  corr={row['corr_exc']:.3f}  "
            f"r_e={row['rate_exc_hz']:.2f} Hz"
        )
    return rows


# Backward-compatible name used by earlier notebooks and scripts.
run_sweep = run_rhythm_sweep


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

# The parameter sweep uses a simple 1 Hz accent modulation against a
# stable 2 Hz meter at the default 0.25 s step duration. Both frequencies
# are derived from the patterns below rather than hardcoded in result rows.
PHASE_ACCENT_PATTERN = 2
PHASE_METER_PATTERN = tuple(stimuli.ONBEAT_BACKGROUND)
PHASE_ACCENT_RATE_ON = 40.0
PHASE_METER_RATE_ON = 20.0

# Everything the parameter sweeps hold fixed. ``stim_w`` sets ``w_thal``
# for the accent channel and ``noise`` sets ``bg_noise``; the meter is
# delivered through the model's separate background channel.
#
# Every parameter accepted by ``model.Params`` is listed here explicitly
# except ``w_thal`` and ``bg_noise`` (the two sweep axes, set per cell in
# ``_phase_params``). No ``Params`` class defaults are relied on
# implicitly, so the values used to produce ``PHASE_CSV`` are fully
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
    """Build the accent and stable meter used by parameter sweeps.

    Jitter is applied to the accent only. With the default step duration,
    the accent pattern has a 1 Hz period and the meter has a 2 Hz period.
    Their overlap produces a 2 Hz pulse train with a 1 Hz amplitude
    modulation.
    """
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
    """Return ``(accent_hz, meter_hz)`` derived from the actual patterns."""
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
    """Backward-compatible helper returning the parameter-sweep accent."""
    accent, _meter = parameter_rhythms(jitter_ms, step_duration, rate_on)
    return accent


def _phase_params(
    stim_w: float,
    noise: float,
    reference: dict | None = None,
    duration_ms: float | None = None,
) -> model.Params:
    """Fixed reference parameters with the two sweep axes set."""
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
    """Simulate the ``stim_w`` x ``noise`` grid; one row per cell.

    Only the stimulus weight and the background noise vary; every other
    parameter comes from ``reference`` (``PHASE_REFERENCE`` by default),
    so the grid isolates drive against noise. ``duration_ms`` overrides
    the reference duration for shorter pilot runs, and ``jitter_ms``
    The accent is clean in the phase diagram; the meter is supplied on
    the model's separate background channel.
    """
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
    """Sweep stimulus jitter at each operating point in ``points``.

    ``points`` are ``(stim_w, noise)`` pairs sampled from the phase
    diagram; one row per (point, jitter) pair. Jitter is applied only to
    the accent rhythm. The meter remains stable at every jitter level.
    """
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
    """Run the parameter experiments and return both result tables.

    The primary table is the ``stim_w`` x ``noise`` phase diagram. The
    secondary table sweeps accent ``jitter_ms`` at fixed operating points.
    Every row contains the same stable meter channel and fixed accent
    pattern, apart from the requested accent jitter.
    """
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
    fieldnames = list(rows[0])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


_SWEEP_MODES = ("rhythms", "parameters", "offbeat", "phase", "jitter", "all")


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
