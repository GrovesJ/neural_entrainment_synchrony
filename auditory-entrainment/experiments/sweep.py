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
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

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


def _score(result) -> dict:
    """All synchrony metrics for one simulated result, as a dict."""
    fs = 1.0 / float(np.mean(np.diff(result.rate_t)))
    exc_mask = result.spike_indices < result.n_exc
    inh_mask = ~exc_mask
    exc_times = result.spike_times[exc_mask]
    exc_idx = result.spike_indices[exc_mask]
    inh_times = result.spike_times[inh_mask]
    inh_idx = result.spike_indices[inh_mask] - result.n_exc
    duration = result.params.duration / 1000.0
    return {
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


def run_sweep(
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


def _save_csv(rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "sweep.csv"
    fieldnames = list(rows[0])
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    duration_ms = float(sys.argv[1]) if len(sys.argv) > 1 else 8000.0
    bg_noise = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    background = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
    beat_rate_on = float(sys.argv[5]) if len(sys.argv) > 5 else 20.0
    print(
        f"Sweeping {len(stimuli.OFFBEAT_GRADIENT)} rhythms over {duration_ms:.0f} ms "
        f"(bg_noise={bg_noise} mV, {n_seeds} seeds, background={background}, "
        f"beat_rate_on={beat_rate_on} Hz) each..."
    )
    rows = run_sweep(
        duration_ms=duration_ms, bg_noise=bg_noise, n_seeds=n_seeds,
        background=background, beat_rate_on=beat_rate_on,
    )
    path = _save_csv(rows)
    print(f"\nSaved results to {path}")


if __name__ == "__main__":
    main()
