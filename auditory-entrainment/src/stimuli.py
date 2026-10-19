"""Stimulus generation utilities for auditory entrainment.

Rhythmic stimuli are binary onset patterns over a fixed number of equal
steps per cycle. A ``1`` marks a step containing a beat and a ``0``
marks a silent step. Patterns repeat for the full duration of a trial
and are converted into deterministic spike trains for the model's input
population, or into a time-varying firing rate for plotting and
analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import brian2 as b2 # type: ignore
import numpy as np # type: ignore

__all__ = [
    "PATTERNS",
    "PATTERN_NAMES",
    "SYNCOPATION_GRADIENT",
    "OFFBEAT_GRADIENT",
    "ONBEAT_BACKGROUND",
    "Rhythm",
    "make_rhythm",
    "pattern_period",
    "pattern_rate",
    "spike_trains",
    "spike_generator_group",
    "NO_INPUT",
    "SILENCE",
    "RHYTHMIC",
    "SYNCOPATED",
]

# Binary onset patterns. 1 = beat present, 0 = silent step.
PATTERNS = {
    1: [1, 0, 0, 0, 0, 0, 0, 0],
    2: [1, 0, 0, 0, 1, 0, 0, 0],
    3: [1, 0, 1, 0, 1, 0, 1, 0],
    4: [1, 0, 0, 1, 0, 0, 1, 0],
    5: [0, 1, 0, 1, 0, 1, 0, 1],
}

PATTERN_NAMES = {
    1: "single",  # one downbeat per cycle
    2: "onbeat",  # quarter-note pulse on the downbeats
    3: "eighths",  # straight eighth notes
    4: "tresillo",  # 3-3-2 syncopation, onsets 0, 3, 6
    5: "offbeat",  # every off-beat eighth, onsets 1, 3, 5, 7
}


# The reduced pattern set used by the sweep experiments.
SYNCOPATION_GRADIENT = (4, 5)
OFFBEAT_GRADIENT = (1, 2, 3, 4, 5)


# On-beat metrical background: a quarter-note pulse (the tactus) on the
# four beats of a 4/4 bar (positions 0, 2, 4, 6). Presented on a
# separate input channel so syncopated accents are heard against a
# steady beat, without changing the accent pattern's duty cycle.
ONBEAT_BACKGROUND = (1, 0, 1, 0, 1, 0, 1, 0)


@dataclass(frozen=True)
class Rhythm:
    """
    A repeating onset pattern mapped onto time.

    Attributes
    ----------
    pattern : tuple of int
        Binary pulse where 1 is on and 0 is off
    step_duration : float
        Duration of one step in seconds
    rate_on : float
        Mean firing rate (Hz) of each input neuron during an onset step
    rate_off : float
        Mean firing rate (Hz) of each input neuron during a silent step
    jitter_ms : float
        Standard deviation of an independent onset time offset for each
        neuron and active pattern step, in milliseconds
    name : str
        Label
    """

    pattern: tuple[int, ...]
    step_duration: float = 0.25
    rate_on: float = 35.0
    rate_off: float = 0.0
    jitter_ms: float = 0.0
    name: str = "rhythm"

    @property
    def n_steps(self) -> int:
        """Number of steps in one cycle."""
        return len(self.pattern)

    @property
    def cycle_duration(self) -> float:
        """Duration of one full cycle in seconds."""
        return self.n_steps * self.step_duration

    @property
    def pulse_frequency(self) -> float:
        """Fundamental frequency in Hz of the repeating onset pattern."""
        return 1.0 / (pattern_period(self.pattern) * self.step_duration)

    def rate(self, t):
        """Firing rate in Hz at time(s) ``t`` in seconds."""
        return pattern_rate(self.pattern, t, self.step_duration, self.rate_on, self.rate_off)

    def spike_trains(self, duration: float, n_neurons: int = 100, seed=0):
        """Spike times and neuron indices over ``duration`` seconds."""
        return spike_trains(self, duration, n_neurons=n_neurons, seed=seed)

    def spike_generator_group(self, duration: float, n_neurons: int = 100, seed=0):
        """Brian2 SpikeGeneratorGroup firing according to this rhythm."""
        return spike_generator_group(self, duration, n_neurons=n_neurons, seed=seed)


def make_rhythm(
    pattern_id, step_duration=0.25, rate_on=40.0, rate_off=0.0, jitter_ms=0.0
) -> Rhythm:
    """Build a Rhythm from a ``PATTERNS`` key."""
    return Rhythm(
        tuple(PATTERNS[pattern_id]),
        step_duration=step_duration,
        rate_on=rate_on,
        rate_off=rate_off,
        jitter_ms=jitter_ms,
        name=PATTERN_NAMES[pattern_id],
    )


def pattern_period(pattern) -> int:
    """Smallest cyclic period of a binary onset pattern, in steps."""
    pattern = [int(x) for x in pattern]
    n = len(pattern)
    for d in range(1, n + 1):
        if n % d == 0 and all(pattern[i] == pattern[i - d] for i in range(d, n)):
            return d
    return n


def pattern_rate(pattern, t, step_duration=0.25, rate_on=40.0, rate_off=0.0):
    """Firing rate in Hz of an onset pattern at times ``t`` in seconds."""
    pattern = np.asarray(pattern, dtype=np.int64)
    n_steps = len(pattern)
    t = np.asarray(t, dtype=float)
    step_idx = np.floor(t / step_duration).astype(np.int64) % n_steps
    return np.where(pattern[step_idx] == 1, rate_on, rate_off)


def spike_trains(rhythm, duration, n_neurons=100, seed=0):
    """Generate spike trains for ``n_neurons`` input neurons.

    Every neuron emits a regular spike train at ``rate_on`` Hz during
    onset steps and ``rate_off`` Hz during silent steps. If ``jitter_ms``
    is nonzero, each neuron receives an independent onset-time offset for
    each active step.

    Returns
    times : ndarray
        Spike times in seconds.
    indices : ndarray of int
        Neuron index of each spike in ``times``.
    """
    if rhythm.jitter_ms < 0:
        raise ValueError("jitter_ms must be non-negative.")
    rng = np.random.default_rng(seed)
    all_times = []
    all_indices = []
    for neuron_idx in range(n_neurons):
        times = _pattern_times(
            np.asarray(rhythm.pattern, dtype=np.int64),
            rhythm.step_duration,
            duration,
            rhythm.rate_on,
            rhythm.rate_off,
            jitter=rhythm.jitter_ms / 1000.0,
            rng=rng,
        )
        all_times.append(times)
        all_indices.append(np.full(times.size, neuron_idx, dtype=np.int32))
    if not all_times:
        return np.empty(0), np.empty(0, dtype=np.int32)
    return np.concatenate(all_times), np.concatenate(all_indices)


def spike_generator_group(rhythm, duration, n_neurons=100, seed=0):
    """Brian2 SpikeGeneratorGroup firing according to ``rhythm``."""
    times, indices = spike_trains(rhythm, duration, n_neurons=n_neurons, seed=seed)
    return b2.SpikeGeneratorGroup(n_neurons, indices, times * b2.second)


def _pattern_times(pattern, step_duration, duration, rate_on, rate_off, jitter=0.0, rng=None):
    """Regularly spaced spike times following a step-wise constant rate."""
    events = []
    step_idx = 0
    t_start = 0.0
    while t_start < duration:
        rate = rate_on if pattern[step_idx % len(pattern)] else rate_off
        t_end = min(t_start + step_duration, duration)
        if rate > 0.0:
            offset = rng.normal(0.0, jitter) if rng is not None else 0.0
            times = np.arange(t_start + offset, t_end + offset, 1.0 / rate)
            events.append(times[(times >= 0.0) & (times < duration)])
        t_start = t_end
        step_idx += 1
    if events:
        return np.concatenate(events)
    return np.empty(0)


# Presets: no input, a rhythmic on-beat pulse, and the syncopated variants.
SILENCE = Rhythm((0,) * 8, rate_on=0.0, rate_off=0.0, name="silence")
NO_INPUT = SILENCE
RHYTHMIC = make_rhythm(2)
SYNCOPATED = {pid: make_rhythm(pid) for pid in (4, 5)}
