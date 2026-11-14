"""
Stimulus generation utilities for auditory entrainment

Rhythmic stimuli are binary onset patterns over a fixed number of equal
steps per cycle.

- "1" marks a step containing a beat 
- "0" marks a silent step.
- Patterns repeat for the full duration of a trial
- Patterns are converted into deterministic spike trains for the models input population
- Steps may carry an optional weight that scales the onset rate for that step
  (see "make_metrical_accent")
"""

from __future__ import annotations

from dataclasses import dataclass

import brian2 as b2 # type: ignore
import numpy as np # type: ignore

__all__ = [
    "PATTERNS",
    "PATTERN_NAMES",
    "PATTERNS_16",
    "PATTERN_NAMES_16",
    "SYNCOPATION_GRADIENT",
    "OFFBEAT_GRADIENT",
    "ONBEAT_BACKGROUND",
    "ONBEAT_BACKGROUND_16",
    "DOWNBEAT_16",
    "BEATS_1_3_16",
    "combined_pattern",
    "sequence_period",
    "METRICAL_WEIGHTS_16",
    "METRICAL_ACCENT_REFERENCE_ONSETS",
    "Rhythm",
    "make_rhythm",
    "make_rhythm_16",
    "make_metrical_accent",
    "minimum_accent_rate",
    "METRICAL_ACCENT_DEFAULT_RATE_ON",
    "required_accent_spikes",
    "accent_is_representable",
    "make_uniform_grid",
    "make_accented_onbeat",
    "stimulus_audit",
    "allocate_spikes",
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

# Sixteenth note patterns with 4 onsets per bar
# Ordered by offbeat count 
PATTERNS_16 = {
    0: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # all on beat
    1: [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],  # 1 offbeat
    2: [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 2 offbeats
    3: [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  # 3 offbeats
    4: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # 4 offbeats
}

PATTERN_NAMES_16 = {
    0: "onbeat",
    1: "offbeat-1",
    2: "offbeat-2",
    3: "offbeat-3",
    4: "offbeat-4",
}



# The set of patterns used in the syncopation demo
SYNCOPATION_GRADIENT = (4, 5)
OFFBEAT_GRADIENT = (1, 2, 3, 4, 5)


# First is 8th note pulse (on beat) second is the 16th note version of the same pulse
ONBEAT_BACKGROUND = (1, 0, 1, 0, 1, 0, 1, 0)
ONBEAT_BACKGROUND_16 = (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)

# Sparse metrical backgrounds for the 16-step grid.
# A dense background is an onset every step, so ``rhythm | background`` collapses
# to an all-ones pattern and the rhythm's syncopation disappears. These sparse
# variants supply metrical context while leaving the rhythm's onsets visible.
DOWNBEAT_16 = (1,) + (0,) * 15
BEATS_1_3_16 = (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)


# Metrical hierarchy of one 4/4 bar sampled at 16 steps
# (Lerdahl & Jackendoff, 1983). Stronger positions take larger weights:
# position 0 is the downbeat, then positions 8, 4/12, 2/6/10/14 and finally
# the odd sixteenths. This is the scoring ladder used by Longuet-Higgins &
# Lee (1984), expressed with stronger = larger.
METRICAL_WEIGHTS_16 = (
    5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1,
)

# Onset count of the equal-density reference pattern used to match the total
# stimulus drive of the metrical accent stimulus.
METRICAL_ACCENT_REFERENCE_ONSETS = 4

# Recommended rate for the metrical accent. Drive matching holds at any rate
# above the representability floor (``minimum_accent_rate``), but integer spike
# counts make the realised hierarchy closer to 5:4:3:2:1 as the rate rises:
# max deviation 0.133 at 62 Hz, 0.086 at 80 Hz, 0.075 at 100 Hz. 100 Hz is used
# as the default because it improves the ratio without a large drive increase.
METRICAL_ACCENT_DEFAULT_RATE_ON = 100.0


@dataclass(frozen=True)
class Rhythm:
    """A class for rhythmnic stimulus with binary onset patterns"""

    pattern: tuple[int, ...]
    step_duration: float = 0.25
    rate_on: float = 35.0
    rate_off: float = 0.0
    jitter_ms: float = 0.0
    name: str = "rhythm"
    step_weights: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.step_weights is None:
            return
        _validate_step_weights(self.step_weights, self.n_steps)

    @property
    def n_steps(self) -> int:
        """Number of steps in one cycle."""
        return len(self.pattern)

    @property
    def cycle_duration(self) -> float:
        """Duration of a full cycle in seconds"""
        return self.n_steps * self.step_duration

    @property
    def pulse_frequency(self) -> float:
        """Fundamental frequency of the pattern.

        Metrical accent stimuli repeat over the whole cycle even though every
        step carries an onset, so the period is taken over both the onset
        pattern and the step weights.
        """
        if self.step_weights is None:
            period_steps = pattern_period(self.pattern)
        else:
            tokens = list(zip(self.pattern, self.step_weights))
            period_steps = _sequence_period(tokens)
        return 1.0 / (period_steps * self.step_duration)

    @property
    def onset_rate(self) -> float:
        """Onset events per second (independent of how many spikes each emits)."""
        return float(np.count_nonzero(self.pattern)) / self.cycle_duration

    @property
    def step_spikes(self) -> tuple[int, ...]:
        """Realised thalamic spikes per step, per input channel, per cycle."""
        if self.step_weights is None:
            rates = np.full(self.n_steps, float(self.rate_on))
        else:
            rates = np.asarray(self.step_weights, dtype=float) * float(self.rate_on)
        counts = np.ceil(float(self.step_duration) * rates - 1e-9).astype(int)
        pattern = np.asarray(self.pattern, dtype=np.int64)
        return tuple(
            int(count) if onset else 0 for count, onset in zip(counts, pattern)
        )

    @property
    def spikes_per_cycle(self) -> int:
        """Total thalamic spikes per cycle, per input channel."""
        return int(sum(self.step_spikes))

    @property
    def accent_ratio(self) -> float | None:
        """Peak-to-trough ratio of realised spikes, or None for flat rhythms."""
        if self.step_weights is None:
            return None
        counts = [count for count in self.step_spikes if count > 0]
        return max(counts) / min(counts) if counts else None

    def rate(self, t):
        """Firing rate in Hz at time t in seconds."""
        return pattern_rate(
            self.pattern,
            t,
            self.step_duration,
            self.rate_on,
            self.rate_off,
            step_weights=self.step_weights,
        )

    def spike_trains(self, duration: float, n_neurons: int = 100, seed=0):
        """Spike times and neuron indices over "duration" seconds"""
        return spike_trains(self, duration, n_neurons=n_neurons, seed=seed)

    def spike_generator_group(self, duration: float, n_neurons: int = 100, seed=0):
        """Brian2 SpikeGeneratorGroup firing according to this rhythm"""
        return spike_generator_group(self, duration, n_neurons=n_neurons, seed=seed)


def make_rhythm(
    pattern_id, step_duration=0.25, rate_on=40.0, rate_off=0.0, jitter_ms=0.0
) -> Rhythm:
    """Make a rhythm from one of the 8th note patterns"""
    return Rhythm(
        tuple(PATTERNS[pattern_id]),
        step_duration=step_duration,
        rate_on=rate_on,
        rate_off=rate_off,
        jitter_ms=jitter_ms,
        name=PATTERN_NAMES[pattern_id],
    )


def make_rhythm_16(
    pattern_id, step_duration=0.125, rate_on=40.0, rate_off=0.0, jitter_ms=0.0
) -> Rhythm:
    """
    Make a 16th note Rhythm from a "PATTERNS_16" key.
    step_duration defaults to 0.125 s so 16 steps = 2 seconds per cycle (same as 8 steps of 0.25s)
    """
    return Rhythm(
        tuple(PATTERNS_16[pattern_id]),
        step_duration=step_duration,
        rate_on=rate_on,
        rate_off=rate_off,
        jitter_ms=jitter_ms,
        name=PATTERN_NAMES_16[pattern_id],
    )





def make_metrical_accent(
    step_duration=0.125,
    rate_on=METRICAL_ACCENT_DEFAULT_RATE_ON,
    reference_onsets=METRICAL_ACCENT_REFERENCE_ONSETS,
    name="metrical-accent",
) -> Rhythm:
    """Metrically weighted accent stimulus.

    Every step carries an onset, but the thalamic drive scales with the
    metrical weight of that step, preserving the 5:4:3:2:1 hierarchy. The
    per-step spike counts are allocated so the number of emitted thalamic
    spikes per cycle matches an equal-duration stimulus with
    ``reference_onsets`` onsets at ``rate_on`` (the four-onset binary
    patterns), which controls for total stimulus drive.

    Representability
    ----------------
    The counts are allocated proportionally over the weights, which sum to
    ``5 + 4 + 2*3 + 4*2 + 8*1 = 31``. So a 31 spike budget is the smallest one
    that can give the weight-5 step 5 spikes, the weight-4 step 4, and so on.

    With ``reference_onsets=4`` and ``step_duration=0.125`` the budget is
    ``4 * ceil(0.125 * rate_on)``, so ``ceil(0.125 * rate_on) >= 8``, i.e.
    ``rate_on > 56 Hz``. The bound is exclusive because of the ``ceil``: at
    exactly 56 Hz only 28 spikes are available.

    Below that budget the allocation compresses rather than failing outright.
    At 24 spikes (about 40.1 Hz) every step still receives at least one spike,
    but the realised profile is 4:1 instead of 5:1. At 20 spikes (40 Hz) one
    weight-1 step rounds to zero and becomes a rest, which changes the
    stimulus class. The check below therefore raises below 31 spikes instead of
    silently emitting a compressed or sparse pattern; callers that genuinely
    want a compressed accent can build the ``Rhythm`` directly with
    ``step_weights``.

    ``rate_on`` is a property of the thalamic input channels, not of the
    cortical population: ``n_thal`` replicates the same train, so a larger
    network does not improve this resolution.

    Even above the threshold the realised ratio is approximate (about 8:7:5:3:2
    at 100 Hz) because integer spike counts cannot hit exact proportions. Use
    ``Rhythm.step_spikes`` and ``Rhythm.accent_ratio`` to audit a stimulus.

    Drive matching is only guaranteed when the comparison stimulus uses the
    same ``rate_on``; it is not enforced across separate stimulus objects.
    """
    if reference_onsets <= 0:
        raise ValueError("reference_onsets must be greater than zero")
    total_spikes = _reference_spikes(rate_on, step_duration, reference_onsets)
    needed = required_accent_spikes()
    if total_spikes < needed:
        raise ValueError(
            f"rate_on={rate_on:g} Hz gives {total_spikes} spikes per cycle, below the "
            f"{needed} needed to represent a 5:4:3:2:1 accent over 16 steps; use "
            f"rate_on > {minimum_accent_rate(step_duration):g} Hz or a longer "
            "step_duration, or build a Rhythm with explicit step_weights for a "
            "deliberately compressed accent"
        )
    counts = allocate_spikes(METRICAL_WEIGHTS_16, total_spikes)
    weights = tuple(
        float(count) / (rate_on * step_duration) * (1.0 - 1e-9) for count in counts
    )
    return Rhythm(
        (1,) * len(METRICAL_WEIGHTS_16),
        step_duration=step_duration,
        rate_on=rate_on,
        rate_off=0.0,
        name=name,
        step_weights=weights,
    )


def make_uniform_grid(
    step_duration=0.125,
    rate_on=100.0,
    reference_onsets=METRICAL_ACCENT_REFERENCE_ONSETS,
    name="uniform-grid",
) -> Rhythm:
    """Flat 16-onset grid: control for onset density without any accent.

    An onset on every step with equal drive, so the onset rate is 8 Hz but the
    metrical hierarchy is absent. Comparing this against
    ``make_metrical_accent`` isolates the effect of weighting; comparing it
    against a 4-onset binary pattern isolates the effect of event rate.
    """
    total_spikes = _reference_spikes(rate_on, step_duration, reference_onsets)
    n_steps = len(METRICAL_WEIGHTS_16)
    counts = allocate_spikes((1.0,) * n_steps, total_spikes)
    if int(counts.min()) < 1:
        needed_rate = n_steps / (reference_onsets * step_duration)
        raise ValueError(
            f"rate_on={rate_on:g} Hz gives {total_spikes} spikes per cycle, too few "
            f"for an onset on every step; use rate_on >= {needed_rate:.0f} Hz"
        )
    weights = tuple(
        float(count) / (rate_on * step_duration) * (1.0 - 1e-9) for count in counts
    )
    return Rhythm(
        (1,) * n_steps,
        step_duration=step_duration,
        rate_on=rate_on,
        rate_off=0.0,
        name=name,
        step_weights=weights,
    )


def make_accented_onbeat(
    step_duration=0.125,
    rate_on=100.0,
    name="accented-onbeat",
) -> Rhythm:
    """On-beat 4-onset pattern whose onsets carry metrical weights.

    The onset rate and onset positions match ``ONBEAT_BACKGROUND_16`` (2 Hz
    event rate), but the four quarter-note beats are accented 5:4:3:2:1-style
    by their metrical position (downbeat strongest, beat 3 next). Total drive
    matches the unweighted on-beat background at the same ``rate_on``.
    """
    positions = tuple(
        index for index, onset in enumerate(ONBEAT_BACKGROUND_16) if onset
    )
    position_weights = tuple(float(METRICAL_WEIGHTS_16[index]) for index in positions)
    total_spikes = _reference_spikes(rate_on, step_duration, len(positions))
    counts = allocate_spikes(position_weights, total_spikes)
    if int(counts.min()) < 1:
        needed_total = int(np.ceil(sum(position_weights) / min(position_weights)))
        needed_rate = needed_total / (len(positions) * step_duration)
        raise ValueError(
            f"rate_on={rate_on:g} Hz gives {total_spikes} spikes per cycle, too few "
            f"to accent every beat; use rate_on >= {needed_rate:.0f} Hz"
        )

    pattern = [0] * len(ONBEAT_BACKGROUND_16)
    full_weights = [0.0] * len(ONBEAT_BACKGROUND_16)
    for index, count in zip(positions, counts):
        pattern[index] = 1
        full_weights[index] = (
            float(count) / (rate_on * step_duration) * (1.0 - 1e-9)
        )
    return Rhythm(
        tuple(pattern),
        step_duration=step_duration,
        rate_on=rate_on,
        rate_off=0.0,
        name=name,
        step_weights=tuple(full_weights),
    )


def stimulus_audit(rhythm) -> dict:
    """Auditable summary of a stimulus, suitable for a results row.

    Records the realised per-step spike counts so the actual accent ratios of
    a run are visible rather than assumed from the nominal weights.
    """
    counts = rhythm.step_spikes
    return {
        "spike_counts": " ".join(str(count) for count in counts),
        "spikes_per_cycle": rhythm.spikes_per_cycle,
        "onset_rate_hz": rhythm.onset_rate,
        "pulse_frequency_hz": rhythm.pulse_frequency,
        "accent_ratio": rhythm.accent_ratio,
        "rate_on_hz": rhythm.rate_on,
        "step_duration_s": rhythm.step_duration,
    }


def minimum_accent_rate(
    step_duration: float = 0.125,
    reference_onsets: int = METRICAL_ACCENT_REFERENCE_ONSETS,
) -> float:
    """Infimum of ``rate_on`` that provides a full 31-spike accent budget.

    The weights sum to 31, and the reference emits
    ``reference_onsets * ceil(step_duration * rate_on)`` spikes, so the
    per-onset count must satisfy ``ceil(step * rate) >= 31 / 4``, i.e.
    ``ceil(step * rate) >= 8``.

    ``ceil`` makes the bound exclusive: ``rate > 7 / step_duration`` rather
    than ``rate >= 31 / (4 * step_duration)``. For the defaults that is
    ``rate > 56 Hz`` (at exactly 56 Hz there are only 28 spikes).

    This is the threshold for *representing the full hierarchy*. A compressed
    accent is still possible below it: about 40.1 Hz gives 24 spikes with every
    step audible but a 4:1 rather than 5:1 profile, and 40 Hz gives one silent
    step. ``make_metrical_accent`` enforces the 31-spike threshold; build a
    ``Rhythm`` with explicit ``step_weights`` to go below it.
    """
    needed = required_accent_spikes()
    per_onset = int(np.ceil(needed / reference_onsets))
    return float(per_onset - 1) / step_duration


def required_accent_spikes() -> int:
    """Minimum spikes per cycle to place an onset on every weighted step."""
    values = np.asarray(METRICAL_WEIGHTS_16, dtype=float)
    return int(np.ceil(values.sum() / values.min()))


def accent_is_representable(
    rate_on: float,
    step_duration: float = 0.125,
    reference_onsets: int = METRICAL_ACCENT_REFERENCE_ONSETS,
) -> bool:
    """Whether ``rate_on`` gives enough reference spikes for a full accent."""
    if rate_on <= 0 or step_duration <= 0:
        return False
    reference = _reference_spikes(rate_on, step_duration, reference_onsets)
    return reference >= required_accent_spikes()


def combined_pattern(*rhythms) -> tuple[int, ...]:
    """Onset union of several rhythms, i.e. the onset pattern presented.

    Used to record the acoustic input factually. Syncopation is scored on the
    foreground rhythm against the fixed metrical hierarchy, not on this union:
    the meter supplies the grid rather than being a measured stimulus.
    """
    patterns = [tuple(int(value) for value in rhythm.pattern) for rhythm in rhythms if rhythm is not None]
    if not patterns:
        raise ValueError("at least one rhythm is required")
    lengths = {len(pattern) for pattern in patterns}
    if len(lengths) != 1:
        raise ValueError(f"all rhythms must share a length, got {sorted(lengths)}")
    return tuple(
        1 if any(pattern[index] for pattern in patterns) else 0
        for index in range(lengths.pop())
    )


def _reference_spikes(rate_on: float, step_duration: float, n_onsets: int) -> int:
    """Spikes emitted by ``n_onsets`` unweighted onset steps."""
    per_onset = int(np.ceil(step_duration * rate_on - 1e-9))
    if per_onset <= 0:
        raise ValueError("rate_on and step_duration must produce at least one spike per onset")
    return per_onset * n_onsets


def allocate_spikes(weights, total_spikes: int) -> np.ndarray:
    """Allocate ``total_spikes`` across weights, proportional with exact total.

    Integer parts are assigned first and the remainder goes to the largest
    fractional parts, so the returned counts always sum to ``total_spikes``.
    Steps that share a weight can therefore differ by at most one spike.
    """
    values = np.asarray(weights, dtype=float)
    if np.any(values < 0):
        raise ValueError("weights must be non-negative")
    if total_spikes < 0:
        raise ValueError("total_spikes must be non-negative")
    if total_spikes == 0:
        return np.zeros(values.shape, dtype=int)

    total_weight = float(values.sum())
    if total_weight <= 0.0:
        raise ValueError("weights must have a positive sum")

    raw = values * total_spikes / total_weight
    counts = np.floor(raw).astype(int)
    remainder = int(total_spikes - counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def _validate_step_weights(step_weights, n_steps: int) -> np.ndarray:
    """Return step weights as an array, validating length and sign."""
    weights = np.asarray(step_weights, dtype=float)
    if weights.shape != (n_steps,):
        raise ValueError(f"step_weights must have {n_steps} entries")
    if np.any(weights < 0):
        raise ValueError("step_weights must be non-negative")
    return weights


def pattern_period(pattern) -> int:
    """Compute period of a pattern in steps"""
    return _sequence_period([int(x) for x in pattern])


def sequence_period(sequence) -> int:
    """Shortest repeating period of a sequence of hashable tokens."""
    return _sequence_period(list(sequence))


def _sequence_period(sequence) -> int:
    """Shortest repeating period of a sequence of hashable tokens."""
    n = len(sequence)
    for d in range(1, n + 1):
        if n % d == 0 and all(sequence[i] == sequence[i - d] for i in range(d, n)):
            return d
    return n


def pattern_rate(pattern, t, step_duration=0.25, rate_on=40.0, rate_off=0.0, step_weights=None):
    """Firing rate of an onset pattern at times "t" in seconds.

    ``step_weights`` scales ``rate_on`` for each step; rests are unaffected.
    """
    pattern = np.asarray(pattern, dtype=np.int64)
    n_steps = len(pattern)
    t = np.asarray(t, dtype=float)
    step_idx = np.floor(t / step_duration).astype(np.int64) % n_steps
    onset_rates = np.full(n_steps, float(rate_on))
    if step_weights is not None:
        onset_rates = onset_rates * _validate_step_weights(step_weights, n_steps)
    return np.where(pattern[step_idx] == 1, onset_rates[step_idx], rate_off)


def spike_trains(rhythm, duration, n_neurons=100, seed=0):
    """Generates spike trains for "n_neurons" input neurons"""
    if rhythm.jitter_ms < 0:
        raise ValueError("jitter_ms must be > 0")
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
            step_weights=rhythm.step_weights,
        )
        all_times.append(times)
        all_indices.append(np.full(times.size, neuron_idx, dtype=np.int32))
    if not all_times:
        return np.empty(0), np.empty(0, dtype=np.int32)
    return np.concatenate(all_times), np.concatenate(all_indices)


def spike_generator_group(rhythm, duration, n_neurons=100, seed=0):
    """Brian2 SpikeGeneratorGroup firing according to rhythm"""
    times, indices = spike_trains(rhythm, duration, n_neurons=n_neurons, seed=seed)
    return b2.SpikeGeneratorGroup(n_neurons, indices, times * b2.second)


def _pattern_times(pattern, step_duration, duration, rate_on, rate_off, jitter=0.0, rng=None,
                   step_weights=None):
    """Regularly spaced spike times following a step-wise constant rate.

    Per-step jitter shifts each step's spike train by an independent offset, so
    a spike can cross a step boundary. When two onsets are adjacent (for example
    at the cycle wrap, positions 15 and 0), the last spike of one step and the
    first spike of the next can land closer together than the integration step
    and crash Brian2. The returned train is therefore sorted and any pair closer
    than ``_MIN_SPIKE_INTERVAL`` is merged to the earlier spike.
    """
    events = []
    n_steps = len(pattern)
    weights = None if step_weights is None else _validate_step_weights(step_weights, n_steps)
    step_idx = 0
    t_start = 0.0
    while t_start < duration:
        idx = step_idx % n_steps
        rate = rate_on * weights[idx] if (pattern[idx] and weights is not None) else (
            rate_on if pattern[idx] else rate_off
        )
        t_end = min(t_start + step_duration, duration)
        if rate > 0.0:
            offset = rng.normal(0.0, jitter) if rng is not None else 0.0
            times = np.arange(t_start + offset, t_end + offset, 1.0 / rate)
            events.append(times[(times >= 0.0) & (times < duration)])
        t_start = t_end
        step_idx += 1
    if events:
        return _enforce_min_interval(np.concatenate(events))
    return np.empty(0)


_MIN_SPIKE_INTERVAL = 0.0002  # 0.2 ms, safely above the model's 0.1 ms dt


def _enforce_min_interval(times: np.ndarray) -> np.ndarray:
    """Sort spike times and merge any pair closer than ``_MIN_SPIKE_INTERVAL``.

    This guards against jitter pushing two spikes of the same input channel
    within one integration time step, which Brian2 rejects.
    """
    times = np.sort(times)
    if times.size < 2:
        return times
    keep = np.empty(times.size, dtype=bool)
    keep[0] = True
    last = times[0]
    for index in range(1, times.size):
        if times[index] - last >= _MIN_SPIKE_INTERVAL:
            keep[index] = True
            last = times[index]
        else:
            keep[index] = False
    return times[keep]


# Presets: no input, a rhythmic on beat pulse, and the syncopated variants
SILENCE = Rhythm((0,) * 8, rate_on=0.0, rate_off=0.0, name="silence")
NO_INPUT = SILENCE
RHYTHMIC = make_rhythm(2)
SYNCOPATED = {pid: make_rhythm(pid) for pid in (4, 5)}
