"""Syncopation measures for rhythmic patterns.

The Longuet-Higgins & Lee (1984) measure quantifies how strongly a
rhythm places notes on weak metrical positions while the following
stronger position is left silent.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["metrical_weights", "lhl_syncopation", "lhl", "offbeat_count", "wnbd"]


def metrical_weights(n_steps: int = 8) -> list[int]:
    """Metrical weights of a 4/4 bar sampled at ``n_steps`` equal steps.

    The downbeat has weight 0 (strongest); weaker positions get higher
    weights. For the default 8 eighth-note steps the weights are
    ``[0, 3, 2, 3, 1, 3, 2, 3]``.

    ``n_steps`` must be a power of two.
    """
    if n_steps < 1 or n_steps & (n_steps - 1):
        raise ValueError("n_steps must be a power of two")
    levels = n_steps.bit_length() - 1
    weights = [0] * n_steps
    for i in range(1, n_steps):
        # Exponent of the largest power of two dividing i.
        v2 = (i & -i).bit_length() - 1
        weights[i] = levels - v2
    return weights


def lhl_syncopation(onsets: Sequence[int], weights: Sequence[int] | None = None) -> int:
    """Longuet-Higgins & Lee (1984) syncopation of a binary onset pattern.

    A note is syncopated when it sits on a metrically weak position and
    the next metrically stronger position is a rest; it contributes the
    weight difference between the two positions.

    Parameters
    ----------
    onsets : sequence of int
        Binary pattern; ``1`` marks a note and ``0`` a rest.
    weights : sequence of int, optional
        Metrical weight per step. Defaults to the 4/4 weights returned
        by :func:`metrical_weights` for ``len(onsets)``.

    Returns
    -------
    int
        Total syncopation (0 = no syncopation; larger = more).
    """
    onsets = [int(o) for o in onsets]
    if weights is None:
        weights = metrical_weights(len(onsets))
    weights = [int(w) for w in weights]
    if len(weights) != len(onsets):
        raise ValueError("weights must have the same length as onsets")

    total = 0
    n = len(onsets)
    for i, is_note in enumerate(onsets):
        if not is_note:
            continue
        # First position after i, in cyclic order, on a stronger beat.
        for step in range(1, n):
            j = (i + step) % n
            if weights[j] < weights[i]:
                if onsets[j] == 0:
                    total += weights[i] - weights[j]
                break
    return total


lhl = lhl_syncopation


def offbeat_count(onsets, beats=None) -> int:
    """Number of onsets that do not fall on a metrical beat.

    By default the beats are the quarter-note positions of a 4/4 bar
    sampled at ``len(onsets)`` steps (positions 0, 2, 4, ...). Pass
    ``beats`` to use different beat positions.
    """
    onsets = [int(o) for o in onsets]
    n = len(onsets)
    if beats is None:
        beats = range(0, n, 2)
    beats = {int(b) % n for b in beats}
    return sum(1 for i, o in enumerate(onsets) if o and i not in beats)


def wnbd(onsets, beats=None) -> float:
    """Weighted Note-to-Beat Distance (Gomez, Melvin, Rappaport, Toussaint 2005).

    For each note, the cyclic distance in grid steps to the nearest beat
    is averaged over all notes and normalised by the largest possible
    distance, so the result is in [0, 1]. 0 means every note falls on a
    beat; 1 means every note is maximally far from one.
    """
    onsets = [int(o) for o in onsets]
    n = len(onsets)
    if beats is None:
        beats = list(range(0, n, 2))
    beats = [int(b) % n for b in beats]
    if not beats:
        return 0.0

    def dist(p):
        return min(min(abs(p - b), n - abs(p - b)) for b in beats)

    max_dist = max(dist(p) for p in range(n)) or 1
    notes = [i for i, o in enumerate(onsets) if o]
    if not notes:
        return 0.0
    return float(sum(dist(p) for p in notes) / (len(notes) * max_dist))
