"""
Three measures of syncopation implemented from literature:

Offbeat count (Huron 2006) :
- counts the number of pulses that do not fall on a metrical beat

Longuet Higgins & Lee (1984) :
- counts the number of notes that fall on weak beats and are followed by a rest on a stronger beat

WNBD (Gomez, Melvin, Rappaport, Toussaint 2005) :
- computes cyclic distance of each note to the nearest beat, averages over all notes and normalises by the largest distance to range of [0, 1]

"""

from __future__ import annotations
from collections.abc import Sequence

__all__ = ["metrical_weights", "lhl_syncopation", "lhl", "offbeat_count", "wnbd"]

def _trailing_zeros(i: int) -> int:
    """Number of times i divides evenly by 2 """
    return (i & -i).bit_length() - 1


def metrical_weights(n_steps: int = 8) -> list[int]:
    """
    Binary metrical hierarchy (Lerdahl & Jackendoff 1983; Longuet-Higgins & Lee 1984).

    Weights are assigned to each position in a pattern of n_steps, where the first step is the strongest:
    - The trailing zeros of the position index are used to determine its level in the metrical hierarchy [n_steps = 2^levels]
    - Weaker positions get higher weights, stronger positions get lower weights

    For eighth notes (8 steps) and sixteenth notes (16 steps) in a 4/4 bar, the weights are:
    n_steps=8:  [0, 3, 2, 3, 1, 3, 2, 3]
    n_steps=16: [0, 4, 3, 4, 2, 4, 3, 4, 1, 4, 3, 4, 2, 4, 3, 4]
    """
    if n_steps < 1 or n_steps & (n_steps - 1):
        raise ValueError("n_steps must be a power of two.")
    levels = n_steps.bit_length() - 1
    return [0 if i == 0 else levels - _trailing_zeros(i) for i in range(n_steps)]


def _quarter_note_beats(n_steps: int) -> range:
    """Positions of the four quarter-note beats in a 4/4 bar sampled at "n_steps"""
    step = n_steps // 4
    if step < 1:
        raise ValueError("n_steps must be 4")
    return range(0, n_steps, step)


def _eighth_note_beats(n_steps: int) -> range:
    """
    Positions of the eight eighth note beats in a 4/4 bar sampled at "n_steps"
    """
    step = n_steps // 8
    if step < 1:
        raise ValueError("n_steps must be 8")
    return range(0, n_steps, step)


def offbeat_count(onsets, beats=None) -> int:
    """
    Number of pulses that do not fall on a metrical beat.

    By default the beats are the four quarter note positions of a 4/4 bar
    sampled at "len(onsets)" steps (positions 0, n//4, 2*(n//4), 3*(n//4)):

    - 8 step pattern:  quarter note beats at (0, 2, 4, 6)
    - 16 step pattern: quarter note beats at (0, 4, 8, 12)

    The offbeats are the remaining positions. Pass "beats" to use different
    beat positions.
    """
    onsets = [int(o) for o in onsets]
    n = len(onsets)
    if beats is None:
        beats = _quarter_note_beats(n)
    beats = {int(b) % n for b in beats}
    return sum(1 for i, o in enumerate(onsets) if o and i not in beats)

def lhl_syncopation(onsets: Sequence[int], weights: Sequence[int] | None = None) -> int:
    """
    Longuet Higgins & Lee (1984) syncopation of a binary onset pattern implemented from:
    The perception of musical rhythm (Longuet-Higgins, M. C., & Lee, D. M. (1984).
    In Music and the Mind (pp. 157-176). Oxford University Press.)

    A note is syncopated when:
    - It happens at a metrically weak position and the next metrically stronger position is a rest
    - We assign weights to each position in the pattern, with lower weights for stronger positions


    Parameters
    ----------
    onsets : sequence of ints
        binary pattern: "1" marks a note and "0" a musical rest
    weights : sequence of int, optional
        Metrical weight per step.
        If None, the default weights are used (metrical_weights(len(onsets))).

    Returns
    -------
    int
        Total syncopation (0 = no syncopation; larger = more)
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


def wnbd(onsets, beats=None) -> float:
    """
    Weighted Note to Beat Distance (Gomez, Melvin, Rappaport, Toussaint 2005)

    WNBD is computed as follows:
    - For each note, the cyclic distance to the nearest beat
    is averaged over all positions and 
    - normalised by the largest distance to range of[0, 1]
    - "0" : every note falls on a beat
    - "1" : means every note is maximally distant from a beat

    By default the beats are the four quarter note positions of a 4/4 bar
    sampled at "len(onsets)" steps (eighth-note patterns use 0, 2, 4, 6 and
    sixteenth-note patterns use 0, 4, 8, 12). Pass "beats" to use different
    beat positions.
    """
    onsets = [int(o) for o in onsets]
    n = len(onsets)
    if beats is None:
        beats = list(_quarter_note_beats(n))
    beats = [int(b) % n for b in beats]
    if not beats:
        return 0.0

    def dist(p):
        return min(min(abs(p-b), n-abs(p-b)) for b in beats)

    max_dist = max(dist(p) for p in range(n)) or 1
    notes = [i for i, o in enumerate(onsets) if o]
    if not notes:
        return 0.0
    return float(sum(dist(p) for p in notes) / (len(notes) * max_dist))
