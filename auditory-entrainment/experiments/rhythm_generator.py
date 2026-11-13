"""Explore fixed-density 16-step rhythm patterns.

The generator enumerates binary rhythms, scores them with the project's
syncopation measures, and optionally filters or exports the results.

Examples
--------
List all four-onset rhythms whose shortest period is one full bar::

    python -m experiments.rhythm_generator --onsets 4 --steps 16 --period 16 \
        --bpm 120 --note-type 16

Require a downbeat and export the candidates as CSV::

    python -m experiments.rhythm_generator --onsets 4 --full-period \
        --require-downbeat --csv results/rhythm_candidates.csv

Show one representative rhythm for every available LHL level::

    python -m experiments.rhythm_generator --onsets 4 --full-period \
        --representatives
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _path in (_ROOT, _SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.stimuli import pattern_period
from experiments.measures.syncopation import lhl, offbeat_count, wnbd

DEFAULT_BPM = 120.0
DEFAULT_NOTE_TYPE = 16
FIELDNAMES = (
    "pattern",
    "onset_positions",
    "n_onsets",
    "period_steps",
    "period_seconds",
    "fundamental_hz",
    "lhl",
    "offbeat_count",
    "wnbd",
)


def step_duration_from_tempo(bpm: float, note_type: int) -> float:
    """Return the duration of a quarter, eighth, or sixteenth-note step."""
    if bpm <= 0:
        raise ValueError("bpm must be greater than zero")
    if note_type not in (4, 8, 16):
        raise ValueError("note_type must be one of 4, 8, or 16")
    return 60.0 / bpm * 4.0 / note_type


def generate_patterns(
    n_onsets: int = 4,
    *,
    n_steps: int = 16,
    step_duration: float | None = None,
    bpm: float = DEFAULT_BPM,
    note_type: int = DEFAULT_NOTE_TYPE,
    require_downbeat: bool = False,
    period: int | None = None,
) -> list[dict[str, object]]:
    """Return fixed-density rhythms and their measures.

    ``n_steps`` must be a power of two because the project's LHL measure uses
    a binary metrical hierarchy. ``period`` refers to the shortest repeating
    period in steps.
    """
    if n_steps < 1 or n_steps & (n_steps - 1):
        raise ValueError("n_steps must be a positive power of two")
    if not 0 <= n_onsets <= n_steps:
        raise ValueError(f"n_onsets must be between 0 and {n_steps}")
    if step_duration is None:
        step_duration = step_duration_from_tempo(bpm, note_type)
    elif step_duration <= 0:
        raise ValueError("step_duration must be greater than zero")
    if period is not None and (period < 1 or period > n_steps):
        raise ValueError(f"period must be between 1 and {n_steps}")

    rows: list[dict[str, object]] = []
    for positions in combinations(range(n_steps), n_onsets):
        if require_downbeat and 0 not in positions:
            continue

        pattern = [0] * n_steps
        for position in positions:
            pattern[position] = 1
        period_steps = pattern_period(pattern)
        if period is not None and period_steps != period:
            continue

        rows.append(
            {
                "pattern": "".join(str(value) for value in pattern),
                "onset_positions": " ".join(str(position) for position in positions),
                "n_onsets": n_onsets,
                "period_steps": period_steps,
                "period_seconds": period_steps * step_duration,
                "fundamental_hz": 1.0 / (period_steps * step_duration),
                "lhl": lhl(pattern),
                "offbeat_count": offbeat_count(pattern),
                "wnbd": wnbd(pattern),
            }
        )
    return rows


def representative_rows(
    rows: list[dict[str, object]], n_per_level: int = 1
) -> list[dict[str, object]]:
    """Choose up to ``n_per_level`` low-WNBD representatives per LHL level."""
    if n_per_level < 1:
        raise ValueError("n_per_level must be at least one")

    representatives: list[dict[str, object]] = []
    for level in sorted({int(row["lhl"]) for row in rows}):
        options = [row for row in rows if int(row["lhl"]) == level]
        options.sort(key=lambda row: (float(row["wnbd"]), str(row["pattern"])))
        representatives.extend(options[:n_per_level])
    return representatives


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _print_rows(rows: list[dict[str, object]]) -> None:
    print("pattern          positions       period  LHL  offbeat  WNBD")
    for row in rows:
        print(
            f"{row['pattern']:<16} "
            f"{row['onset_positions']:<15} "
            f"{row['period_steps']:>2} steps  "
            f"{row['lhl']:>3}  "
            f"{row['offbeat_count']:>7}  "
            f"{float(row['wnbd']):.3f}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate and score fixed-density 16-step rhythms."
    )
    parser.add_argument("--onsets", type=int, default=4)
    parser.add_argument(
        "--steps",
        type=int,
        default=16,
        help="Number of metrical steps in the cycle (power of two; default: 16).",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=DEFAULT_BPM,
        help="Quarter-note tempo in beats per minute (default: 120).",
    )
    parser.add_argument(
        "--note-type",
        type=int,
        choices=(4, 8, 16),
        default=DEFAULT_NOTE_TYPE,
        help="Note value represented by each step: 4, 8, or 16 (default: 16).",
    )
    parser.add_argument(
        "--step-duration",
        type=float,
        default=None,
        help="Explicit step duration in seconds; overrides --bpm and --note-type.",
    )
    parser.add_argument("--require-downbeat", action="store_true")
    parser.add_argument(
        "--period",
        type=int,
        default=None,
        help="Keep only rhythms with this shortest period, in steps.",
    )
    parser.add_argument(
        "--full-period",
        action="store_true",
        help="Compatibility alias for --period equal to --steps.",
    )
    parser.add_argument(
        "--representatives",
        action="store_true",
        help="Show one representative with the lowest WNBD per LHL level.",
    )
    parser.add_argument(
        "--n-representatives",
        type=int,
        default=None,
        metavar="N",
        help="Show up to N lowest-WNBD representatives per LHL level.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.full_period and args.period is not None:
        parser.error("--full-period and --period cannot be used together")
    if args.representatives and args.n_representatives is not None:
        parser.error("--representatives and --n-representatives cannot be used together")
    if args.n_representatives is not None and args.n_representatives < 1:
        parser.error("--n-representatives must be at least 1")
    period = args.steps if args.full_period else args.period
    rows = generate_patterns(
        args.onsets,
        n_steps=args.steps,
        step_duration=args.step_duration,
        bpm=args.bpm,
        note_type=args.note_type,
        require_downbeat=args.require_downbeat,
        period=period,
    )
    n_representatives = 1 if args.representatives else args.n_representatives
    displayed_rows = (
        representative_rows(rows, n_representatives)
        if n_representatives is not None
        else rows
    )
    print(
        f"{len(rows)} candidates; showing {len(displayed_rows)} "
        f"({'representatives' if n_representatives is not None else 'all patterns'})"
    )
    _print_rows(displayed_rows)

    if args.csv is not None:
        _write_csv(displayed_rows, args.csv)
        print(f"Saved {len(displayed_rows)} patterns to {args.csv}")


if __name__ == "__main__":
    main()
