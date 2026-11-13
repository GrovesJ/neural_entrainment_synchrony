"""Visualise and listen to every rhythm signal defined in ``src.stimuli``.

Each signal is written out as

* a figure with the onset pattern, the tonic input rate, the input spike
  raster and the synthesised waveform (written to ``<out>/plots``), and
* a WAV file that can be played back (written to ``<out>/sounds``), with one
  tone per onset step, or one click per input spike so what you hear matches
  what the network receives.

Examples
--------
    python experiments/stimulus_inspector.py --list
    python experiments/stimulus_inspector.py
    python experiments/stimulus_inspector.py --signals preset --play
    python experiments/stimulus_inspector.py --signals eighth/5 --mode spike --play
    python experiments/stimulus_inspector.py --signals sixteenth/3 --with-background
    python experiments/stimulus_inspector.py --pitch 440 --signals eighth

Pitch follows the pattern period, not the firing rate. The 40 Hz thalamic drive
is a spike rate rather than a frequency, so it carries no pitch; the only
repeating periodicity in a ``Rhythm`` is how often the onset pattern repeats.
That period (0.5-2 Hz here, far below hearing) is transposed up eight octaves
(x256), so signals whose periods differ by an octave really do sound an octave
apart (128 / 256 / 512 Hz). Because pitch tracks the period rather than the
pattern itself, signals sharing a period share a pitch and differ only in
rhythm. ``--pitch`` takes an explicit Hz value for a single fixed tone.

Signals are named ``group/identifier``:

    eighth/1 ... eighth/5        PATTERNS
    sixteenth/0 ... sixteenth/4  PATTERNS_16
    background/eighth            ONBEAT_BACKGROUND
    background/sixteenth         ONBEAT_BACKGROUND_16
    preset/rhythmic              RHYTHMIC
    preset/syncopated-tresillo   SYNCOPATED[4]
    preset/syncopated-offbeat    SYNCOPATED[5]
    preset/silence               SILENCE

Select a whole group (``eighth``), a single key (``eighth/5``) or any
substring of a key or name (``tresillo``).

In a notebook the generated WAV files can be embedded with
``IPython.display.Audio(path)``.
"""

from __future__ import annotations

# Standard library imports
import argparse
import os
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (_ROOT, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Internal imports
from src import stimuli

try:  # metrics are informational only, never required
    from experiments.measures import syncopation
except Exception:  # pragma: no cover - optional dependency chain
    syncopation = None


FIGURES_DIR = _ROOT / "figures" / "stimuli"
PLOTS_SUBDIR = "plots"
SOUNDS_SUBDIR = "sounds"
DEFAULT_PITCH = 523.25  # C5, fallback for patterns with no repeat rate
DEFAULT_SAMPLE_RATE = 44_100

# Pattern fundamentals are sub-audio (0.5-2 Hz), so the tone is the fundamental
# transposed up by eight octaves (x256) to land in the audible range. A single
# constant factor keeps every interval intact: 0.5 Hz -> 128 Hz, 1 Hz -> 256 Hz,
# 2 Hz -> 512 Hz. Pitch therefore tracks the pattern period (its repeat rate),
# not the firing rate, so patterns sharing a period share a pitch.
PITCH_TRANSPOSE = 256.0


@dataclass(frozen=True)
class SignalSpec:
    """A named onset pattern plus the parameters that turn it into a rhythm."""

    key: str
    name: str
    pattern: tuple[int, ...]
    step_duration: float
    rate_on: float = 40.0
    rate_off: float = 0.0
    jitter_ms: float = 0.0

    @property
    def group(self) -> str:
        return self.key.split("/", 1)[0]

    @property
    def n_steps(self) -> int:
        return len(self.pattern)

    @property
    def cycle_duration(self) -> float:
        return self.n_steps * self.step_duration

    @property
    def pulse_frequency(self) -> float:
        """Fundamental frequency implied by the pattern period.

        This is the rate at which the pattern repeats, and the source of the
        audible pitch. Silent patterns have no repetition rate, so it is ``nan``.
        """
        return pattern_fundamental(self.pattern, self.step_duration)

    @property
    def onset_steps(self) -> list[int]:
        return [i for i, value in enumerate(self.pattern) if value]

    def rhythm(
        self,
        *,
        step_duration: float | None = None,
        rate_on: float | None = None,
        rate_off: float | None = None,
        jitter_ms: float | None = None,
    ) -> stimuli.Rhythm:
        """Build the ``stimuli.Rhythm`` for this signal, optionally overriding parameters."""
        return stimuli.Rhythm(
            self.pattern,
            step_duration=self.step_duration if step_duration is None else step_duration,
            rate_on=self.rate_on if rate_on is None else rate_on,
            rate_off=self.rate_off if rate_off is None else rate_off,
            jitter_ms=self.jitter_ms if jitter_ms is None else jitter_ms,
            name=self.name,
        )


def signal_specs() -> dict[str, SignalSpec]:
    """Every rhythm signal exposed by ``src.stimuli``, in a stable order."""
    specs: dict[str, SignalSpec] = {}
    for pid, pattern in stimuli.PATTERNS.items():
        key = f"eighth/{pid}"
        specs[key] = SignalSpec(
            key=key,
            name=stimuli.PATTERN_NAMES[pid],
            pattern=tuple(pattern),
            step_duration=0.25,
        )
    for pid, pattern in stimuli.PATTERNS_16.items():
        key = f"sixteenth/{pid}"
        specs[key] = SignalSpec(
            key=key,
            name=stimuli.PATTERN_NAMES_16[pid],
            pattern=tuple(pattern),
            step_duration=0.125,
        )
    specs["background/eighth"] = SignalSpec(
        key="background/eighth",
        name="onbeat background (eighth)",
        pattern=tuple(stimuli.ONBEAT_BACKGROUND),
        step_duration=0.25,
    )
    specs["background/sixteenth"] = SignalSpec(
        key="background/sixteenth",
        name="onbeat background (sixteenth)",
        pattern=tuple(stimuli.ONBEAT_BACKGROUND_16),
        step_duration=0.125,
    )
    specs["preset/rhythmic"] = SignalSpec(
        key="preset/rhythmic",
        name=stimuli.RHYTHMIC.name,
        pattern=tuple(stimuli.RHYTHMIC.pattern),
        step_duration=stimuli.RHYTHMIC.step_duration,
        rate_on=stimuli.RHYTHMIC.rate_on,
    )
    for pid, rhythm in stimuli.SYNCOPATED.items():
        key = f"preset/syncopated-{stimuli.PATTERN_NAMES[pid]}"
        specs[key] = SignalSpec(
            key=key,
            name=rhythm.name,
            pattern=tuple(rhythm.pattern),
            step_duration=rhythm.step_duration,
            rate_on=rhythm.rate_on,
        )
    specs["preset/silence"] = SignalSpec(
        key="preset/silence",
        name=stimuli.SILENCE.name,
        pattern=tuple(stimuli.SILENCE.pattern),
        step_duration=stimuli.SILENCE.step_duration,
        rate_on=stimuli.SILENCE.rate_on,
        rate_off=stimuli.SILENCE.rate_off,
    )
    return specs


def resolve_signals(tokens: list[str] | None) -> list[SignalSpec]:
    """Expand user tokens (groups, keys, or name substrings) into signal specs."""
    specs = signal_specs()
    if not tokens:
        return list(specs.values())

    selected: list[SignalSpec] = []
    for raw in tokens:
        token = raw.strip().lower()
        if not token:
            continue
        if token in {"all", "*"}:
            matches = list(specs.values())
        elif token in specs:
            matches = [specs[token]]
        elif any(key.startswith(f"{token}/") for key in specs):
            matches = [spec for spec in specs.values() if spec.group == token]
        else:
            matches = [
                spec
                for key, spec in specs.items()
                if token in key or token in spec.name.lower()
            ]
        if not matches:
            raise SystemExit(
                f"Unknown signal {raw!r}.\nKnown signals:\n  "
                + "\n  ".join(specs)
            )
        selected.extend(matches)

    unique: list[SignalSpec] = []
    seen: set[str] = set()
    for spec in selected:
        if spec.key not in seen:
            seen.add(spec.key)
            unique.append(spec)
    return unique


def background_for(spec: SignalSpec, *, rate_on: float | None = None) -> stimuli.Rhythm:
    """Isochronous background pulse matching this signal's resolution."""
    if spec.n_steps == 16:
        pattern, step = stimuli.ONBEAT_BACKGROUND_16, 0.125
    else:
        pattern, step = stimuli.ONBEAT_BACKGROUND, 0.25
    return stimuli.Rhythm(
        tuple(pattern),
        step_duration=spec.step_duration if spec.n_steps in (8, 16) else step,
        rate_on=spec.rate_on if rate_on is None else rate_on,
        name="beat",
    )


# Audio synthesis
def pattern_fundamental(pattern, step_duration: float) -> float:
    """Repeat rate of a pattern in Hz, or ``nan`` for an all-silent pattern."""
    values = tuple(int(v) for v in pattern)
    if not any(values):
        return float("nan")
    period = stimuli.pattern_period(values) or len(values)
    return 1.0 / (period * step_duration)


def effective_pitch(
    fundamental: float,
    pitch: float | str = "auto",
    transpose: float = PITCH_TRANSPOSE,
) -> float:
    """Audible tone frequency for a sub-audio stimulus fundamental.

    ``pitch="auto"`` (default) returns ``fundamental * transpose``, so the tone
    follows the pattern's repeat rate and patterns whose fundamentals differ by
    an octave really do sound an octave apart. Signals without a repeat rate
    (silence) fall back to ``DEFAULT_PITCH``. Pass a number for a fixed tone.
    """
    if isinstance(pitch, str):
        token = pitch.strip().lower()
        if token != "auto":
            return effective_pitch(fundamental, float(token), transpose)
        if not np.isfinite(fundamental) or fundamental <= 0.0:
            return DEFAULT_PITCH
        return float(fundamental * transpose)
    return float(pitch) if pitch > 0.0 else DEFAULT_PITCH


def _add_note(
    buffer: np.ndarray,
    sample_rate: int,
    start: float,
    duration: float,
    frequency: float,
    amplitude: float,
) -> None:
    """Add one plucked tone spanning ``duration`` seconds."""
    i0 = max(int(round(start * sample_rate)), 0)
    i1 = min(buffer.size, int(round((start + duration) * sample_rate)))
    if i1 <= i0 or amplitude <= 0.0:
        return
    t = np.arange(i1 - i0) / sample_rate
    attack = 0.006
    envelope = np.minimum(t / attack, 1.0) * np.exp(-np.maximum(t - attack, 0.0) / max(duration * 0.5, 1e-3))
    tone = (
        np.sin(2.0 * np.pi * frequency * t)
        + 0.30 * np.sin(2.0 * np.pi * 2.0 * frequency * t)
        + 0.12 * np.sin(2.0 * np.pi * 3.0 * frequency * t)
    )
    buffer[i0:i1] += amplitude * envelope * tone / 1.42


def _add_click(buffer: np.ndarray, sample_rate: int, start: float, amplitude: float = 0.35) -> None:
    """Add a short click, mirroring one input spike."""
    i0 = max(int(round(start * sample_rate)), 0)
    i1 = min(buffer.size, i0 + int(round(0.006 * sample_rate)))
    if i1 <= i0:
        return
    t = np.arange(i1 - i0) / sample_rate
    envelope = np.exp(-t / 0.0018) * np.minimum(t / 0.0003, 1.0)
    buffer[i0:i1] += amplitude * envelope * np.sin(2.0 * np.pi * 1500.0 * t)


def synthesize(
    rhythm: stimuli.Rhythm,
    duration: float,
    *,
    mode: str = "pulse",
    pitch: float | str = "auto",
    transpose: float = PITCH_TRANSPOSE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seed: int = 0,
) -> np.ndarray:
    """Render a rhythm to a mono float signal in [-1, 1].

    ``mode="pulse"`` gives one tone per stimulus step (scaled by firing rate),
    ``mode="spike"`` gives one click per spike the network actually receives.
    ``pitch`` is either an explicit frequency in Hz or ``"auto"`` to derive it
    from the pattern period via ``effective_pitch``.
    """
    if duration <= 0.0:
        return np.zeros(1)
    pitch = effective_pitch(pattern_fundamental(rhythm.pattern, rhythm.step_duration), pitch, transpose)
    n_samples = max(1, int(round(duration * sample_rate)))
    audio = np.zeros(n_samples)
    rate_on = max(rhythm.rate_on, 1e-9)

    if mode == "spike":
        times, _ = rhythm.spike_trains(duration, n_neurons=1, seed=seed)
        for spike_time in times:
            _add_click(audio, sample_rate, float(spike_time))
    elif mode == "pulse":
        n_steps = rhythm.n_steps
        n_plays = int(np.ceil(duration / rhythm.step_duration))
        for step_idx in range(n_plays):
            level = rhythm.pattern[step_idx % n_steps]
            rate = rhythm.rate_on if level else rhythm.rate_off
            if rate <= 0.0:
                continue
            amplitude = 0.6 * min(1.0, rate / rate_on)
            _add_note(
                audio,
                sample_rate,
                step_idx * rhythm.step_duration,
                rhythm.step_duration,
                pitch,
                amplitude,
            )
    else:
        raise ValueError("mode must be 'pulse' or 'spike'")

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        audio *= 0.9 / peak
    fade = min(int(0.01 * sample_rate), audio.size // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]
    return audio


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Path:
    """Write a mono 16-bit PCM WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(audio, -1.0, 1.0)
    pcm = np.round(samples * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return path


def play_wav(path: Path) -> None:
    """Play a WAV file with whatever the platform provides."""
    try:
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME)
            return
        if sys.platform == "darwin":
            subprocess.run(["afplay", str(path)], check=True)
            return
        for player in ("paplay", "aplay", "ffplay"):
            if _which(player):
                args = [player, str(path)]
                if player == "ffplay":
                    args = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
                subprocess.run(args, check=True)
                return
    except Exception as exc:  # pragma: no cover - platform dependent
        print(f"  could not play audio ({exc}); file is at {path}")
        return
    print(f"  no audio player found; file is at {path}")


def _which(command: str) -> bool:
    from shutil import which

    return which(command) is not None


# Figures
def _metrics_line(spec: SignalSpec) -> str:
    if syncopation is None:
        return ""
    return (
        f"LHL={syncopation.lhl(spec.pattern)}  "
        f"offbeats={syncopation.offbeat_count(spec.pattern)}  "
        f"WNBD={syncopation.wnbd(spec.pattern):.2f}"
    )


def plot_signal(
    spec: SignalSpec,
    path: Path,
    *,
    cycles: int = 4,
    mode: str = "pulse",
    pitch: float | str = "auto",
    transpose: float = PITCH_TRANSPOSE,
    sample_rate: int = 22_050,
    n_neurons: int = 40,
    seed: int = 0,
    rhythm: stimuli.Rhythm | None = None,
    show: bool = False,
) -> Path:
    """Draw pattern, input rate, spike raster and waveform for one signal."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rhythm = rhythm or spec.rhythm()
    fundamental = pattern_fundamental(rhythm.pattern, rhythm.step_duration)
    tone = effective_pitch(fundamental, pitch, transpose)
    duration = max(cycles, 1) * rhythm.cycle_duration
    audio = synthesize(
        rhythm,
        duration,
        mode=mode,
        pitch=tone,
        sample_rate=sample_rate,
        seed=seed,
    )

    fig, (ax_pattern, ax_rate, ax_raster, ax_audio) = plt.subplots(
        4,
        1,
        figsize=(11.0, 9.0),
        gridspec_kw={"height_ratios": [1.0, 0.8, 1.2, 0.9]},
    )

    # 1. One cycle of the binary pattern, with quarter-note beats highlighted.
    steps = np.arange(spec.n_steps)
    beats = set(stimuli_beats(spec.n_steps))
    colors = ["tab:blue" if i in beats else "tab:orange" for i in steps]
    ax_pattern.bar(steps, spec.pattern, color=colors, width=0.75)
    ax_pattern.set_xticks(steps)
    ax_pattern.set_xlim(-0.5, spec.n_steps - 0.5)
    ax_pattern.set_ylim(0.0, 1.35)
    ax_pattern.set_xlabel("step (quarter-note beats in blue)")
    ax_pattern.set_ylabel("onset")
    ax_pattern.grid(axis="y", alpha=0.3)
    ax_pattern.set_title(
        f"{spec.key} \u2014 {spec.name}   "
        f"({spec.n_steps} steps \u00d7 {spec.step_duration:g} s, "
        f"cycle {rhythm.cycle_duration:g} s)\n"
        f"pitch from pattern period: "
        + (
            f"{fundamental:.2f} Hz \u00d7 {transpose:g} \u2192 {tone:.0f} Hz tone"
            if np.isfinite(fundamental)
            else f"silent pattern, fixed {tone:.0f} Hz tone"
        )
        + (f"   |   {_metrics_line(spec)}" if _metrics_line(spec) else ""),
        fontsize=10,
        loc="left",
    )
    for onset in spec.onset_steps:
        ax_pattern.annotate(
            str(onset),
            (onset, 1.05),
            ha="center",
            fontsize=8,
            color="dimgray",
        )

    # 2. Tonic input rate over time.
    t = np.arange(0.0, duration, 1.0 / 2000.0)
    ax_rate.plot(t, rhythm.rate(t), color="tab:green", drawstyle="steps-post")
    ax_rate.set_ylabel("rate (Hz)")
    ax_rate.set_title("Input firing rate", fontsize=9, loc="left")
    ax_rate.grid(alpha=0.3)

    # 3. Spike raster for a sample of input neurons.
    spike_times, spike_indices = rhythm.spike_trains(duration, n_neurons=n_neurons, seed=seed)
    if spike_times.size:
        ax_raster.scatter(spike_times, spike_indices, s=1.0, marker="|", color="black")
    ax_raster.set_ylabel("input neuron")
    ax_raster.set_ylim(-0.5, n_neurons - 0.5)
    ax_raster.set_title(f"Spike raster ({n_neurons} neurons)", fontsize=9, loc="left")

    # 4. Waveform envelope (min/max pooled so long files stay readable).
    if audio.size:
        bins = min(2000, audio.size)
        edges = np.linspace(0, audio.size, bins + 1).astype(int)
        lo = np.array([audio[a:b].min() for a, b in zip(edges[:-1], edges[1:])])
        hi = np.array([audio[a:b].max() for a, b in zip(edges[:-1], edges[1:])])
        centres = np.linspace(0.0, duration, bins)
        ax_audio.fill_between(centres, lo, hi, color="tab:purple", linewidth=0.0)
        ax_audio.set_ylim(-1.05, 1.05)
    ax_audio.set_ylabel("audio")
    ax_audio.set_xlabel("time (s)")
    ax_audio.set_title(f"Waveform ({mode} rendering)", fontsize=9, loc="left")

    for ax in (ax_rate, ax_raster, ax_audio):
        ax.set_xlim(0.0, duration)
        step = rhythm.step_duration
        for cycle_idx in range(int(np.ceil(duration / rhythm.cycle_duration)) + 1):
            ax.axvline(cycle_idx * rhythm.cycle_duration, color="tab:red", lw=0.8, alpha=0.35)
        for step_idx in range(int(np.ceil(duration / step)) + 1):
            ax.axvline(step_idx * step, color="grey", lw=0.4, alpha=0.20)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return path


def stimuli_beats(n_steps: int) -> list[int]:
    """Quarter-note beat positions for a 4/4 bar sampled at ``n_steps``."""
    stride = max(n_steps // 4, 1)
    return list(range(0, n_steps, stride))


def plot_overview(specs: list[SignalSpec], path: Path, *, cycles: int = 2) -> Path:
    """One compact strip per signal, aligned on step time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(specs),
        1,
        figsize=(11.0, max(2.4, 0.62 * len(specs))),
        squeeze=False,
        sharex=False,
    )
    for ax, spec in zip(axes[:, 0], specs):
        strip = np.tile(np.asarray(spec.pattern, dtype=float), max(cycles, 1))
        duration = strip.size * spec.step_duration
        ax.imshow(
            strip[None, :],
            aspect="auto",
            cmap="Greys",
            interpolation="nearest",
            extent=(0.0, duration, 0.0, 1.0),
            vmin=0.0,
            vmax=1.0,
        )
        for step_idx in range(strip.size + 1):
            colour = "tab:red" if step_idx % spec.n_steps == 0 else "grey"
            ax.axvline(step_idx * spec.step_duration, color=colour, lw=0.6, alpha=0.45)
        ax.set_yticks([])
        ax.set_ylabel(spec.key, rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle(
        f"All stimuli signals ({cycles} cycles each; red lines mark cycle starts)",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# CLI
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot and/or listen to every rhythm signal in src.stimuli",
    )
    parser.add_argument(
        "--signals",
        default=None,
        help="Comma separated keys, groups or name fragments "
        "(e.g. 'eighth', 'preset', 'eighth/5,tresillo'). Default: every signal.",
    )
    parser.add_argument("--list", action="store_true", help="List signals and exit.")
    parser.add_argument("--out", default=str(FIGURES_DIR),
                        help="Base output directory; figures go to <out>/plots and "
                             f"audio to <out>/{SOUNDS_SUBDIR}.")
    parser.add_argument("--cycles", type=int, default=4, help="Cycles to render/audition.")
    parser.add_argument("--mode", choices=("pulse", "spike"), default="pulse",
                        help="pulse = one tone per step, spike = one click per input spike.")
    parser.add_argument("--pitch", default="auto",
                        help="Tone frequency in Hz, or 'auto' (default) to derive it "
                             "from the pattern period.")
    parser.add_argument("--transpose", type=float, default=PITCH_TRANSPOSE,
                        help="Octave factor applied to the pattern period when "
                             f"--pitch auto (default: {PITCH_TRANSPOSE:g} = eight octaves).")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--n-neurons", type=int, default=40, help="Neurons in the raster.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for jitter and the raster.")
    parser.add_argument("--step", type=float, default=None, dest="step_duration",
                        help="Override step duration (s) for every signal.")
    parser.add_argument("--rate-on", type=float, default=None, help="Override onset rate (Hz).")
    parser.add_argument("--rate-off", type=float, default=None, help="Override off-step rate (Hz).")
    parser.add_argument("--jitter-ms", type=float, default=None, help="Override spike jitter (ms).")
    parser.add_argument("--with-background", action="store_true",
                        help="Mix in the isochronous background pulse.")
    parser.add_argument("--no-audio", action="store_true", help="Skip WAV rendering.")
    parser.add_argument("--no-plot", action="store_true", help="Skip figures.")
    parser.add_argument("--overview", action="store_true",
                        help="Also write a single overview figure of all selected signals.")
    parser.add_argument("--play", action="store_true", help="Play each rendered WAV.")
    parser.add_argument("--play-delay", type=float, default=0.25,
                        help="Seconds to wait between played signals.")
    parser.add_argument("--open", action="store_true",
                        help="Open generated figures in the default image viewer.")
    return parser.parse_args(argv)


def _print_signals(specs: list[SignalSpec], transpose: float = PITCH_TRANSPOSE) -> None:
    header = (
        f"{'key':<26} {'steps':>5} {'step(s)':>8} {'cycle(s)':>9} "
        f"{'period(s)':>9} {'f0(Hz)':>7} {'tone(Hz)':>8}  metrics"
    )
    print(header)
    print("-" * len(header))
    for spec in specs:
        fundamental = spec.pulse_frequency
        tone = effective_pitch(fundamental, "auto", transpose)
        period = (
            f"{1.0 / fundamental:.2f}" if np.isfinite(fundamental) else "-"
        )
        f0 = f"{fundamental:.2f}" if np.isfinite(fundamental) else "-"
        print(
            f"{spec.key:<26} {spec.n_steps:>5} {spec.step_duration:>8g} "
            f"{spec.cycle_duration:>9g} {period:>9} {f0:>7} {tone:>8.0f}  "
            f"{_metrics_line(spec)}"
        )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    specs = signal_specs()

    if args.list:
        _print_signals(list(specs.values()), args.transpose)
        return

    tokens = args.signals.split(",") if args.signals else None
    selected = resolve_signals(tokens)
    outdir = Path(args.out)
    plots_dir = outdir / PLOTS_SUBDIR
    sounds_dir = outdir / SOUNDS_SUBDIR

    print(f"Inspecting {len(selected)} signal(s) -> {outdir}")
    for spec in selected:
        rhythm = spec.rhythm(
            step_duration=args.step_duration,
            rate_on=args.rate_on,
            rate_off=args.rate_off,
            jitter_ms=args.jitter_ms,
        )
        duration = max(args.cycles, 1) * rhythm.cycle_duration
        stem = spec.key.replace("/", "_")
        tone = effective_pitch(
            pattern_fundamental(rhythm.pattern, rhythm.step_duration),
            args.pitch,
            args.transpose,
        )
        print(
            f"  {spec.key:<26} {spec.name:<18} "
            f"{spec.n_steps:>2} steps  cycle={rhythm.cycle_duration:g}s  "
            f"duration={duration:g}s  tone={tone:.0f} Hz"
        )

        if not args.no_plot:
            figure_path = plot_signal(
                spec,
                outdir / PLOTS_SUBDIR / f"{stem}.png",
                cycles=args.cycles,
                mode=args.mode,
                pitch=args.pitch,
                transpose=args.transpose,
                sample_rate=min(args.sample_rate, 22_050),
                n_neurons=args.n_neurons,
                seed=args.seed,
                rhythm=rhythm,
            )
            print(f"    figure: {figure_path}")
            if args.open and sys.platform == "win32":
                os.startfile(str(figure_path))  # noqa: S606 - user requested

        if not args.no_audio:
            audio = synthesize(
                rhythm,
                duration,
                mode=args.mode,
                pitch=args.pitch,
                transpose=args.transpose,
                sample_rate=args.sample_rate,
                seed=args.seed,
            )
            if args.with_background:
                background = background_for(spec, rate_on=rhythm.rate_on)
                background_audio = synthesize(
                    background,
                    duration,
                    mode=args.mode,
                    pitch=(
                        args.pitch
                        if isinstance(args.pitch, str)
                        else args.pitch * 0.5
                    ),
                    transpose=args.transpose,
                    sample_rate=args.sample_rate,
                    seed=args.seed,
                )
                audio = audio + 0.35 * background_audio
                peak = float(np.max(np.abs(audio))) if audio.size else 0.0
                if peak > 0.0:
                    audio *= 0.9 / peak
            wav_path = write_wav(outdir / SOUNDS_SUBDIR / f"{stem}.wav", audio, args.sample_rate)
            print(f"    audio:  {wav_path}")
            if args.play:
                print("    playing...")
                play_wav(wav_path)
                time.sleep(max(args.play_delay, 0.0))

    if args.overview:
        overview_path = plot_overview(
            selected,
            outdir / PLOTS_SUBDIR / "overview.png",
            cycles=max(args.cycles, 1),
        )
        print(f"Overview: {overview_path}")


if __name__ == "__main__":
    main()
