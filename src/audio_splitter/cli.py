"""Command line entry point: `split-solo INPUT --from M:SS --to M:SS`."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE, __version__
from .io_audio import (
    AudioError,
    decode,
    encode_mp3,
    probe_title,
    true_peak,
    write_float_wav,
)
from .report import Report, null_test, rms
from .separate import DEFAULT_MODEL, SeparationError, separate
from .solo import complement, extract_solo

DEFAULT_SOURCES = "guitar"
PREVIEW_CONTEXT = 2.0          # seconds of context either side of the window in --preview
PEAK_CEILING_DB = -0.3         # leave a little headroom; MP3 can overshoot the PCM peak

_TIMECODE_RE = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$")


def parse_timecode(text: str) -> float:
    """Parse `M:SS`, `M:SS.mmm`, `H:MM:SS`, or bare seconds into seconds.

    Raises `ValueError` on anything else, including negative values and out-of-range
    minute/second fields such as `2:75`.
    """
    if not isinstance(text, str):
        raise ValueError(f"bad timecode: {text!r}")
    match = _TIMECODE_RE.match(text.strip())
    if match is None:
        raise ValueError(
            f"bad timecode {text!r} — use M:SS, M:SS.mmm, H:MM:SS, or seconds"
        )

    first, second, secs = match.groups()
    seconds = float(secs)
    if first is None:
        return seconds

    if seconds >= 60.0:
        raise ValueError(f"bad timecode {text!r} — seconds field must be under 60")
    if second is None:
        return int(first) * 60.0 + seconds

    if int(second) >= 60:
        raise ValueError(f"bad timecode {text!r} — minutes field must be under 60")
    return int(first) * 3600.0 + int(second) * 60.0 + seconds


def _timecode_arg(text: str) -> float:
    try:
        return parse_timecode(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="split-solo",
        description=(
            "Isolate a guitar solo over a time window into <name>.solo.mp3, and write "
            "everything else — rhythm guitar included — to <name>.backing.mp3."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="input audio (anything ffmpeg reads)")
    parser.add_argument("--from", dest="start", type=_timecode_arg, required=True,
                        metavar="TIME", help="solo start (M:SS, H:MM:SS, or seconds)")
    parser.add_argument("--to", dest="end", type=_timecode_arg, required=True,
                        metavar="TIME", help="solo end")
    parser.add_argument("-o", "--outdir", type=Path, default=None,
                        help="output directory (default: next to the input)")

    sep = parser.add_argument_group("separation")
    sep.add_argument("--model", default=DEFAULT_MODEL, help="demucs model with a guitar source")
    sep.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    sep.add_argument("--shifts", type=int, default=1,
                     help="demucs shift-trick passes; 5 is slower and cleaner")
    sep.add_argument("--overlap", type=float, default=0.25, help="demucs segment overlap")
    sep.add_argument("--source", default=DEFAULT_SOURCES, metavar="NAMES",
                     help="comma-separated stems the solo is built from; add 'other' or "
                          "'vocals' when a distorted lead has leaked into them")

    gate = parser.add_argument_group("window")
    gate.add_argument("--fade", type=float, default=30.0, metavar="MS",
                      help="raised-cosine fade at the window boundaries, in ms")
    gate.add_argument("--pad", type=float, default=0.0, metavar="SECONDS",
                      help="extra context kept either side of the window")

    mask = parser.add_argument_group("lead refinement")
    mask.add_argument("--center-strength", type=float, default=1.0,
                      help="center/coherence mask exponent; 0 disables it")
    mask.add_argument("--center-pan", type=float, default=0.0,
                      help="where the solo sits: -1 hard left, 0 center, +1 hard right")
    mask.add_argument("--melody-mask", action="store_true",
                      help="also keep only the tracked lead pitch's harmonics")
    mask.add_argument("--harmonics", type=int, default=12, help="harmonics kept per f0")
    mask.add_argument("--harm-cents", type=float, default=60.0,
                      help="Gaussian width around each harmonic, in cents")
    mask.add_argument("--harm-floor", type=float, default=0.1,
                      help="floor of the harmonic mask; keeps pick attack and distortion")

    out = parser.add_argument_group("output")
    out.add_argument("--bitrate", default="320k", help="MP3 CBR bitrate")
    out.add_argument("--vbr", action="store_true", help="encode VBR (-q:a 0) instead")
    out.add_argument("--preview", action="store_true",
                     help=f"encode only the window +/-{PREVIEW_CONTEXT:g} s, for fast auditioning")
    out.add_argument("--keep-stems", action="store_true", help="also write the raw demucs stems")
    out.add_argument("--no-cache", action="store_true", help="don't read or write the stem cache")
    out.add_argument("--no-normalize", action="store_true",
                     help="don't pull levels down when a file would clip")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _normalize(solo: np.ndarray, backing: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Scale both files by the *same* gain if either would clip, preserving their balance."""
    ceiling = 10.0 ** (PEAK_CEILING_DB / 20.0)
    peak = max(true_peak(solo), true_peak(backing))
    if peak <= 1.0 or peak == 0.0:
        return solo, backing, None
    gain = ceiling / peak
    gain_db = 20.0 * np.log10(gain)
    return (solo * gain).astype(np.float32), (backing * gain).astype(np.float32), float(gain_db)


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    src = args.input
    if args.end <= args.start:
        print(f"error: --to ({args.end:g}s) must be after --from ({args.start:g}s)", file=sys.stderr)
        return 2
    if not src.is_file():
        print(f"error: no such file: {src}", file=sys.stderr)
        return 2

    sources = [name.strip() for name in args.source.split(",") if name.strip()]
    if not sources:
        print("error: --source needs at least one stem name", file=sys.stderr)
        return 2

    print(f"decoding {src} ...")
    audio = decode(src, SAMPLE_RATE)
    n = audio.shape[-1]
    duration = n / SAMPLE_RATE

    if args.start >= duration:
        print(f"error: --from {args.start:g}s is past the end of the track ({duration:.2f}s)",
              file=sys.stderr)
        return 2
    if args.end > duration:
        print(f"  note: --to {args.end:g}s is past the end ({duration:.2f}s) — clamping")

    win_start = max(0.0, args.start - args.pad)
    win_end = min(duration, args.end + args.pad)
    start_i = int(round(win_start * SAMPLE_RATE))
    end_i = min(n, int(round(win_end * SAMPLE_RATE)))
    fade_i = int(round(args.fade / 1000.0 * SAMPLE_RATE))

    print(f"separating with {args.model} ...")
    stems, device, from_cache = separate(
        audio, SAMPLE_RATE,
        model=args.model, device=args.device, shifts=args.shifts, overlap=args.overlap,
        use_cache=not args.no_cache, verbose=args.verbose, require_sources=sources,
    )
    missing = [name for name in sources if name not in stems]
    if missing:
        print(f"error: model {args.model!r} produced no {', '.join(missing)} stem — its sources "
              f"are {', '.join(stems)}.", file=sys.stderr)
        return 2

    # The solo estimate is the sum of the chosen stems; everything else is the backing.
    stem = np.sum([stems[name] for name in sources], axis=0, dtype=np.float32)
    print("refining to lead only ...")
    solo = extract_solo(
        stem, SAMPLE_RATE, start_i, end_i,
        fade=fade_i,
        center_strength=args.center_strength, center_pan=args.center_pan,
        use_melody_mask=args.melody_mask,
        harmonics=args.harmonics, harm_cents=args.harm_cents, harm_floor=args.harm_floor,
    )
    backing = complement(audio, solo)

    error = null_test(audio, solo, backing)
    solo_rms_inside = rms(solo[:, start_i:end_i])
    solo_rms_outside = rms(np.concatenate([solo[:, :start_i], solo[:, end_i:]], axis=-1))
    stem_rms_inside = rms(stem[:, start_i:end_i])

    if not args.no_normalize:
        solo, backing, gain_db = _normalize(solo, backing)
    else:
        gain_db = None

    preview = None
    if args.preview:
        preview = PREVIEW_CONTEXT
        lo = max(0, start_i - int(PREVIEW_CONTEXT * SAMPLE_RATE))
        hi = min(n, end_i + int(PREVIEW_CONTEXT * SAMPLE_RATE))
        solo, backing = solo[:, lo:hi], backing[:, lo:hi]

    outdir = args.outdir or src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    track = probe_title(src) or src.stem
    solo_path = outdir / f"{src.stem}.solo.mp3"
    backing_path = outdir / f"{src.stem}.backing.mp3"

    print("encoding ...")
    for path, data, suffix in (
        (solo_path, solo, "guitar solo"),
        (backing_path, backing, "backing"),
    ):
        encode_mp3(path, data, SAMPLE_RATE, bitrate=args.bitrate, vbr=args.vbr,
                   title=f"{track} — {suffix}")

    outputs = [str(solo_path), str(backing_path)]
    if args.keep_stems:
        for name, data in stems.items():
            stem_path = outdir / f"{src.stem}.stem-{name}.wav"
            write_float_wav(stem_path, data, SAMPLE_RATE)
            outputs.append(str(stem_path))

    report = Report(
        input_path=str(src), duration=duration, samplerate=SAMPLE_RATE,
        model=args.model, device=device, shifts=args.shifts, overlap=args.overlap,
        window=(win_start, win_end), pad=args.pad, fade_ms=args.fade, sources=sources,
        center_strength=args.center_strength, center_pan=args.center_pan,
        melody_mask=args.melody_mask,
        solo_rms_inside=solo_rms_inside, solo_rms_outside=solo_rms_outside,
        stem_rms_inside=stem_rms_inside,
        null_error=error, gain_db=gain_db,
        peaks=(true_peak(solo), true_peak(backing)),
        outputs=outputs, preview=preview, from_cache=from_cache,
        runtime=time.monotonic() - started,
    )
    print(report.render())
    return 0 if report.null_ok else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (AudioError, SeparationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
