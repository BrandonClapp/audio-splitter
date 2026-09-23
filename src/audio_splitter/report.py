"""Run metrics, including the null test that proves the split is lossless."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NULL_TOLERANCE = 1e-6


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def dbfs(amplitude: float) -> float:
    return 20.0 * math.log10(amplitude) if amplitude > 0 else -math.inf


def fmt_db(amplitude: float) -> str:
    value = dbfs(amplitude)
    return "-inf dBFS" if value == -math.inf else f"{value:+.1f} dBFS"


def fmt_time(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{sign}{int(hours)}:{int(minutes):02d}:{secs:06.3f}"
    return f"{sign}{int(minutes)}:{secs:06.3f}"


def null_test(original: np.ndarray, solo: np.ndarray, backing: np.ndarray) -> float:
    """`max|(solo + backing) - original|` — must be ~0 before MP3 encoding."""
    return float(np.max(np.abs((solo.astype(np.float64) + backing) - original)))


@dataclass
class Report:
    input_path: str
    duration: float
    samplerate: int
    model: str
    device: str
    shifts: int
    overlap: float
    window: tuple[float, float]
    sources: list[str]
    pad: float
    fade_ms: float
    center_strength: float
    center_pan: float
    melody_mask: bool
    solo_rms_inside: float
    solo_rms_outside: float
    stem_rms_inside: float
    null_error: float
    gain_db: float | None
    peaks: tuple[float, float]
    outputs: list[str] = field(default_factory=list)
    preview: float | None = None
    from_cache: bool = False
    runtime: float = 0.0

    @property
    def null_ok(self) -> bool:
        return self.null_error < NULL_TOLERANCE

    def render(self) -> str:
        start, end = self.window
        retained = (
            fmt_db(self.solo_rms_inside / self.stem_rms_inside)
            if self.stem_rms_inside > 0 else "n/a"
        )
        melody = "on" if self.melody_mask else "off"
        device = f"{self.device} (cached stems)" if self.from_cache else self.device
        gain = (
            "none needed (no clipping)" if self.gain_db is None
            else f"{self.gain_db:+.2f} dB applied to both files"
        )
        verdict = "OK" if self.null_ok else f"FAIL (>= {NULL_TOLERANCE:g})"

        lines = [
            "",
            f"  input      : {self.input_path}  ({self.duration:.2f} s @ {self.samplerate} Hz)",
            f"  model      : {self.model} on {device}  (shifts={self.shifts}, overlap={self.overlap})",
            f"  window     : {fmt_time(start)} - {fmt_time(end)}  ({end - start:.3f} s, pad {self.pad:g} s, fade {self.fade_ms:g} ms)",
            f"  sources    : solo built from {' + '.join(self.sources)}",
            f"  masks      : center strength={self.center_strength:g} pan={self.center_pan:g} | melody {melody}",
            f"  solo level : {fmt_db(self.solo_rms_inside)} inside window, {fmt_db(self.solo_rms_outside)} outside",
            f"  mask keep  : {retained} of the source stems' energy inside the window",
            f"  peaks      : solo {fmt_db(self.peaks[0])}, backing {fmt_db(self.peaks[1])} (true peak)",
            f"  level      : {gain}",
            f"  null test  : max|(solo + backing) - original| = {self.null_error:.3e}  {verdict}",
        ]
        if self.preview is not None:
            lines.append(f"  PREVIEW    : encoded window +/-{self.preview:g} s only, not the full track")
        for path in self.outputs:
            lines.append(f"  wrote      : {path}")
        lines.append(f"  runtime    : {self.runtime:.1f} s")
        lines.append("")
        return "\n".join(lines)
