"""Audio I/O: ffmpeg decode/encode, float32 WAV cache, level measurement.

Audio is `(2, n)` float32 everywhere — channels-first, matching demucs, so arrays and
tensors pass back and forth without transposing.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import SAMPLE_RATE


class AudioError(RuntimeError):
    """ffmpeg failed, or produced something we can't use."""


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise AudioError(f"{tool} not found on PATH — install it (brew install ffmpeg)")


def decode(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any ffmpeg-readable file to `(2, n)` float32 at `sr`."""
    require_ffmpeg()
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "2", "-ar", str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg could not decode {path}:\n{proc.stderr.decode(errors='replace')}")
    flat = np.frombuffer(proc.stdout, dtype="<f4")
    if flat.size == 0:
        raise AudioError(f"{path} decoded to zero samples")
    return np.ascontiguousarray(flat.reshape(-1, 2).T.astype(np.float32))


def probe_title(path: Path) -> str | None:
    """The file's ID3/container title tag, if it has one."""
    require_ffmpeg()
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format_tags=title",
        "-of", "json", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        return None
    try:
        tags = json.loads(proc.stdout or b"{}").get("format", {}).get("tags", {})
    except json.JSONDecodeError:
        return None
    title = tags.get("title") or tags.get("TITLE")
    return title.strip() or None if isinstance(title, str) else None


def encode_mp3(
    path: Path,
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    *,
    bitrate: str | None = "320k",
    vbr: bool = False,
    title: str | None = None,
    artist: str | None = None,
) -> None:
    """Encode `(2, n)` float32 to MP3 with libmp3lame."""
    require_ffmpeg()
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-f", "f32le", "-ar", str(sr), "-ac", str(audio.shape[0]),
        "-i", "-",
        "-codec:a", "libmp3lame",
    ]
    cmd += ["-q:a", "0"] if vbr else ["-b:a", bitrate or "320k"]
    if title:
        cmd += ["-metadata", f"title={title}"]
    if artist:
        cmd += ["-metadata", f"artist={artist}"]
    cmd.append(str(path))

    raw = np.ascontiguousarray(audio.T.astype(np.float32)).tobytes()
    proc = subprocess.run(cmd, input=raw, capture_output=True)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg could not encode {path}:\n{proc.stderr.decode(errors='replace')}")


def duration_seconds(path: Path) -> float:
    """Container duration in seconds, via ffprobe."""
    require_ffmpeg()
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nk=1:nw=1", str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise AudioError(f"ffprobe failed on {path}:\n{proc.stderr.decode(errors='replace')}")
    return float(proc.stdout.strip())


def write_float_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Write `(2, n)` float32 losslessly — used for the stem cache and --keep-stems."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.T.astype(np.float32), sr, subtype="FLOAT")


def read_float_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.ascontiguousarray(data.T), sr


def hash_audio(audio: np.ndarray, sr: int, *extra: object) -> str:
    """Stable key over decoded samples plus any extra parameters."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(audio, dtype=np.float32).tobytes())
    h.update(repr((sr, *extra)).encode())
    return h.hexdigest()[:32]


def sample_peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def true_peak(audio: np.ndarray, oversample: int = 4, block: int = 1 << 19) -> float:
    """Inter-sample peak, via `oversample`x polyphase upsampling.

    Falls back to the sample peak if scipy isn't importable. Blocked so a full-length
    track doesn't allocate an oversampled copy of itself.
    """
    if audio.size == 0:
        return 0.0
    try:
        from scipy.signal import resample_poly
    except ImportError:
        return sample_peak(audio)

    overlap = 64
    peak = 0.0
    n = audio.shape[-1]
    for start in range(0, n, block):
        lo = max(0, start - overlap)
        hi = min(n, start + block + overlap)
        chunk = resample_poly(audio[:, lo:hi], oversample, 1, axis=-1)
        peak = max(peak, float(np.max(np.abs(chunk))))
    return max(peak, sample_peak(audio))
