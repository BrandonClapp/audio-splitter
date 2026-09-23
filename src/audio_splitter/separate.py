"""demucs wrapper: device selection, source separation, and an on-disk stem cache.

The cache is the reason the tuning loop is usable. Separation is the only slow step, and
the user iterates on the window and the mask knobs by ear — so stems are keyed by
(audio, model, shifts, overlap) and every run after the first takes seconds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .io_audio import hash_audio, read_float_wav, write_float_wav

DEFAULT_MODEL = "htdemucs_6s"
CACHE_ROOT = Path.home() / ".cache" / "audio-splitter" / "stems"


class SeparationError(RuntimeError):
    pass


def pick_device(requested: str = "auto") -> str:
    """Resolve `auto` to mps when it's available, else cpu."""
    import torch

    requested = (requested or "auto").lower()
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SeparationError("--device mps requested but torch reports MPS unavailable")
    return requested


def load_separator(model: str, device: str, shifts: int, overlap: float, progress: bool):
    from demucs.api import Separator

    try:
        return Separator(
            model=model, device=device, shifts=shifts, overlap=overlap, progress=progress,
        )
    except Exception as exc:  # demucs raises bare exceptions for unknown model names
        raise SeparationError(f"could not load model {model!r}: {exc}") from exc


def sources_of(separator) -> list[str]:
    return list(separator.model.sources)


def _cache_key(audio: np.ndarray, sr: int, model: str, shifts: int, overlap: float) -> str:
    return hash_audio(audio, sr, model, shifts, round(float(overlap), 6))


def _read_cache(path: Path) -> dict[str, np.ndarray] | None:
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        return {name: read_float_wav(path / f"{name}.wav")[0] for name in meta["sources"]}
    except (OSError, ValueError, KeyError):
        return None


def _write_cache(path: Path, stems: dict[str, np.ndarray], sr: int, meta: dict) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        for name, stem in stems.items():
            write_float_wav(path / f"{name}.wav", stem, sr)
        (path / "meta.json").write_text(json.dumps({"sources": list(stems), **meta}, indent=2))
    except OSError as exc:  # a full or unwritable cache must never fail the run
        print(f"  warning: could not write stem cache ({exc})")


def separate(
    audio: np.ndarray,
    sr: int,
    *,
    model: str = DEFAULT_MODEL,
    device: str = "auto",
    shifts: int = 1,
    overlap: float = 0.25,
    use_cache: bool = True,
    verbose: bool = False,
    require_sources: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], str, bool]:
    """Separate `(2, n)` float32 audio into named stems.

    Returns `(stems, device_used, from_cache)`. Stems are `(2, n)` float32, the same
    length as the input and at the original level, so `original - stem` is exact.
    """
    import torch

    key = _cache_key(audio, sr, model, shifts, overlap)
    cache_dir = CACHE_ROOT / key
    if use_cache:
        cached = _read_cache(cache_dir)
        if cached is not None and (
            require_sources is None or all(name in cached for name in require_sources)
        ):
            if verbose:
                print(f"  stem cache hit: {cache_dir}")
            return cached, "cache", True

    device = pick_device(device)
    separator = load_separator(model, device, shifts, overlap, progress=verbose)

    if require_sources:
        available = sources_of(separator)
        missing = [name for name in require_sources if name not in available]
        if missing:
            raise SeparationError(
                f"model {model!r} has no {', '.join(repr(m) for m in missing)} source — "
                f"its sources are {', '.join(available)}."
                + ("" if "guitar" in available else " Use --model htdemucs_6s.")
            )

    wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))

    def run(sep, dev: str) -> dict[str, np.ndarray]:
        started = time.monotonic()
        _origin, out = sep.separate_tensor(wav, sr)
        stems = {name: t.detach().to("cpu").numpy().astype(np.float32) for name, t in out.items()}
        if verbose:
            print(f"  separated on {dev} in {time.monotonic() - started:.1f}s")
        return stems

    try:
        stems = run(separator, device)
        bad = device != "cpu" and any(not np.all(np.isfinite(s)) for s in stems.values())
        if bad:
            print(f"  warning: {device} produced non-finite samples — falling back to cpu")
            raise SeparationError("non-finite output")
    except Exception as exc:
        if device == "cpu":
            raise SeparationError(f"separation failed: {exc}") from exc
        print(f"  warning: separation on {device} failed ({exc}) — retrying on cpu")
        device = "cpu"
        separator = load_separator(model, device, shifts, overlap, progress=verbose)
        stems = run(separator, device)

    if use_cache:
        _write_cache(
            cache_dir, stems, sr,
            {"model": model, "shifts": shifts, "overlap": overlap, "samplerate": sr},
        )
    return stems, device, False
