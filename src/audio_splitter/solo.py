"""Turning the guitar stem into a lead-only estimate, and its exact complement.

Two independent masks refine the stem inside the user's window:

* a **center/coherence mask**, which keeps what sits at a given pan position *and* is
  phase-coherent between the channels — that is what strips hard-panned, double-tracked
  rhythm guitars, which is how rhythm parts are usually recorded;
* a **melody/harmonic mask**, which keeps only bins near the harmonics of the tracked
  monophonic pitch.

Everything the solo estimate doesn't claim ends up in `backing = original - solo`, so a
weak estimate degrades gracefully: the backing stays complete either way.
"""

from __future__ import annotations

import numpy as np

N_FFT = 4096
HOP = 1024
EPS = 1e-10


# --------------------------------------------------------------------------- gating

def fade_envelope(n: int, start: int, end: int, fade: int) -> np.ndarray:
    """Full-length envelope: exactly 0 outside `[start, end)`, raised-cosine ramps inside.

    The fades live *inside* the window, so the result is sample-exact zero everywhere
    outside it and there is nothing to click on at the boundaries.
    """
    env = np.zeros(n, dtype=np.float32)
    start = max(0, min(n, int(start)))
    end = max(start, min(n, int(end)))
    width = end - start
    if width == 0:
        return env

    env[start:end] = 1.0
    fade = max(0, min(int(fade), width // 2))
    if fade:
        # 0 -> 1 raised cosine, endpoints excluded so the ramp never sits flat at 0 or 1.
        ramp = 0.5 * (1.0 - np.cos(np.pi * (np.arange(1, fade + 1) / (fade + 1))))
        env[start:start + fade] = ramp
        env[end - fade:end] = ramp[::-1]
    return env


def time_gate(audio: np.ndarray, start: int, end: int, fade: int) -> np.ndarray:
    """Apply `fade_envelope` to `(c, n)` audio, keeping full length."""
    env = fade_envelope(audio.shape[-1], start, end, fade)
    return (audio * env[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------- masks

def center_mask(
    audio: np.ndarray,
    *,
    strength: float = 1.0,
    pan: float = 0.0,
    n_fft: int = N_FFT,
    hop: int = HOP,
) -> np.ndarray:
    """Keep bins that sit at `pan` and are phase-coherent across L/R.

    `pan` is -1 (hard left) .. 0 (center) .. +1 (hard right). `strength` is the exponent
    on the mask, so `strength=0` is a no-op (`x ** 0 == 1`) and larger is more aggressive.
    """
    import librosa

    if strength <= 0 or audio.shape[0] < 2:
        return audio.astype(np.float32)

    n = audio.shape[-1]
    left = librosa.stft(np.ascontiguousarray(audio[0]), n_fft=n_fft, hop_length=hop)
    right = librosa.stft(np.ascontiguousarray(audio[1]), n_fft=n_fft, hop_length=hop)

    mag_l, mag_r = np.abs(left), np.abs(right)
    balance = (mag_r - mag_l) / (mag_r + mag_l + EPS)          # -1 (hard L) .. +1 (hard R)
    pan_score = np.clip(1.0 - np.abs(balance - pan), 0.0, 1.0)
    coherence = np.clip(np.cos(np.angle(left) - np.angle(right)), 0.0, 1.0)
    mask = (pan_score * coherence).astype(np.float32) ** strength

    out = np.stack([
        librosa.istft(left * mask, n_fft=n_fft, hop_length=hop, length=n),
        librosa.istft(right * mask, n_fft=n_fft, hop_length=hop, length=n),
    ])
    return out.astype(np.float32)


def harmonic_weights(
    f0: np.ndarray,
    freqs: np.ndarray,
    *,
    harmonics: int = 12,
    cents: float = 60.0,
    floor: float = 0.1,
    n_frames: int | None = None,
    block: int = 256,
) -> np.ndarray:
    """`(n_bins, n_frames)` mask keeping bins near `k * f0`, Gaussian roll-off in cents.

    Built in time-blocks: the naive `(freq x harmonic x frame)` array is ~1 GB for a
    full-length track. Frames with no `f0` (unvoiced, or past the end of the pitch track)
    collapse to `floor`.
    """
    n_pitch = int(f0.shape[0])
    n_frames = n_pitch if n_frames is None else int(n_frames)
    mask = np.full((freqs.shape[0], n_frames), np.float32(floor), dtype=np.float32)

    log_freqs = (1200.0 * np.log2(np.maximum(freqs, 1e-6))).astype(np.float32)
    ks = np.arange(1, harmonics + 1, dtype=np.float32)[:, None]

    usable = min(n_pitch, n_frames)
    for lo in range(0, usable, block):
        hi = min(lo + block, usable)
        f0_block = np.asarray(f0[lo:hi], dtype=np.float32)
        voiced = np.isfinite(f0_block) & (f0_block > 0)
        if not voiced.any():
            continue

        log_harm = 1200.0 * np.log2(ks * np.where(voiced, f0_block, 1.0)[None, :])  # (K, B)
        delta = log_freqs[:, None, None] - log_harm[None, :, :]                     # (F, K, B)
        block_mask = np.exp(-0.5 * (delta / np.float32(cents)) ** 2).max(axis=1)    # (F, B)
        block_mask *= voiced[None, :]
        mask[:, lo:hi] = floor + (1.0 - floor) * block_mask
    return mask


def melody_mask(
    audio: np.ndarray,
    sr: int,
    *,
    harmonics: int = 12,
    cents: float = 60.0,
    floor: float = 0.1,
    fmin: float = 80.0,
    fmax: float = 1400.0,
    n_fft: int = N_FFT,
    hop: int = HOP,
) -> np.ndarray:
    """Keep only the harmonic series of the tracked lead pitch.

    `librosa.pyin` runs on the mono sum with the STFT's own frame/hop so the two grids
    line up; the frame counts are then trimmed to the shorter of the two rather than
    trusted to match. `floor` leaves pick attack and distortion texture in place.
    """
    import librosa

    mono = np.ascontiguousarray(audio.mean(axis=0, dtype=np.float32))
    n = audio.shape[-1]

    f0, _voiced, _prob = librosa.pyin(
        mono, fmin=fmin, fmax=fmax, sr=sr, frame_length=n_fft, hop_length=hop,
    )

    spec = np.stack([
        librosa.stft(np.ascontiguousarray(ch), n_fft=n_fft, hop_length=hop) for ch in audio
    ])
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = harmonic_weights(
        f0, freqs,
        harmonics=harmonics, cents=cents, floor=floor,
        n_frames=spec.shape[-1],
    )

    out = np.stack([
        librosa.istft(ch * mask, n_fft=n_fft, hop_length=hop, length=n) for ch in spec
    ])
    return out.astype(np.float32)


# ------------------------------------------------------------------------- pipeline

def extract_solo(
    stem: np.ndarray,
    sr: int,
    start: int,
    end: int,
    *,
    fade: int,
    center_strength: float = 1.0,
    center_pan: float = 0.0,
    use_melody_mask: bool = False,
    harmonics: int = 12,
    harm_cents: float = 60.0,
    harm_floor: float = 0.1,
    n_fft: int = N_FFT,
    hop: int = HOP,
) -> np.ndarray:
    """Gate the stem to `[start, end)` and refine it to a lead-only estimate.

    The masks run on a slice of the stem rather than the whole track: it's faster, and it
    puts the STFT's edge effects exactly where the fades are.
    """
    n = stem.shape[-1]
    start = max(0, min(n, int(start)))
    end = max(start, min(n, int(end)))
    env = fade_envelope(n, start, end, fade)

    solo = np.zeros_like(stem, dtype=np.float32)
    if end == start:
        return solo

    lo = max(0, start - fade)
    hi = min(n, end + fade)
    piece = np.ascontiguousarray(stem[:, lo:hi].astype(np.float32))

    piece = center_mask(piece, strength=center_strength, pan=center_pan, n_fft=n_fft, hop=hop)
    if use_melody_mask:
        piece = melody_mask(
            piece, sr,
            harmonics=harmonics, cents=harm_cents, floor=harm_floor,
            n_fft=n_fft, hop=hop,
        )

    solo[:, lo:hi] = piece * env[None, lo:hi]
    return solo


def complement(original: np.ndarray, solo: np.ndarray) -> np.ndarray:
    """`backing = original - solo`, sample-exact.

    Whatever the solo estimate didn't claim — rhythm guitar, drums, bass, vocals, keys,
    and the separation model's own error — lands here. The two outputs sum back to the
    original by construction.
    """
    if original.shape != solo.shape:
        raise ValueError(f"shape mismatch: original {original.shape} vs solo {solo.shape}")
    return (original - solo).astype(np.float32)
