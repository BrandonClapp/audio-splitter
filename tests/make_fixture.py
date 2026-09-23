"""Synthesize a fake 'song' for the integration test.

No copyrighted audio lives in this repo. The fixture has the four things the pipeline
has to tell apart: a click track, a bass line, hard-panned double-tracked rhythm chords,
and a center lead melody that only plays inside the solo window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SR = 44100
BPM = 120.0

# A minor pentatonic-ish set, in Hz.
ROOTS = [110.00, 146.83, 164.81, 130.81]
CHORDS = [
    (220.00, 261.63, 329.63),
    (293.66, 349.23, 440.00),
    (329.63, 392.00, 493.88),
    (261.63, 311.13, 392.00),
]
LEAD_NOTES = [440.00, 523.25, 587.33, 659.25, 587.33, 523.25, 440.00, 392.00]


def _saw(freq: float, t: np.ndarray, harmonics: int = 12, phase: float = 0.0) -> np.ndarray:
    out = np.zeros_like(t)
    for k in range(1, harmonics + 1):
        if freq * k >= SR / 2:
            break
        out += np.sin(2 * np.pi * freq * k * t + phase * k) / k
    return out


def _env(n: int, attack: float = 0.004, release: float = 0.35) -> np.ndarray:
    t = np.arange(n) / SR
    rise = np.clip(t / max(attack, 1e-6), 0.0, 1.0)
    return (rise * np.exp(-t / release)).astype(np.float32)


def _add(buf: np.ndarray, at: int, block: np.ndarray) -> None:
    end = min(buf.shape[-1], at + block.shape[-1])
    if end > at:
        buf[..., at:end] += block[..., : end - at]


def make_song(
    duration: float = 20.0,
    solo: tuple[float, float] = (8.0, 16.0),
    seed: int = 7,
) -> np.ndarray:
    """A `(2, n)` float32 fake song with a center lead only inside `solo`."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    mix = np.zeros((2, n), dtype=np.float32)
    beat = 60.0 / BPM
    n_beats = int(duration / beat)

    for b in range(n_beats):
        at = int(b * beat * SR)
        bar = (b // 4) % len(CHORDS)

        # click track — center, broadband
        click_n = int(0.006 * SR)
        click = (rng.standard_normal(click_n) * _env(click_n, 0.0002, 0.0015)).astype(np.float32)
        _add(mix, at, np.stack([click, click]) * (0.5 if b % 4 == 0 else 0.3))

        # bass — center, on the beat
        bass_n = int(beat * SR)
        t = np.arange(bass_n) / SR
        bass = (np.sin(2 * np.pi * ROOTS[bar] * t) * _env(bass_n, 0.01, 0.30)).astype(np.float32)
        _add(mix, at, np.stack([bass, bass]) * 0.30)

        # rhythm guitar — double-tracked, hard L and hard R, detuned so the two takes are
        # incoherent between channels, which is what the center mask is built to remove
        chord_n = int(beat * SR)
        t = np.arange(chord_n) / SR
        env = _env(chord_n, 0.005, 0.25)
        left = np.zeros(chord_n, dtype=np.float32)
        right = np.zeros(chord_n, dtype=np.float32)
        for note in CHORDS[bar]:
            left += (_saw(note * 0.999, t, 10, float(rng.uniform(0, np.pi))) * env).astype(np.float32)
            right += (_saw(note * 1.001, t, 10, float(rng.uniform(0, np.pi))) * env).astype(np.float32)
        _add(mix, at, np.stack([left, right]) * 0.11)

    # lead guitar — center, only inside the solo window
    solo_start, solo_end = solo
    note_dur = (solo_end - solo_start) / len(LEAD_NOTES)
    for i, freq in enumerate(LEAD_NOTES):
        at = int((solo_start + i * note_dur) * SR)
        note_n = int(note_dur * SR)
        t = np.arange(note_n) / SR
        vibrato = 1.0 + 0.006 * np.sin(2 * np.pi * 5.5 * t)
        lead = (_saw(freq, t * vibrato, 14) * _env(note_n, 0.01, note_dur * 0.9)).astype(np.float32)
        _add(mix, at, np.stack([lead, lead]) * 0.28)

    peak = float(np.max(np.abs(mix)))
    if peak > 0:
        mix *= np.float32(0.7 / peak)      # leave headroom so --no-normalize can't clip
    return mix


def write_fixture(path: Path, **kwargs) -> Path:
    """Write the fixture as a float WAV (lossless, so the null test measures the pipeline)."""
    import soundfile as sf

    song = make_song(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), song.T, SR, subtype="FLOAT")
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "fixture.wav")
    print(write_fixture(out))
