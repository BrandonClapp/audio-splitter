"""The DSP: time gate, center/coherence mask, harmonic mask. No model needed."""

import numpy as np
import pytest

from audio_splitter.solo import center_mask, extract_solo, fade_envelope, melody_mask, time_gate

SR = 44100


def tone(freq, dur=3.0, amp=0.4, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def saw(freq, dur=3.0, amp=0.45, harmonics=12, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    out = sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, harmonics + 1))
    return (out * (amp / np.max(np.abs(out)))).astype(np.float32)


def band_level(audio, freq, bw=6.0, sr=SR):
    """RMS-ish magnitude in a narrow band around `freq` of the mono sum."""
    mono = audio.mean(axis=0)
    spec = np.fft.rfft(mono * np.hanning(mono.shape[-1]))
    freqs = np.fft.rfftfreq(mono.shape[-1], 1 / sr)
    sel = (freqs > freq - bw) & (freqs < freq + bw)
    return float(np.sqrt(np.sum(np.abs(spec[sel]) ** 2)))


def band_change_db(before, after, freq):
    return 20 * np.log10(max(band_level(after, freq), 1e-12) / max(band_level(before, freq), 1e-12))


# ----------------------------------------------------------------------------- gate

def test_envelope_is_zero_outside_and_one_inside():
    env = fade_envelope(1000, 200, 800, fade=50)
    assert env.shape == (1000,)
    assert np.all(env[:200] == 0.0)
    assert np.all(env[800:] == 0.0)
    assert np.all(env[250:750] == 1.0)


def test_envelope_fades_are_monotonic():
    env = fade_envelope(1000, 200, 800, fade=50)
    rise, fall = env[200:250], env[750:800]
    assert np.all(np.diff(rise) > 0)
    assert np.all(np.diff(fall) < 0)
    assert 0.0 < rise[0] < rise[-1] < 1.0     # no flat step at either end of the ramp
    assert np.allclose(rise, fall[::-1])


def test_envelope_clamps_fade_to_half_the_window():
    env = fade_envelope(1000, 400, 500, fade=10_000)
    assert np.all(env[:400] == 0.0) and np.all(env[500:] == 0.0)
    assert 0.0 < env.max() <= 1.0


def test_envelope_handles_degenerate_windows():
    assert np.all(fade_envelope(100, 50, 50, fade=10) == 0.0)
    assert np.all(fade_envelope(100, 80, 20, fade=10) == 0.0)   # end before start


def test_time_gate_keeps_length_and_silences_outside():
    audio = np.tile(tone(440, 2.0), (2, 1))
    gated = time_gate(audio, 20_000, 60_000, fade=1000)
    assert gated.shape == audio.shape
    assert np.all(gated[:, :20_000] == 0.0)
    assert np.all(gated[:, 60_000:] == 0.0)
    assert np.max(np.abs(gated[:, 30_000:50_000])) == pytest.approx(
        np.max(np.abs(audio[:, 30_000:50_000])), rel=1e-6
    )


# ---------------------------------------------------------------------- center mask

def test_center_mask_removes_panned_keeps_center():
    """The plan's case: a center tone plus two near-coincident hard-panned tones.

    300 L and 301 R land in the same FFT bin, so their magnitudes balance out and the
    *phase-coherence* term is the only thing that can catch them — which is exactly the
    double-tracked-rhythm-guitar case.
    """
    center = tone(440)
    mix = np.stack([center + tone(300), center + tone(301)])

    out = center_mask(mix, strength=1.0)

    assert band_change_db(mix, out, 440) > -0.5
    assert band_change_db(mix, out, 300) < -3.0


def test_center_mask_annihilates_well_separated_panned_content():
    center = tone(440)
    mix = np.stack([center + tone(300), center + tone(1000)])

    out = center_mask(mix, strength=1.0)

    assert band_change_db(mix, out, 440) > -0.5
    assert band_change_db(mix, out, 300) < -40.0
    assert band_change_db(mix, out, 1000) < -40.0


def test_center_strength_zero_is_a_no_op():
    mix = np.stack([tone(440) + tone(300), tone(440) + tone(1000)])
    assert np.array_equal(center_mask(mix, strength=0.0), mix)


def test_center_pan_retargets_the_kept_position():
    # A tone panned right: balance = (0.6 - 0.2) / 0.8 = +0.5.
    mix = np.stack([tone(440, amp=0.2), tone(440, amp=0.6)])

    assert band_change_db(mix, center_mask(mix, pan=0.0), 440) < -3.0
    assert band_change_db(mix, center_mask(mix, pan=0.5), 440) > -0.5


def test_higher_strength_attenuates_more():
    center = tone(440)
    mix = np.stack([center + tone(300), center + tone(301)])
    gentle = band_change_db(mix, center_mask(mix, strength=0.5), 300)
    firm = band_change_db(mix, center_mask(mix, strength=2.0), 300)
    assert firm < gentle


# -------------------------------------------------------------------- harmonic mask

def test_melody_mask_keeps_the_lead_and_attenuates_the_chord():
    lead = saw(220)
    chord = sum(tone(f, amp=0.08) for f in (130.81, 164.81, 196.00))
    mix = np.stack([lead + chord, lead + chord]).astype(np.float32)

    out = melody_mask(mix, SR, harmonics=12, cents=60.0, floor=0.1)

    assert band_change_db(mix, out, 220) > -3.0      # fundamental survives
    assert band_change_db(mix, out, 440) > -3.0      # so does its 2nd harmonic
    for note in (130.81, 164.81, 196.00):
        assert band_change_db(mix, out, note) < -12.0


def test_harm_floor_sets_how_much_of_the_rest_survives():
    lead = saw(220)
    chord = sum(tone(f, amp=0.08) for f in (130.81, 164.81, 196.00))
    mix = np.stack([lead + chord, lead + chord]).astype(np.float32)

    quiet = band_change_db(mix, melody_mask(mix, SR, floor=0.02), 130.81)
    loud = band_change_db(mix, melody_mask(mix, SR, floor=0.30), 130.81)
    assert quiet < loud < -3.0


# ------------------------------------------------------------------------ pipeline

def test_extract_solo_is_silent_outside_the_window():
    stem = np.stack([saw(220, 4.0), saw(220, 4.0)])
    start, end = SR, 3 * SR

    solo = extract_solo(stem, SR, start, end, fade=1024, center_strength=1.0)

    assert solo.shape == stem.shape
    assert np.all(solo[:, :start] == 0.0)
    assert np.all(solo[:, end:] == 0.0)
    assert np.max(np.abs(solo[:, start:end])) > 0.0


def test_extract_solo_with_empty_window_returns_silence():
    stem = np.stack([saw(220, 1.0), saw(220, 1.0)])
    assert np.all(extract_solo(stem, SR, 5000, 5000, fade=100) == 0.0)
