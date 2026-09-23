"""The invariant the whole design rests on: solo + backing == original."""

import numpy as np
import pytest

from audio_splitter.report import NULL_TOLERANCE, null_test
from audio_splitter.solo import complement, extract_solo

SR = 44100


def test_complement_reconstructs_the_original_for_random_input():
    rng = np.random.default_rng(0)
    for _ in range(20):
        original = rng.uniform(-1, 1, size=(2, 30_000)).astype(np.float32)
        mask = rng.uniform(0, 1, size=(2, 30_000)).astype(np.float32)
        solo = (original * mask).astype(np.float32)

        backing = complement(original, solo)

        assert null_test(original, solo, backing) < NULL_TOLERANCE


def test_complement_holds_when_the_solo_is_unrelated_to_the_original():
    """The solo comes from a *stem*, not from the original — it is not a subset of it."""
    rng = np.random.default_rng(1)
    original = rng.uniform(-1, 1, size=(2, 50_000)).astype(np.float32)
    solo = rng.uniform(-1, 1, size=(2, 50_000)).astype(np.float32)

    assert null_test(original, solo, complement(original, solo)) < NULL_TOLERANCE


def test_complement_holds_through_the_real_mask_chain():
    rng = np.random.default_rng(2)
    n = 4 * SR
    original = (0.5 * rng.standard_normal((2, n))).astype(np.float32)
    stem = (0.3 * rng.standard_normal((2, n))).astype(np.float32)

    solo = extract_solo(stem, SR, SR, 3 * SR, fade=1024, center_strength=1.0)
    backing = complement(original, solo)

    assert solo.shape == original.shape
    assert null_test(original, solo, backing) < NULL_TOLERANCE


def test_backing_carries_everything_the_solo_left_behind():
    """Outside the window the backing is bit-identical to the original."""
    rng = np.random.default_rng(3)
    n = 3 * SR
    original = (0.5 * rng.standard_normal((2, n))).astype(np.float32)
    stem = (0.3 * rng.standard_normal((2, n))).astype(np.float32)

    solo = extract_solo(stem, SR, SR, 2 * SR, fade=512, center_strength=1.0)
    backing = complement(original, solo)

    assert np.array_equal(backing[:, :SR], original[:, :SR])
    assert np.array_equal(backing[:, 2 * SR:], original[:, 2 * SR:])


def test_complement_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        complement(np.zeros((2, 100), np.float32), np.zeros((2, 90), np.float32))


def test_null_test_reports_real_error():
    original = np.ones((2, 10), np.float32)
    solo = np.zeros((2, 10), np.float32)
    assert null_test(original, solo, np.zeros((2, 10), np.float32)) == pytest.approx(1.0)
