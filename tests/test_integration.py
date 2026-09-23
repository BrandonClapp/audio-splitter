"""End-to-end on a synthesized song. Marked slow: it runs the real demucs model."""

import contextlib
import io
import re

import numpy as np
import pytest

from audio_splitter import SAMPLE_RATE
from audio_splitter.cli import main
from audio_splitter.io_audio import decode, duration_seconds, encode_mp3
from audio_splitter.report import NULL_TOLERANCE, rms

from make_fixture import write_fixture

pytestmark = pytest.mark.slow

SOLO_START, SOLO_END = 8.0, 16.0
FRAME = 1152 / SAMPLE_RATE      # one MP3 frame


def null_db(residual: np.ndarray, reference: np.ndarray) -> float:
    return 20 * np.log10(max(rms(residual), 1e-12) / rms(reference))


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """Run the full pipeline once; every assertion below reads the same outputs."""
    out = tmp_path_factory.mktemp("render")
    src = write_fixture(out / "fixture.wav", duration=20.0, solo=(SOLO_START, SOLO_END))

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = main([
            str(src),
            "--from", str(SOLO_START), "--to", str(SOLO_END),
            "--no-normalize", "-v",
        ])
    assert code == 0, f"pipeline exited {code}:\n{stdout.getvalue()}"
    return src, out / "fixture.solo.mp3", out / "fixture.backing.mp3", stdout.getvalue()


@pytest.fixture(scope="module")
def codec_floor(rendered):
    """How well a plain re-encode of this fixture survives MP3 at all.

    The fixture's click track is white-noise bursts, which libmp3lame handles poorly, so
    the floor here (~-33 dB) is much worse than real music would give. Comparing against
    it measures the pipeline rather than the codec.
    """
    src, solo_path, _backing, _out = rendered
    original = decode(src)
    control = solo_path.parent / "control.mp3"
    encode_mp3(control, original, SAMPLE_RATE, bitrate="320k")
    reencoded = decode(control)
    n = min(original.shape[-1], reencoded.shape[-1])
    return null_db(reencoded[:, :n] - original[:, :n], original[:, :n])


def test_both_outputs_exist(rendered):
    _src, solo, backing, _out = rendered
    assert solo.is_file() and solo.stat().st_size > 0
    assert backing.is_file() and backing.stat().st_size > 0


def test_durations_match_the_source(rendered):
    src, solo, backing, _out = rendered
    expected = duration_seconds(src)
    assert duration_seconds(solo) == pytest.approx(expected, abs=FRAME)
    assert duration_seconds(backing) == pytest.approx(expected, abs=FRAME)


def test_float_stage_split_is_lossless(rendered):
    """The headline invariant, as the run itself reported it."""
    *_ , output = rendered
    match = re.search(r"max\|\(solo \+ backing\) - original\| = ([0-9.eE+-]+)", output)
    assert match, output
    assert float(match.group(1)) < NULL_TOLERANCE
    assert "null test" in output and "OK" in output


def test_decoded_sum_nulls_at_the_codec_floor(rendered, codec_floor):
    """After MP3, `solo + backing` is as close to the original as the codec allows."""
    src, solo_path, backing_path, _out = rendered
    original = decode(src)
    solo, backing = decode(solo_path), decode(backing_path)

    n = min(original.shape[-1], solo.shape[-1], backing.shape[-1])
    residual = (solo[:, :n] + backing[:, :n]) - original[:, :n]

    assert null_db(residual, original[:, :n]) < codec_floor + 3.0
    assert codec_floor < -25.0      # sanity: the control itself must be reasonable


def test_solo_is_confined_to_the_window(rendered):
    _src, solo_path, _backing, _out = rendered
    solo = decode(solo_path)
    lo, hi = int(SOLO_START * SAMPLE_RATE), int(SOLO_END * SAMPLE_RATE)

    inside = rms(solo[:, lo:hi])
    outside = rms(np.concatenate([solo[:, :lo], solo[:, hi:]], axis=-1))

    assert inside > 0.0
    assert 20 * np.log10(outside / inside) < -40.0


def test_backing_keeps_the_song_outside_the_window(rendered, codec_floor):
    src, _solo, backing_path, _out = rendered
    original, backing = decode(src), decode(backing_path)
    lo = int(SOLO_START * SAMPLE_RATE)

    # Before the solo starts the backing *is* the song, down to the codec floor.
    residual = backing[:, :lo] - original[:, :lo]
    assert null_db(residual, original[:, :lo]) < codec_floor + 3.0


def test_preview_encodes_only_the_window(tmp_path):
    src = write_fixture(tmp_path / "fixture.wav", duration=20.0, solo=(SOLO_START, SOLO_END))

    assert main([str(src), "--from", str(SOLO_START), "--to", str(SOLO_END), "--preview"]) == 0

    expected = (SOLO_END + 2.0) - (SOLO_START - 2.0)
    assert duration_seconds(tmp_path / "fixture.solo.mp3") == pytest.approx(expected, abs=0.1)
    assert duration_seconds(tmp_path / "fixture.backing.mp3") == pytest.approx(expected, abs=0.1)


def test_model_without_a_guitar_source_fails_clearly(tmp_path, capsys):
    src = write_fixture(tmp_path / "fixture.wav", duration=2.0, solo=(0.5, 1.5))

    code = main([str(src), "--from", "0.5", "--to", "1.5", "--model", "htdemucs", "--no-cache"])

    assert code == 1
    err = capsys.readouterr().err
    assert "no 'guitar' source" in err and "vocals" in err


def test_unknown_source_name_fails_clearly(tmp_path, capsys):
    src = write_fixture(tmp_path / "fixture.wav", duration=2.0, solo=(0.5, 1.5))

    code = main([str(src), "--from", "0.5", "--to", "1.5", "--source", "guitar,kazoo",
                 "--no-cache"])

    assert code == 1
    assert "'kazoo'" in capsys.readouterr().err


def test_source_flag_moves_energy_out_of_the_backing(tmp_path):
    """Adding stems to --source must leave strictly less behind in the backing."""
    src = write_fixture(tmp_path / "fixture.wav", duration=20.0, solo=(SOLO_START, SOLO_END))
    lo, hi = int(SOLO_START * SAMPLE_RATE), int(SOLO_END * SAMPLE_RATE)
    levels = []

    for sources, outdir in (("guitar", "narrow"), ("guitar,other,vocals", "wide")):
        out = tmp_path / outdir
        assert main([str(src), "--from", str(SOLO_START), "--to", str(SOLO_END),
                     "--source", sources, "-o", str(out), "--no-normalize"]) == 0
        levels.append(rms(decode(out / "fixture.backing.mp3")[:, lo:hi]))

    assert levels[1] < levels[0]
