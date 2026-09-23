"""Timecode parsing, and the one window check that needs no audio."""

import pytest

from audio_splitter.cli import main, parse_timecode


@pytest.mark.parametrize("text,expected", [
    ("0", 0.0),
    ("90", 90.0),
    ("90.25", 90.25),
    ("2:14", 134.0),
    ("0:00", 0.0),
    ("2:14.500", 134.5),
    ("10:00", 600.0),
    ("1:02:03", 3723.0),
    ("1:02:03.250", 3723.25),
    ("  2:14  ", 134.0),
])
def test_parses_supported_forms(text, expected):
    assert parse_timecode(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", [
    "",
    "   ",
    "abc",
    "-5",
    "2:75",          # seconds field out of range
    "1:70:00",       # minutes field out of range
    "1:2:3:4",       # too many fields
    "2.5:14",        # fractional minutes
    "2:",
    ":14",
    "2::14",
    "1e3",
    None,
    12,
])
def test_rejects_malformed(text):
    with pytest.raises(ValueError):
        parse_timecode(text)


def test_seconds_only_may_exceed_sixty():
    # A bare number is seconds, so 134 is legal where 2:134 is not.
    assert parse_timecode("134") == 134.0


def test_end_must_follow_start(capsys):
    code = main(["nonexistent.mp3", "--from", "2:48", "--to", "2:14"])
    assert code == 2
    assert "must be after" in capsys.readouterr().err


def test_equal_window_is_rejected(capsys):
    code = main(["nonexistent.mp3", "--from", "10", "--to", "10"])
    assert code == 2
    assert "must be after" in capsys.readouterr().err


def test_bad_timecode_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        main(["nonexistent.mp3", "--from", "banana", "--to", "2:14"])
    assert exc.value.code == 2


def test_empty_source_list_is_rejected(capsys, tmp_path):
    audio = tmp_path / "x.wav"
    audio.write_bytes(b"")
    code = main([str(audio), "--from", "0", "--to", "1", "--source", " , "])
    assert code == 2
    assert "--source needs at least one stem" in capsys.readouterr().err
