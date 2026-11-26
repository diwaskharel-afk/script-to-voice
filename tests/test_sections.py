import pytest

from script_to_voice.models import format_timecode, parse_timecode
from script_to_voice.sections import normalize_sections


@pytest.mark.parametrize("raw,expected", [
    (12, 12.0), (12.5, 12.5), ("12.5", 12.5), ("1:05", 65.0), ("01:05.5", 65.5),
    ("1:02:03", 3723.0), ("12s", 12.0), (" 0:07 ", 7.0),
])
def test_parse_timecode(raw, expected):
    assert parse_timecode(raw) == expected


@pytest.mark.parametrize("bad", ["abc", "", "1:2:3:4", True])
def test_parse_timecode_rejects(bad):
    with pytest.raises(ValueError):
        parse_timecode(bad)


def test_format_timecode():
    assert format_timecode(65.5) == "01:05.50"
    assert format_timecode(3723) == "1:02:03.00"


def test_normalize_sorts_numbers_and_clamps():
    raw = [
        {"start": "0:20", "end": "0:30", "title": "b"},
        {"start": "0:00", "end": "0:10", "title": "a"},
        {"start": "0:50", "end": "1:30", "title": "c"},  # end beyond video
    ]
    out = normalize_sections(raw, duration=60)
    assert [s.index for s in out] == [1, 2, 3]
    assert [s.title for s in out] == ["a", "b", "c"]
    assert out[2].end == 60


def test_normalize_cuts_overlaps_and_fills_missing_end():
    raw = [{"start": 0, "end": 20, "title": "a"}, {"start": 10, "title": "b"}]
    out = normalize_sections(raw, duration=30)
    assert out[0].end == 10  # cut where b begins
    assert out[1].end == 30  # no end -> video end


def test_normalize_skips_garbage_and_merges_tiny_sections():
    raw = [
        {"start": "nope", "end": 5},
        {"start": 0, "end": 10, "title": "a", "description": "one"},
        {"start": 10, "end": 10.5, "title": "blip", "description": "flash"},
        {"start": 11, "end": 20, "title": "b"},
    ]
    out = normalize_sections(raw, duration=30, min_seconds=1.5)
    assert [s.title for s in out] == ["a", "b"]
    assert "flash" in out[0].description


def test_normalize_drops_sections_past_video_end():
    out = normalize_sections([{"start": 0, "end": 5}, {"start": 99, "end": 120}], duration=10)
    assert len(out) == 1
