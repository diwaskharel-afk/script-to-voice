"""Core data types shared by every pipeline step."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any


class StvError(Exception):
    """Base class for errors we want to show to the user as-is."""


def parse_timecode(value: Any) -> float:
    """Accept 12, 12.5, '12.5', '1:05', '01:05.5', '1:02:03', '12s'. Return seconds."""
    if isinstance(value, bool):
        raise ValueError(f"not a timecode: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().lower().rstrip("s").strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    m = re.fullmatch(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", s)
    if not m:
        raise ValueError(f"not a timecode: {value!r}")
    hours = int(m.group(1) or 0)
    return hours * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def format_timecode(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", re.sub(r"<[^>]+>|\[[^\]]+\]", " ", text)))


@dataclass
class Section:
    """One moment in the video that gets one line of narration."""

    index: int
    start: float
    end: float
    title: str = ""
    description: str = ""  # what happens on screen
    script: str = ""  # narration line

    @property
    def slot(self) -> float:
        """Seconds available for narration."""
        return max(0.0, self.end - self.start)

    def word_budget(self, words_per_second: float, tail_padding: float = 0.0) -> int:
        return max(1, int((self.slot - tail_padding) * words_per_second))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Section":
        return cls(
            index=int(d["index"]),
            start=parse_timecode(d["start"]),
            end=parse_timecode(d["end"]),
            title=d.get("title", ""),
            description=d.get("description", ""),
            script=d.get("script", ""),
        )


# Clip statuses, from best to worst.
OK, SPED_UP, SHORTENED, OVERFLOW, SKIPPED, ERROR = (
    "ok", "sped_up", "shortened", "overflow", "skipped", "error",
)
NEEDS_HUMAN = {OVERFLOW, ERROR}


@dataclass
class Clip:
    """The generated, fitted audio for one section."""

    section_index: int
    path: str = ""
    duration: float = 0.0
    slot: float = 0.0
    status: str = SKIPPED
    speed: float = 1.0  # tempo factor applied after generation
    attempts: int = 0  # number of TTS generations used
    original_text: str = ""
    final_text: str = ""
    message: str = ""

    @property
    def needs_human(self) -> bool:
        return self.status in NEEDS_HUMAN

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Clip":
        return cls(**d)


@dataclass
class Issue:
    """A script-accuracy problem found by the verification pass."""

    section_index: int
    problem: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Placement:
    """Where a clip ended up on the final track."""

    section_index: int
    start: float
    end: float
    trimmed: bool = False
    gap_after: float = 0.0
