"""Turn raw timestamps from a video model (or a hand-written file) into clean sections."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Section, StvError, parse_timecode


def normalize_sections(
    raw: list[dict[str, Any]], duration: float | None = None, min_seconds: float = 1.5
) -> list[Section]:
    """Sort, clamp, de-overlap, drop too-short sections, and number from 1.

    A missing/invalid end becomes the next section's start (or the video end).
    A section that overlaps the next one is cut where the next begins.
    """
    items = []
    for r in raw:
        try:
            start = parse_timecode(r["start"])
        except (KeyError, ValueError):
            continue  # a section without a usable start can't be placed
        end = None
        if r.get("end") not in (None, ""):
            try:
                end = parse_timecode(r["end"])
            except ValueError:
                end = None
        items.append({
            "start": max(0.0, start), "end": end,
            "title": str(r.get("title", "")).strip(),
            "description": str(r.get("description", "")).strip(),
            "script": str(r.get("script", "")).strip(),
        })
    items.sort(key=lambda r: r["start"])

    if duration is not None:
        items = [r for r in items if r["start"] < duration]

    out: list[Section] = []
    for i, r in enumerate(items):
        nxt = items[i + 1]["start"] if i + 1 < len(items) else duration
        end = r["end"]
        if end is None or end <= r["start"]:
            end = nxt if nxt is not None else r["start"] + 5.0
        if nxt is not None and end > nxt:
            end = nxt
        if duration is not None and end > duration:
            end = duration
        if end - r["start"] < min_seconds:
            # too short to narrate: fold its description into the previous section
            if out and r["description"]:
                out[-1].description = (out[-1].description + " " + r["description"]).strip()
            continue
        out.append(Section(0, round(r["start"], 2), round(end, 2), r["title"], r["description"], r["script"]))
    for n, s in enumerate(out, 1):
        s.index = n
    return out


def load_sections(path: str | Path, duration: float | None = None, min_seconds: float = 0.0) -> list[Section]:
    """Load hand-written slots: a JSON list of {start, end, title, description, script}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("sections", [])
    if not isinstance(data, list) or not data:
        raise StvError(f"{path}: expected a non-empty JSON list of sections")
    return normalize_sections(data, duration, min_seconds)


def save_sections(sections: list[Section], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps([s.to_dict() for s in sections], indent=2, ensure_ascii=False), encoding="utf-8"
    )
