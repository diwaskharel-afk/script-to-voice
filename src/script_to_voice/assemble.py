"""Place each clip at its section's start time on one blank track.

The gaps are simply the silence in the blank track, so there is no per-gap logic.
Pydub's overlay is used instead of ffmpeg's amix/adelay, which makes output quieter
and has timing quirks.
"""
from __future__ import annotations

from pathlib import Path

from .media import load_audio, _segment_cls
from .models import Clip, Placement, Section, StvError, format_timecode, SKIPPED, ERROR

FADE_MS = 15


def assemble(
    sections: list[Section],
    clips: list[Clip],
    out_path: str | Path,
    total_duration: float | None = None,
    *,
    sample_rate: int = 44100,
    normalize_dbfs: float | None = -20.0,
    on_overflow: str = "trim",
) -> list[Placement]:
    """Mix clips onto a silent track as long as the video and export it. Returns placements."""
    AudioSegment = _segment_cls()
    by_index = {c.section_index: c for c in clips}
    placeable = [(s, by_index[s.index]) for s in sorted(sections, key=lambda s: s.start)
                 if s.index in by_index and by_index[s.index].status not in (SKIPPED, ERROR)
                 and by_index[s.index].path]
    if not placeable:
        raise StvError("nothing to assemble: no clips were generated")

    last_end = max(s.start + c.duration for s, c in placeable)
    length_s = max(total_duration or 0.0, last_end) + 0.05
    track = AudioSegment.silent(duration=int(length_s * 1000), frame_rate=sample_rate)

    placements: list[Placement] = []
    for i, (section, clip) in enumerate(placeable):
        seg = load_audio(clip.path, sample_rate)
        if normalize_dbfs is not None and seg.dBFS != float("-inf"):
            seg = seg.apply_gain(normalize_dbfs - seg.dBFS)
        start_ms = int(round(section.start * 1000))
        trimmed = False

        if on_overflow == "trim" and i + 1 < len(placeable):
            room_ms = int(round(placeable[i + 1][0].start * 1000)) - start_ms
            if len(seg) > room_ms > 0:  # would talk over the next line
                seg = seg[:room_ms]
                trimmed = True
        seg = seg.fade_in(FADE_MS).fade_out(FADE_MS if not trimmed else 60)
        track = track.overlay(seg, position=start_ms)

        end_s = (start_ms + len(seg)) / 1000.0
        next_start = placeable[i + 1][0].start if i + 1 < len(placeable) else length_s
        placements.append(Placement(section.index, section.start, round(end_s, 3), trimmed,
                                    round(max(0.0, next_start - end_s), 3)))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = out.suffix.lstrip(".").lower() or "wav"
    track.export(str(out), format=fmt, **({"bitrate": "192k"} if fmt == "mp3" else {}))
    return placements


def export_srt(sections: list[Section], clips: list[Clip], path: str | Path) -> None:
    """Subtitles of the narration; handy as captions or for checking timing in an editor."""
    by_index = {c.section_index: c for c in clips}

    def stamp(t: float) -> str:
        ms = int(round(t * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines, n = [], 0
    for s in sorted(sections, key=lambda s: s.start):
        c = by_index.get(s.index)
        if not c or not c.final_text or c.status in (SKIPPED, ERROR):
            continue
        n += 1
        lines += [str(n), f"{stamp(s.start)} --> {stamp(s.start + c.duration)}", c.final_text, ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def timeline_rows(sections: list[Section], clips: list[Clip]) -> list[dict]:
    by_index = {c.section_index: c for c in clips}
    rows = []
    for s in sections:
        c = by_index.get(s.index)
        rows.append({
            "#": s.index, "start": format_timecode(s.start), "slot (s)": round(s.slot, 1),
            "clip (s)": round(c.duration, 1) if c else None,
            "status": c.status if c else "pending", "speed": c.speed if c else None,
            "note": c.message if c else "",
        })
    return rows
