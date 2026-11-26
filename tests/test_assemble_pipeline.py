"""Assembly (gap logic, no credits needed) and the whole pipeline on mocks."""
import pytest

from script_to_voice import pipeline as pl
from script_to_voice.assemble import assemble, export_srt, timeline_rows
from script_to_voice.media import load_audio
from script_to_voice.models import Clip, Section, StvError
from script_to_voice.project import Project
from script_to_voice.sections import load_sections
from script_to_voice.tts import MockTTS


def make_clip(tmp_path, index, seconds, text="word"):
    syn = MockTTS(seconds).synthesize(text)  # 1 word * seconds
    p = tmp_path / f"clip{index}.wav"
    p.write_bytes(syn.data)
    return Clip(index, str(p), seconds, 10, "ok", final_text=text)


def active_ranges(path, window_ms=20, threshold_db=-50):
    """[(start_s, end_s)] of non-silent audio."""
    seg = load_audio(path)
    ranges, start = [], None
    for i in range(0, len(seg), window_ms):
        loud = seg[i:i + window_ms].dBFS > threshold_db
        if loud and start is None:
            start = i
        if not loud and start is not None:
            ranges.append((start / 1000, i / 1000))
            start = None
    if start is not None:
        ranges.append((start / 1000, len(seg) / 1000))
    return ranges


def test_clips_land_at_start_times_with_silent_gaps(tmp_path):
    secs = [Section(1, 1.0, 5.0), Section(2, 6.0, 9.0), Section(3, 12.0, 15.0)]
    clips = [make_clip(tmp_path, 1, 2.0), make_clip(tmp_path, 2, 1.0), make_clip(tmp_path, 3, 2.5)]
    out = tmp_path / "out.wav"
    placements = assemble(secs, clips, out, total_duration=20.0)
    r = active_ranges(out)
    assert len(r) == 3
    for (a, b), start, dur in zip(r, (1.0, 6.0, 12.0), (2.0, 1.0, 2.5)):
        assert a == pytest.approx(start, abs=0.06) and b == pytest.approx(start + dur, abs=0.06)
    assert len(load_audio(out)) / 1000 == pytest.approx(20.05, abs=0.02)  # as long as the video
    assert [p.gap_after for p in placements][:2] == pytest.approx([3.0, 5.0], abs=0.02)


def test_overflow_is_trimmed_not_overlapped(tmp_path):
    secs = [Section(1, 0.0, 3.0), Section(2, 3.0, 6.0)]
    clips = [make_clip(tmp_path, 1, 5.0), make_clip(tmp_path, 2, 2.0)]  # clip 1 runs 2s into section 2
    out = tmp_path / "o.wav"
    placements = assemble(secs, clips, out, 8.0, on_overflow="trim")
    assert placements[0].trimmed and placements[0].end == pytest.approx(3.0, abs=0.01)
    placements = assemble(secs, clips, out, 8.0, on_overflow="overlap")
    assert not placements[0].trimmed


def test_skipped_and_error_clips_leave_silence(tmp_path):
    secs = [Section(1, 0, 3), Section(2, 4, 7)]
    clips = [make_clip(tmp_path, 1, 1.0), Clip(2, status="skipped")]
    out = tmp_path / "o.wav"
    assemble(secs, clips, out, 8.0)
    assert len(active_ranges(out)) == 1
    with pytest.raises(StvError):
        assemble(secs, [Clip(1, status="skipped")], out, 8.0)


def test_track_grows_if_last_clip_overruns_video(tmp_path):
    out = tmp_path / "o.wav"
    assemble([Section(1, 4.0, 5.0)], [make_clip(tmp_path, 1, 3.0)], out, total_duration=5.0)
    assert len(load_audio(out)) / 1000 >= 7.0


def test_mp3_export(tmp_path):
    out = tmp_path / "o.mp3"
    assemble([Section(1, 0, 3)], [make_clip(tmp_path, 1, 1.0)], out, 4.0)
    assert out.stat().st_size > 1000


def test_srt_and_timeline(tmp_path):
    secs = [Section(1, 1.0, 5.0)]
    clips = [make_clip(tmp_path, 1, 2.0, "Hello there")]
    export_srt(secs, clips, tmp_path / "n.srt")
    assert "00:00:01,000 --> 00:00:03,000\nHello there" in (tmp_path / "n.srt").read_text()
    assert timeline_rows(secs, clips)[0]["status"] == "ok"


# ------------------------------------------------------------------ full pipeline
def test_full_pipeline_on_mocks(settings, tmp_path):
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples" / "sample_sections.json"
    project = Project.create(settings.projects_dir, "e2e")
    project.duration = 60.0
    project.sections = load_sections(example, 60.0)
    svc = pl.Services.build(settings)
    svc.tts = MockTTS(0.5)  # slow narrator forces the fit loop to work
    out = pl.run_all(project, svc, skip_detect=True)

    assert out.exists() and project.output == str(out)
    assert len(project.clips) == 5 and not project.flagged
    for sec, clip in zip(project.sections, project.clips):
        assert clip.duration <= sec.slot  # every line fits its slot
    # clips are where the sections start
    r = active_ranges(out)
    assert [round(a) for a, _ in r] == [round(s.start) for s in project.sections]
    assert (project.output_dir / "narration.srt").exists()
    # state survives a reload
    assert Project.load(project.root).output == str(out)


def test_rerun_uses_cache_and_only_regenerates_edited_line(settings):
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples" / "sample_sections.json"
    project = Project.create(settings.projects_dir, "cache")
    project.duration = 60.0
    project.sections = load_sections(example, 60.0)
    svc = pl.Services.build(settings)
    tts = svc.tts = MockTTS(0.4)
    pl.run_all(project, svc, skip_detect=True)
    first = tts.calls
    pl.generate_clips(project, svc)
    assert tts.calls == first  # nothing changed, nothing spent
    project.sections[2].script = "A brand new line."
    pl.generate_clips(project, svc)
    assert tts.calls == first + 1  # exactly one regeneration


def test_pipeline_errors_are_clear(settings):
    project = Project.create(settings.projects_dir, "bad")
    svc = pl.Services.build(settings)
    with pytest.raises(StvError, match="no video"):
        pl.detect_sections(project, svc)
    with pytest.raises(StvError, match="no sections"):
        pl.write_script(project, svc)
    with pytest.raises(StvError, match="script is empty"):
        pl.generate_clips(project, svc)
    with pytest.raises(StvError, match="assemble"):
        pl.make_preview(project)


def test_tempo_and_preview_mux(tmp_path):
    from script_to_voice.media import audio_duration, change_tempo, mux_preview, video_duration, find_ffmpeg
    import subprocess

    src = tmp_path / "a.wav"
    src.write_bytes(MockTTS(1.0).synthesize("one two three four").data)  # 4s
    dst = tmp_path / "fast.wav"
    change_tempo(src, dst, 1.25)
    assert audio_duration(dst) == pytest.approx(3.2, abs=0.05)
    with pytest.raises(StvError):
        change_tempo(src, dst, 3.0)

    video = tmp_path / "v.mp4"  # 4s synthetic video
    subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=blue:s=160x120:d=4:r=10", "-pix_fmt", "yuv420p", str(video)], check=True)
    assert video_duration(video) == pytest.approx(4.0, abs=0.2)
    out = mux_preview(video, src, tmp_path / "p.mp4")
    assert out.exists() and out.stat().st_size > 0
