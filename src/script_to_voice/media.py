"""ffmpeg / pydub helpers.

pydub shells out to ffprobe for non-WAV files, and the ffmpeg bundled with
imageio-ffmpeg has no ffprobe. So we never let pydub probe: everything is decoded
to WAV by ffmpeg directly and then loaded with AudioSegment.from_wav.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path

from .models import StvError

# pydub warns at import if ffmpeg is not on PATH; we point it at our own binary afterwards.
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv", category=RuntimeWarning)


class MediaError(StvError):
    pass


_FFMPEG: str | None = None


def find_ffmpeg() -> str:
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    env = os.environ.get("FFMPEG_BINARY")
    if env and Path(env).exists():
        _FFMPEG = env
        return env
    found = shutil.which("ffmpeg")
    if found:
        _FFMPEG = found
        return found
    try:
        import imageio_ffmpeg

        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        return _FFMPEG
    except Exception as e:  # noqa: BLE001
        raise MediaError(
            "ffmpeg not found. Install it, set FFMPEG_BINARY, or `pip install imageio-ffmpeg`."
        ) from e


def _run(args: list[str], *, capture_stdout: bool = False) -> subprocess.CompletedProcess:
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise MediaError(f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc


def _segment_cls():
    from pydub import AudioSegment

    AudioSegment.converter = find_ffmpeg()
    return AudioSegment


def load_audio(source: str | Path | bytes, sample_rate: int = 44100):
    """Decode any audio (mp3, wav, ...) to a mono 16-bit AudioSegment at sample_rate."""
    AudioSegment = _segment_cls()
    if isinstance(source, (bytes, bytearray)):
        proc = subprocess.run(
            [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-f", "wav", "-ac", "1", "-ar", str(sample_rate), "-sample_fmt", "s16", "pipe:1"],
            input=bytes(source), capture_output=True,
        )
        if proc.returncode != 0:
            raise MediaError(f"could not decode audio: {proc.stderr.decode(errors='replace').strip()}")
        return AudioSegment.from_wav(io.BytesIO(proc.stdout))
    proc = _run(["-i", str(source), "-f", "wav", "-ac", "1", "-ar", str(sample_rate),
                 "-sample_fmt", "s16", "pipe:1"])
    return AudioSegment.from_wav(io.BytesIO(proc.stdout))


def audio_duration(source: str | Path | bytes) -> float:
    return len(load_audio(source)) / 1000.0


def change_tempo(src: str | Path, dst: str | Path, factor: float) -> None:
    """Speed audio up (factor > 1) without changing pitch."""
    if not 0.5 <= factor <= 2.0:
        raise MediaError(f"tempo factor {factor} outside ffmpeg atempo range 0.5-2.0")
    _run(["-y", "-i", str(src), "-filter:a", f"atempo={factor:.5f}",
          "-ac", "1", "-ar", "44100", str(dst)])


def video_duration(path: str | Path) -> float:
    """Read duration from `ffmpeg -i` output (no ffprobe needed)."""
    proc = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True)
    text = proc.stderr.decode(errors="replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not m:
        raise MediaError(f"could not read video duration of {path}")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def split_video(path: str | Path, chunk_seconds: float, out_dir: str | Path) -> list[tuple[Path, float]]:
    """Cut a video into chunks. Returns [(chunk_path, offset_seconds)]."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = video_duration(path)
    chunks: list[tuple[Path, float]] = []
    start, i = 0.0, 0
    while start < total - 0.5:
        dst = out / f"chunk_{i:03d}.mp4"
        _run(["-y", "-ss", f"{start:.3f}", "-t", f"{chunk_seconds:.3f}", "-i", str(path),
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-an", str(dst)])
        chunks.append((dst, start))
        start += chunk_seconds
        i += 1
    return chunks


def mux_preview(video: str | Path, audio: str | Path, out: str | Path) -> Path:
    """Put the narration over the video (replacing its own audio) for a quick preview."""
    _run(["-y", "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
          "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)])
    return Path(out)
