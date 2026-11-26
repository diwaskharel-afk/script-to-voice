"""The workflow, one function per step, so a UI can pause for human edits between steps."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .assemble import assemble, export_srt
from .cache import ClipCache
from .config import Settings
from .fit import FitEngine
from .llm import LLM, GeminiLLM, MockLLM, Persona, ScriptContext, load_personas
from .media import mux_preview, video_duration
from .models import StvError, count_words
from .pronunciation import Glossary
from .project import Project
from .tts import TTS, ElevenLabsTTS, MockTTS

Progress = Callable[[str, float], None]


def _noop(msg: str, frac: float) -> None:  # pragma: no cover
    pass


@dataclass
class Services:
    settings: Settings
    llm: LLM
    tts: TTS
    cache: ClipCache
    glossary: Glossary

    @classmethod
    def build(cls, settings: Settings, glossary: Glossary | None = None) -> "Services":
        llm: LLM = GeminiLLM(settings) if settings.llm_provider == "gemini" else MockLLM(settings)
        tts: TTS = ElevenLabsTTS(settings) if settings.tts_provider == "elevenlabs" else MockTTS()
        return cls(settings, llm, tts, ClipCache(settings.cache_dir), glossary or Glossary())

    @property
    def is_mock(self) -> bool:
        return self.settings.llm_provider == "mock" or self.settings.tts_provider == "mock"


def _context(project: Project, svc: Services, personas: dict[str, Persona] | None = None) -> ScriptContext:
    personas = personas or load_personas(Path("personas"))
    persona = personas.get(project.persona) or next(iter(personas.values()))
    s = svc.settings
    return ScriptContext(persona, project.document, project.notes, s.words_per_second, s.tail_padding)


# ---------------------------------------------------------------- step 2: timestamps
def detect_sections(project: Project, svc: Services, progress: Progress = _noop) -> None:
    if not project.video:
        raise StvError("no video set on this project")
    progress("Reading video length", 0.05)
    project.duration = video_duration(project.video)
    progress("Finding the moments in the video (this can take a minute)", 0.2)
    project.sections = svc.llm.detect_sections(Path(project.video), project.document, project.notes)
    project.clips, project.issues, project.placements = [], [], []
    project.save()
    progress(f"Found {len(project.sections)} sections", 1.0)


# ---------------------------------------------------------------- step 3: script
def write_script(project: Project, svc: Services, personas: dict[str, Persona] | None = None,
                 progress: Progress = _noop) -> None:
    if not project.sections:
        raise StvError("no sections yet: detect them or load them from a file first")
    progress("Writing the script", 0.2)
    ctx = _context(project, svc, personas)
    lines = svc.llm.write_script(project.sections, ctx)
    for s in project.sections:
        s.script = lines.get(s.index, "").strip()
    project.clips, project.placements = [], []
    project.save()
    progress("Script written", 1.0)


def verify_script(project: Project, svc: Services, progress: Progress = _noop) -> None:
    progress("Checking the script against the documentation", 0.3)
    project.issues = svc.llm.verify(project.sections, project.document)
    project.save()
    progress(f"{len(project.issues)} possible problems found", 1.0)


def over_budget(project: Project, svc: Services) -> list[int]:
    """Sections whose script is already longer than the rule-of-thumb word budget."""
    s = svc.settings
    return [sec.index for sec in project.sections
            if count_words(sec.script) > sec.word_budget(s.words_per_second, s.tail_padding) * 1.2]


# ---------------------------------------------------------------- steps 4-5: voice + fit
def generate_clips(project: Project, svc: Services, personas: dict[str, Persona] | None = None,
                   progress: Progress = _noop) -> None:
    if not any(s.script.strip() for s in project.sections):
        raise StvError("the script is empty")
    engine = FitEngine(svc.tts, svc.llm, svc.cache, svc.settings, _context(project, svc, personas), svc.glossary)
    secs = sorted(project.sections, key=lambda s: s.start)
    clips = []
    for i, sec in enumerate(secs):
        progress(f"Voicing section {sec.index} of {len(secs)}", i / len(secs))
        prev_text = secs[i - 1].script if i > 0 else ""
        next_text = secs[i + 1].script if i + 1 < len(secs) else ""
        clip = engine.fit(sec, prev_text, next_text, project.clips_dir / f"section_{sec.index:02d}.wav")
        if clip.final_text and clip.final_text != sec.script:
            sec.script = clip.final_text  # keep the shortened line so edits and reruns start from it
        clips.append(clip)
    project.clips = clips
    project.placements = []
    project.save()
    progress("Voice clips done", 1.0)


# ---------------------------------------------------------------- step 6: assemble
def assemble_audio(project: Project, svc: Services, progress: Progress = _noop) -> Path:
    s = svc.settings
    progress("Placing clips on the timeline", 0.3)
    out = project.output_dir / f"narration.{s.output_format}"
    project.placements = assemble(
        project.sections, project.clips, out, project.duration or None,
        sample_rate=s.sample_rate, normalize_dbfs=s.normalize_dbfs, on_overflow=s.on_overflow,
    )
    export_srt(project.sections, project.clips, project.output_dir / "narration.srt")
    project.output = str(out)
    project.save()
    progress("Audio exported", 1.0)
    return out


# ---------------------------------------------------------------- step 7: preview
def make_preview(project: Project, progress: Progress = _noop) -> Path:
    if not (project.video and project.output):
        raise StvError("assemble the audio first")
    progress("Building preview video", 0.5)
    out = mux_preview(project.video, project.output, project.output_dir / "preview.mp4")
    progress("Preview ready", 1.0)
    return out


# ---------------------------------------------------------------- everything
def run_all(project: Project, svc: Services, personas: dict[str, Persona] | None = None,
            progress: Progress = _noop, *, skip_detect: bool = False, verify: bool = True) -> Path:
    if not skip_detect:
        detect_sections(project, svc, lambda m, f: progress(m, f * 0.2))
    write_script(project, svc, personas, lambda m, f: progress(m, 0.2 + f * 0.1))
    if verify:
        verify_script(project, svc, lambda m, f: progress(m, 0.3 + f * 0.05))
    generate_clips(project, svc, personas, lambda m, f: progress(m, 0.35 + f * 0.55))
    return assemble_audio(project, svc, lambda m, f: progress(m, 0.9 + f * 0.1))
