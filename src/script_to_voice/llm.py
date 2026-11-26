"""Video analysis + script writing. Gemini for real runs, a deterministic mock for tests/demos."""
from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import Settings
from .media import split_video, video_duration
from .models import Issue, Section, StvError, parse_timecode
from .sections import normalize_sections


# --------------------------------------------------------------------------- personas
@dataclass
class Persona:
    name: str
    instructions: str


BUILTIN_PERSONAS = {
    "Friendly trainer": Persona(
        "Friendly trainer",
        "You are a warm, encouraging trainer. Speak directly to the learner ('you'). "
        "Plain language, short sentences, one idea per line. Name the exact button or menu shown on screen.",
    ),
    "Concise technical": Persona(
        "Concise technical",
        "You are a precise technical writer. Neutral tone, imperative mood ('Select...', 'Enter...'). "
        "No filler, no enthusiasm, no marketing language.",
    ),
    "Executive overview": Persona(
        "Executive overview",
        "You explain what is happening and why it matters, in confident, brisk business language. "
        "Skip click-by-click detail unless it is essential.",
    ),
}


def load_personas(directory: str | Path | None) -> dict[str, Persona]:
    """Built-in personas plus any `*.md` file in `directory` (file name = persona name)."""
    personas = dict(BUILTIN_PERSONAS)
    if directory and Path(directory).is_dir():
        for f in sorted(Path(directory).glob("*.md")):
            personas[f.stem.replace("_", " ")] = Persona(
                f.stem.replace("_", " "), f.read_text(encoding="utf-8").strip()
            )
    return personas


@dataclass
class ScriptContext:
    persona: Persona
    document: str = ""
    notes: str = ""
    words_per_second: float = 2.5
    tail_padding: float = 0.25


class LLM(Protocol):
    def detect_sections(self, video: Path, document: str, notes: str) -> list[Section]: ...
    def write_script(self, sections: list[Section], ctx: ScriptContext) -> dict[int, str]: ...
    def shorten(self, text: str, max_words: int, ctx: ScriptContext) -> str: ...
    def verify(self, sections: list[Section], document: str) -> list[Issue]: ...


# --------------------------------------------------------------------------- prompts
SECTION_PROMPT = """You are analysing a screen-recorded technical training video so a narrator can be added.
List the distinct moments of the video, in order. Each moment becomes one narration line, so
split where the on-screen action changes (a new screen, dialog, or task step), not at every click.
Aim for moments of roughly 4-20 seconds. Cover the whole video.

Return ONLY a JSON array. Each item: {{"start": "MM:SS", "end": "MM:SS", "title": "short name", "description": "what is on screen and what the user does"}}.
Timestamps are relative to the start of this video file.

Context from the documentation (may help you name things correctly):
{document}

Notes from the author: {notes}
"""

SCRIPT_PROMPT = """{persona}

Write the spoken narration for a training video, one line per section below.
Rules:
- Each line is read aloud, so write plain spoken sentences. No markdown, no bullets, no stage directions, no emoji.
- HARD LIMIT: never exceed the word budget of a section. Under budget is fine. If a section needs no narration, return an empty string.
- Use ONLY facts from the documentation or visible in the section description. Do not invent steps, menu names, or values.
- Write numbers and symbols the way they are spoken.
- Keep tone and wording consistent from one section to the next, as one narrator speaking continuously. Do not re-introduce yourself.

Author notes: {notes}

DOCUMENTATION:
{document}

SECTIONS:
{sections}

Return ONLY a JSON array: [{{"index": <int>, "text": "<narration>"}}, ...] with one item per section.
"""

SHORTEN_PROMPT = """{persona}

Rewrite this narration line so it keeps the same meaning and tone but has AT MOST {max_words} words.
Drop the least important detail first. Return ONLY the rewritten line, no quotes.

Line: {text}
"""

VERIFY_PROMPT = """Check a narration script against its documentation. For each section, flag any claim, step,
menu name or value in the narration that is NOT supported by the documentation or the section description.
Return ONLY a JSON array: [{{"index": <int>, "problem": "<what is unsupported>"}}]. Return [] if all is supported.

DOCUMENTATION:
{document}

SCRIPT:
{script}
"""


def _fmt_sections(sections: list[Section], ctx: ScriptContext) -> str:
    return "\n".join(
        f"- index {s.index} | {s.slot:.1f}s available | word budget {s.word_budget(ctx.words_per_second, ctx.tail_padding)}"
        f" | title: {s.title} | on screen: {s.description}"
        for s in sections
    )


def extract_json(text: str):
    """Parse JSON from a model reply, tolerating ``` fences and surrounding prose."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*\]|\{.*\})", t, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise


# --------------------------------------------------------------------------- Gemini
class GeminiLLM:
    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise StvError("GEMINI_API_KEY is not set")
        from google import genai

        self.s = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)

    # -- low level
    def _generate(self, contents, *, json_mode: bool, temperature: float = 0.3) -> str:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        last: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(
                    model=self.s.gemini_model, contents=contents, config=cfg
                )
                if not resp.text:
                    raise StvError("Gemini returned an empty response")
                return resp.text
            except Exception as e:  # noqa: BLE001 - SDK raises many types; retry transient ones
                last = e
                time.sleep(2 ** attempt)
        raise StvError(f"Gemini request failed: {last}") from last

    def _generate_json(self, contents, **kw):
        text = self._generate(contents, json_mode=True, **kw)
        try:
            return extract_json(text)
        except json.JSONDecodeError:
            text = self._generate(contents, json_mode=True, temperature=0.0)
            try:
                return extract_json(text)
            except json.JSONDecodeError as e:
                raise StvError(f"Gemini did not return valid JSON: {text[:200]!r}") from e

    def _upload(self, path: Path):
        f = self.client.files.upload(file=str(path))
        deadline = time.time() + 600
        while f.state and f.state.name == "PROCESSING":
            if time.time() > deadline:
                raise StvError("Gemini took too long to process the video")
            time.sleep(3)
            f = self.client.files.get(name=f.name)
        if f.state and f.state.name == "FAILED":
            raise StvError("Gemini could not process the video file")
        return f

    # -- tasks
    def detect_sections(self, video: Path, document: str, notes: str) -> list[Section]:
        total = video_duration(video)
        prompt = SECTION_PROMPT.format(document=document[:20_000] or "(none)", notes=notes or "(none)")
        raw: list[dict] = []
        # Timestamps drift on long videos, so analyse pieces and add the offsets back.
        with tempfile.TemporaryDirectory(prefix="stv_chunks_") as tmp:
            if total <= self.s.chunk_seconds:
                chunks = [(video, 0.0)]
            else:
                chunks = split_video(video, self.s.chunk_seconds, tmp)
            for path, offset in chunks:
                f = self._upload(Path(path))
                try:
                    items = self._generate_json([f, prompt], temperature=0.2)
                finally:
                    try:  # don't leave the user's video sitting on Google's servers
                        self.client.files.delete(name=f.name)
                    except Exception:  # noqa: BLE001 - best effort; files expire on their own
                        pass
                if not isinstance(items, list):
                    raise StvError("Gemini returned timestamps in an unexpected shape")
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    for key in ("start", "end"):
                        if it.get(key) not in (None, ""):
                            try:
                                it[key] = parse_timecode(it[key]) + offset
                            except ValueError:
                                it[key] = None
                    raw.append(it)
        sections = normalize_sections(raw, total, self.s.min_section_seconds)
        if not sections:
            raise StvError("No usable sections were detected in the video")
        return sections

    def write_script(self, sections: list[Section], ctx: ScriptContext) -> dict[int, str]:
        prompt = SCRIPT_PROMPT.format(
            persona=ctx.persona.instructions, notes=ctx.notes or "(none)",
            document=ctx.document or "(none)", sections=_fmt_sections(sections, ctx),
        )
        items = self._generate_json(prompt)
        if not isinstance(items, list):
            raise StvError("Gemini returned the script in an unexpected shape")
        out: dict[int, str] = {}
        for it in items:
            try:
                out[int(it["index"])] = str(it.get("text", "")).strip()
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def shorten(self, text: str, max_words: int, ctx: ScriptContext) -> str:
        out = self._generate(
            SHORTEN_PROMPT.format(persona=ctx.persona.instructions, max_words=max_words, text=text),
            json_mode=False, temperature=0.2,
        )
        return out.strip().strip('"').strip()

    def verify(self, sections: list[Section], document: str) -> list[Issue]:
        script = "\n".join(f"[{s.index}] (on screen: {s.description}) {s.script}" for s in sections if s.script)
        items = self._generate_json(VERIFY_PROMPT.format(document=document or "(none)", script=script), temperature=0.0)
        issues = []
        for it in items if isinstance(items, list) else []:
            try:
                issues.append(Issue(int(it["index"]), str(it["problem"])))
            except (KeyError, TypeError, ValueError):
                continue
        return issues


# --------------------------------------------------------------------------- Mock
class MockLLM:
    """Deterministic stand-in: no network, no credits. Good for demos and tests."""

    def __init__(self, settings: Settings | None = None, duration: float | None = None,
                 verbosity: tuple[float, ...] = (0.9,)):
        """`verbosity` cycles per section: 0.9 fits, 1.1 needs a speed-up, 1.5 needs shortening."""
        self.s = settings or Settings()
        self.duration = duration
        self.verbosity = verbosity

    def detect_sections(self, video: Path, document: str, notes: str) -> list[Section]:
        total = self.duration or video_duration(video)
        n = max(1, round(total / 8))
        step = total / n
        raw = [{"start": i * step, "end": (i + 1) * step, "title": f"Step {i + 1}",
                "description": f"Something happens on screen in step {i + 1}"} for i in range(n)]
        return normalize_sections(raw, total, self.s.min_section_seconds)

    def write_script(self, sections: list[Section], ctx: ScriptContext) -> dict[int, str]:
        out = {}
        for s in sections:
            budget = s.word_budget(ctx.words_per_second, ctx.tail_padding)
            base = f"In this step, {s.title or 'we continue'}. {s.description}".strip()
            words = base.split()
            v = self.verbosity[(s.index - 1) % len(self.verbosity)]
            n = max(1, int(budget * v))
            out[s.index] = " ".join(words[i % len(words)] for i in range(n))
        return out

    def shorten(self, text: str, max_words: int, ctx: ScriptContext) -> str:
        return " ".join(text.split()[: max(1, max_words)])

    def verify(self, sections: list[Section], document: str) -> list[Issue]:
        return []
