"""Settings, loaded from environment variables (and a .env file if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# Models whose text-to-speech endpoint does NOT accept previous_text/next_text
# and does NOT understand <break> tags (it uses audio tags like [pause] instead).
V3_MODEL_PREFIXES = ("eleven_v3",)


@dataclass
class Settings:
    # providers: "gemini"/"mock" and "elevenlabs"/"mock"
    llm_provider: str = "mock"
    tts_provider: str = "mock"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    elevenlabs_api_key: str = ""
    elevenlabs_model: str = "eleven_multilingual_v2"
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    pronunciation_dictionary_ids: list[str] = field(default_factory=list)

    # voice feel
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0

    # fit-check behaviour
    words_per_second: float = 2.5
    max_speedup: float = 1.15
    max_shorten_attempts: int = 2
    tail_padding: float = 0.25  # seconds kept free at the end of every slot
    on_overflow: str = "trim"  # "trim" or "overlap"

    # assembly
    sample_rate: int = 44100
    normalize_dbfs: float | None = -20.0
    output_format: str = "wav"  # "wav" or "mp3"

    # video analysis
    chunk_seconds: float = 600.0  # split videos longer than this before sending to Gemini
    min_section_seconds: float = 1.5

    # storage
    projects_dir: Path = Path("projects")
    cache_dir: Path = Path(".cache/clips")

    @classmethod
    def from_env(cls) -> "Settings":
        g = os.environ.get
        s = cls(
            gemini_api_key=g("GEMINI_API_KEY", ""),
            elevenlabs_api_key=g("ELEVENLABS_API_KEY", ""),
            gemini_model=g("GEMINI_MODEL", cls.gemini_model),
            elevenlabs_model=g("ELEVENLABS_MODEL", cls.elevenlabs_model),
            elevenlabs_voice_id=g("ELEVENLABS_VOICE_ID", cls.elevenlabs_voice_id),
        )
        # Use real providers automatically when a key is present.
        if s.gemini_api_key:
            s.llm_provider = "gemini"
        if s.elevenlabs_api_key:
            s.tts_provider = "elevenlabs"
        return s

    @property
    def supports_context(self) -> bool:
        """Whether the TTS model accepts previous_text / next_text."""
        return not self.elevenlabs_model.startswith(V3_MODEL_PREFIXES)

    @property
    def supports_break_tags(self) -> bool:
        return not self.elevenlabs_model.startswith(V3_MODEL_PREFIXES)

    def public_dict(self) -> dict:
        """Settings without secrets, safe to store in a project file."""
        d = asdict(self)
        d.pop("gemini_api_key")
        d.pop("elevenlabs_api_key")
        d["projects_dir"] = str(self.projects_dir)
        d["cache_dir"] = str(self.cache_dir)
        return d
