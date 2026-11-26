import pytest

from script_to_voice.cache import ClipCache
from script_to_voice.config import Settings
from script_to_voice.llm import BUILTIN_PERSONAS, MockLLM, ScriptContext
from script_to_voice.pronunciation import Glossary
from script_to_voice.tts import MockTTS


@pytest.fixture
def settings(tmp_path):
    s = Settings()  # mock providers
    s.projects_dir = tmp_path / "projects"
    s.cache_dir = tmp_path / "cache"
    return s


@pytest.fixture
def ctx():
    return ScriptContext(BUILTIN_PERSONAS["Friendly trainer"])


@pytest.fixture
def cache(tmp_path):
    return ClipCache(tmp_path / "cache")


@pytest.fixture
def make_engine(settings, cache, ctx):
    from script_to_voice.fit import FitEngine

    def _make(tts=None, llm=None, glossary=None):
        return FitEngine(tts or MockTTS(0.4), llm or MockLLM(settings), cache, settings, ctx, glossary or Glossary())

    return _make
