"""The fit check: ok / sped_up / shortened / overflow / error paths."""
import pytest

from script_to_voice.models import ERROR, OK, OVERFLOW, SHORTENED, SKIPPED, SPED_UP, Section, StvError
from script_to_voice.llm import MockLLM
from script_to_voice.pronunciation import Glossary
from script_to_voice.tts import MockTTS

WORDS = lambda n: " ".join(f"w{i}" for i in range(n))  # noqa: E731


def section(words, slot):
    return Section(1, 0.0, slot, script=WORDS(words))


def test_fits_untouched(make_engine, tmp_path):
    clip = make_engine().fit(section(10, 8), "", "", tmp_path / "c.wav")  # 4.0s in 8s
    assert clip.status == OK and clip.speed == 1.0 and clip.attempts == 1
    assert clip.duration == pytest.approx(4.0, abs=0.05)


def test_slightly_long_is_sped_up(make_engine, tmp_path):
    # 20 words = 8.0s; slot 8.0 - 0.25 padding = 7.75 target -> ratio 1.03
    clip = make_engine().fit(section(20, 8), "", "", tmp_path / "c.wav")
    assert clip.status == SPED_UP and 1.0 < clip.speed <= 1.15
    assert clip.duration <= 8.0 - 0.25 + 0.1
    assert clip.attempts == 1  # speeding up costs no extra credits


def test_way_too_long_is_shortened_then_fits(make_engine, tmp_path):
    # 20 words = 8s in a 3s slot: needs the LLM to cut words
    clip = make_engine().fit(section(20, 3), "", "", tmp_path / "c.wav")
    assert clip.status == SHORTENED
    assert clip.duration <= 3.0
    assert len(clip.final_text.split()) < 20 and clip.original_text != clip.final_text
    assert clip.attempts >= 2


def test_overflow_flagged_when_llm_cannot_help(make_engine, tmp_path):
    class Stubborn(MockLLM):
        def shorten(self, text, max_words, ctx):
            return text  # does not get shorter

    clip = make_engine(llm=Stubborn()).fit(section(20, 3), "", "", tmp_path / "c.wav")
    assert clip.status == OVERFLOW and clip.needs_human
    assert clip.speed == pytest.approx(1.15)
    assert clip.duration > 3.0  # honest about not fitting


def test_overflow_when_attempts_run_out(make_engine, settings, tmp_path):
    settings.max_shorten_attempts = 1

    class Barely(MockLLM):
        def shorten(self, text, max_words, ctx):
            return " ".join(text.split()[:-1])  # one word at a time

    clip = make_engine(llm=Barely()).fit(section(30, 3), "", "", tmp_path / "c.wav")
    assert clip.status == OVERFLOW


def test_empty_script_is_skipped(make_engine, tmp_path):
    clip = make_engine().fit(Section(1, 0, 5, script="  "), "", "", tmp_path / "c.wav")
    assert clip.status == SKIPPED and not clip.path


def test_llm_failure_flags_instead_of_crashing(make_engine, tmp_path):
    class Broken(MockLLM):
        def shorten(self, text, max_words, ctx):
            raise StvError("quota exceeded")

    clip = make_engine(llm=Broken()).fit(section(20, 3), "", "", tmp_path / "c.wav")
    assert clip.status == OVERFLOW  # best-effort clip, still produced


def test_tts_failure_becomes_error_clip(make_engine, tmp_path):
    class Down:
        def synthesize(self, *a, **k):
            from script_to_voice.tts import TTSError
            raise TTSError("HTTP 401")

    clip = make_engine(tts=Down()).fit(section(5, 5), "", "", tmp_path / "c.wav")
    assert clip.status == ERROR and "401" in clip.message and clip.needs_human


def test_cache_prevents_second_generation(make_engine, tmp_path):
    tts = MockTTS(0.4)
    eng = make_engine(tts=tts)
    s = section(10, 8)
    eng.fit(s, "", "", tmp_path / "a.wav")
    again = eng.fit(s, "different neighbour", "", tmp_path / "b.wav")
    assert tts.calls == 1 and again.attempts == 0  # neighbours do not invalidate the cache


def test_glossary_applies_to_spoken_text_only(make_engine, tmp_path):
    seen = []

    class Spy(MockTTS):
        def synthesize(self, text, **kw):
            seen.append(text)
            return super().synthesize(text, **kw)

    s = Section(1, 0, 8, script="Export as CSV")
    clip = make_engine(tts=Spy(), glossary=Glossary({"CSV": "C S V"})).fit(s, "", "", tmp_path / "c.wav")
    assert seen == ["Export as C S V"] and clip.final_text == "Export as CSV"
