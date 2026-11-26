"""Glossary, cache, project persistence, JSON extraction."""
import json

import pytest

from script_to_voice.llm import GeminiLLM, extract_json
from script_to_voice.models import Clip, Issue, Section, StvError, count_words
from script_to_voice.project import Project
from script_to_voice.pronunciation import Glossary


def test_glossary_replaces_whole_words_case_insensitively():
    g = Glossary({"SQL": "sequel", "SQL Server": "sequel server", "EMEA": "ee-mee-uh"})
    assert g.apply("Use SQL Server, not mysql. Pick emea.") == "Use sequel server, not mysql. Pick ee-mee-uh."
    assert g.apply("nothing here") == "nothing here"


def test_glossary_load_text_and_json(tmp_path):
    t = tmp_path / "g.txt"
    t.write_text("# c\nCSV = C S V\n", encoding="utf-8")
    j = tmp_path / "g.json"
    j.write_text(json.dumps({"API": "A P I"}), encoding="utf-8")
    assert Glossary.load(t).apply("a CSV file") == "a C S V file"
    assert Glossary.load(j).apply("the API") == "the A P I"
    assert Glossary.load(None).apply("x") == "x"


def test_cache_roundtrip_and_key_stability(cache):
    k = cache.key(text="hi", voice="v")
    assert k == cache.key(voice="v", text="hi")
    assert k != cache.key(text="hi!", voice="v")
    assert cache.get(k) is None
    cache.put(k, b"abc", "mp3")
    assert cache.get(k) == (b"abc", "mp3")
    assert (cache.hits, cache.misses) == (1, 1)


def test_count_words_ignores_tags():
    assert count_words('Hello <break time="1s" /> there [pause] friend') == 3


def test_project_roundtrip(tmp_path):
    p = Project.create(tmp_path, "My Video!")
    p.video, p.duration, p.document = "v.mp4", 12.5, "doc text"
    p.sections = [Section(1, 0, 5, "t", "d", "line")]
    p.clips = [Clip(1, "c.wav", 3.0, 5.0, "ok")]
    p.issues = [Issue(1, "unsupported")]
    p.save()
    q = Project.load(p.root)
    assert q.document == "doc text" and q.duration == 12.5
    assert q.sections[0].script == "line" and q.clips[0].status == "ok" and q.issues[0].problem == "unsupported"
    assert Project.create(tmp_path, "My Video!").sections  # re-creating resumes


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert extract_json('Sure! Here you go: [{"a": 1}] hope that helps') == [{"a": 1}]
    with pytest.raises(json.JSONDecodeError):
        extract_json("no json at all")


def _fake_gemini(reply: str) -> GeminiLLM:
    llm = object.__new__(GeminiLLM)
    llm._generate = lambda *a, **k: reply
    return llm


def test_gemini_write_script_parses_reply(ctx):
    llm = _fake_gemini('[{"index": 1, "text": " Hello. "}, {"index": "2", "text": ""}, {"bad": 1}]')
    out = llm.write_script([Section(1, 0, 5), Section(2, 5, 9)], ctx)
    assert out == {1: "Hello.", 2: ""}


def test_gemini_verify_parses_reply():
    llm = _fake_gemini('[{"index": 2, "problem": "invented menu"}, {"oops": 1}]')
    issues = llm.verify([Section(2, 0, 5, script="x")], "doc")
    assert [(i.section_index, i.problem) for i in issues] == [(2, "invented menu")]


class _FakeFiles:
    def __init__(self):
        self.deleted = []

    def upload(self, file):
        return type("F", (), {"name": f"files/{file}", "state": type("S", (), {"name": "ACTIVE"})()})()

    def delete(self, name):
        self.deleted.append(name)


def _gemini_with_replies(monkeypatch, replies, total, chunk_seconds):
    from script_to_voice.config import Settings

    llm = object.__new__(GeminiLLM)
    llm.s = Settings(chunk_seconds=chunk_seconds)
    llm.client = type("C", (), {"files": _FakeFiles()})()
    it = iter(replies)
    llm._generate_json = lambda *a, **k: next(it)
    monkeypatch.setattr("script_to_voice.llm.video_duration", lambda p: total)
    return llm


def test_gemini_detect_sections_single_video(monkeypatch, tmp_path):
    llm = _gemini_with_replies(
        monkeypatch,
        [[{"start": "0:00", "end": "0:10", "title": "a"}, {"start": "0:10", "end": "0:25", "title": "b"}]],
        total=25.0, chunk_seconds=600,
    )
    out = llm.detect_sections(tmp_path / "v.mp4", "", "")
    assert [(s.start, s.end, s.title) for s in out] == [(0.0, 10.0, "a"), (10.0, 25.0, "b")]
    assert len(llm.client.files.deleted) == 1  # the upload is removed afterwards


def test_gemini_detect_sections_long_video_adds_chunk_offsets(monkeypatch, tmp_path):
    llm = _gemini_with_replies(
        monkeypatch,
        [[{"start": "0:00", "end": "0:30", "title": "first"}],
         [{"start": "0:05", "end": "0:30", "title": "second"}]],  # relative to chunk 2
        total=70.0, chunk_seconds=40,
    )
    monkeypatch.setattr("script_to_voice.llm.split_video",
                        lambda path, secs, d: [(tmp_path / "c0.mp4", 0.0), (tmp_path / "c1.mp4", 40.0)])
    out = llm.detect_sections(tmp_path / "v.mp4", "", "")
    assert [(s.start, s.end, s.title) for s in out] == [(0.0, 30.0, "first"), (45.0, 70.0, "second")]
    assert len(llm.client.files.deleted) == 2


def test_gemini_detect_sections_rejects_bad_shape(monkeypatch, tmp_path):
    llm = _gemini_with_replies(monkeypatch, [{"oops": 1}], total=20.0, chunk_seconds=600)
    with pytest.raises(StvError, match="unexpected shape"):
        llm.detect_sections(tmp_path / "v.mp4", "", "")
