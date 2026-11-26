"""Headless smoke test of the Streamlit app using Streamlit's AppTest."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from script_to_voice.project import Project  # noqa: E402
from script_to_voice.sections import load_sections  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_app_walkthrough_in_mock_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the app writes to ./projects
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "")

    p = Project.create(tmp_path / "projects", "t")
    p.duration = 60.0
    p.sections = load_sections(ROOT / "examples" / "sample_sections.json", 60.0)
    p.save()

    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    name_box = next(t for t in at.text_input if t.label == "Project name")
    name_box.set_value("t").run()
    assert not at.exception

    button(at, "Write script with AI").click().run()
    assert not at.exception
    assert Project.load(tmp_path / "projects" / "t").sections[0].script  # script written and shown

    button(at, "Generate voice and fit to slots").click().run()
    assert not at.exception
    assert len(Project.load(tmp_path / "projects" / "t").clips) == 5

    button(at, "Assemble final audio").click().run()
    assert not at.exception
    assert Path(Project.load(tmp_path / "projects" / "t").output).exists()
