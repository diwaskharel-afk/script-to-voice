"""Streamlit front end. Run with:  streamlit run streamlit_app.py"""
from __future__ import annotations

import zlib
from pathlib import Path

import streamlit as st

from script_to_voice import pipeline as pl
from script_to_voice.assemble import timeline_rows
from script_to_voice.config import Settings
from script_to_voice.docs import read_documents
from script_to_voice.llm import load_personas
from script_to_voice.media import video_duration
from script_to_voice.models import StvError, count_words
from script_to_voice.project import Project
from script_to_voice.pronunciation import Glossary
from script_to_voice.sections import load_sections, normalize_sections
from script_to_voice.tts import ElevenLabsTTS

st.set_page_config(page_title="Script to Voice", page_icon="🎙️", layout="wide")

ELEVEN_MODELS = ["eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_v3"]
STATUS_ICON = {"ok": "✅", "sped_up": "⏩", "shortened": "✂️", "overflow": "⚠️", "error": "❌",
               "skipped": "➖", "pending": "⏳"}

# --------------------------------------------------------------------------- sidebar
env = Settings.from_env()
with st.sidebar:
    st.title("🎙️ Script to Voice")
    st.subheader("Keys")
    gem_key = st.text_input("Gemini API key", env.gemini_api_key, type="password",
                            help="Watches the video and writes the script.")
    el_key = st.text_input("ElevenLabs API key", env.elevenlabs_api_key, type="password",
                           help="Turns each line into a voice clip.")
    mock = st.checkbox("Mock mode (fake AI and voices, spends nothing)", value=not (gem_key and el_key))

    st.subheader("Voice")
    model = st.selectbox("ElevenLabs model", ELEVEN_MODELS,
                         index=ELEVEN_MODELS.index(env.elevenlabs_model) if env.elevenlabs_model in ELEVEN_MODELS else 0,
                         help="v3 is more expressive but takes no timed pauses or neighbour context.")
    voice_id = st.text_input("Voice ID", env.elevenlabs_voice_id)
    if el_key and st.button("Load my voices"):
        try:
            tmp = Settings(elevenlabs_api_key=el_key)
            st.session_state["voices"] = ElevenLabsTTS(tmp).list_voices()
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    if st.session_state.get("voices"):
        names = {f"{n} ({v[:6]}…)": v for v, n in st.session_state["voices"]}
        pick = st.selectbox("Pick a voice", list(names))
        voice_id = names[pick]
    stability = st.slider("Stability", 0.0, 1.0, 0.5, help="Lower = more expressive, higher = more even.")
    style = st.slider("Style", 0.0, 1.0, 0.0)

    st.subheader("Fit check")
    wps = st.number_input("Words per second", 1.0, 5.0, 2.5, 0.1)
    max_speed = st.slider("Max speed-up", 1.0, 1.3, 1.15, 0.01, help="Above ~1.15 narration starts to sound rushed.")
    max_shorten = st.number_input("Max rewrite attempts", 0, 5, 2)
    pad = st.number_input("Silence kept at end of each slot (s)", 0.0, 2.0, 0.25, 0.05)
    out_fmt = st.selectbox("Output format", ["wav", "mp3"])

    st.subheader("Pronunciation")
    gloss_text = st.text_area("One per line: term = spoken form", "EMEA = ee-mee-uh\nCSV = C S V", height=90)


def build_settings() -> Settings:
    s = Settings(
        gemini_api_key=gem_key, elevenlabs_api_key=el_key, elevenlabs_model=model,
        elevenlabs_voice_id=voice_id, stability=stability, style=style, words_per_second=wps,
        max_speedup=max_speed, max_shorten_attempts=int(max_shorten), tail_padding=pad,
        output_format=out_fmt, gemini_model=env.gemini_model,
    )
    s.llm_provider = "gemini" if (gem_key and not mock) else "mock"
    s.tts_provider = "elevenlabs" if (el_key and not mock) else "mock"
    return s


def services() -> pl.Services:
    return pl.Services.build(build_settings(), Glossary.from_text(gloss_text))


PERSONAS = load_personas(Path("personas"))


def run_step(label: str, fn, *args, **kw):
    """Run a pipeline step with a progress bar; show errors instead of a stack trace."""
    bar = st.progress(0.0, text=label)
    try:
        result = fn(*args, progress=lambda msg, f: bar.progress(min(1.0, f), text=msg), **kw)
        bar.empty()
        return result if result is not None else True
    except StvError as e:
        bar.empty()
        st.error(str(e))
    except Exception as e:  # noqa: BLE001
        bar.empty()
        st.error(f"Unexpected error: {e}")
    return None


# --------------------------------------------------------------------------- project
st.header("Narration from script to voice")
if mock:
    st.info("Mock mode: sections, script and voices are fake so you can try the workflow for free. "
            "Add both API keys in the sidebar for the real thing.")

name = st.text_input("Project name", st.session_state.get("name", "my-training-video"))
st.session_state["name"] = name
settings0 = build_settings()
settings0.projects_dir = Path("projects")
project: Project = Project.create(settings0.projects_dir, name)

tab_in, tab_sec, tab_script, tab_voice, tab_out = st.tabs(
    ["1 · Input", "2 · Sections", "3 · Script", "4 · Voice & fit", "5 · Result"])

# --------------------------------------------------------------------------- 1 input
with tab_in:
    c1, c2 = st.columns(2)
    video_up = c1.file_uploader("Video", type=["mp4", "mov", "webm", "mkv", "avi"])
    docs_up = c2.file_uploader("Documentation (PDF, Markdown, text)", type=["pdf", "md", "txt"],
                               accept_multiple_files=True)
    notes = st.text_area("Notes on tone and audience", project.notes,
                         placeholder="e.g. New hires, relaxed tone, don't read out URLs.")
    persona = st.selectbox("Persona", list(PERSONAS),
                           index=list(PERSONAS).index(project.persona) if project.persona in PERSONAS else 0)
    sections_up = st.file_uploader("…or skip video analysis with hand-written slots (JSON)", type=["json"])

    if st.button("Save input", type="primary"):
        inputs = project.root / "inputs"
        inputs.mkdir(exist_ok=True)
        try:
            if video_up:
                vp = inputs / video_up.name
                vp.write_bytes(video_up.getbuffer())
                project.video = str(vp)
                project.duration = video_duration(vp)
            paths = []
            for d in docs_up or []:
                dp = inputs / d.name
                dp.write_bytes(d.getbuffer())
                paths.append(dp)
            if paths:
                project.document = read_documents(paths)
            if sections_up:
                sp = inputs / "sections.json"
                sp.write_bytes(sections_up.getbuffer())
                if not project.duration:
                    project.duration = st.session_state.get("manual_duration", 0.0) or 0.0
                project.sections = load_sections(sp, project.duration or None)
            project.notes, project.persona = notes, persona
            project.save()
            st.success("Saved.")
        except StvError as e:
            st.error(str(e))

    if project.video:
        st.caption(f"Video: {Path(project.video).name} · {project.duration:.1f}s")
        st.video(project.video)
    if project.document:
        st.caption(f"Documentation loaded: {len(project.document):,} characters")
    if not project.video:
        project.duration = st.number_input("Video length in seconds (needed only without a video)",
                                           0.0, 36000.0, project.duration)
        st.session_state["manual_duration"] = project.duration

    if project.video and st.button("Find sections in the video"):
        if run_step("Analysing video", pl.detect_sections, project, services()):
            st.rerun()

# --------------------------------------------------------------------------- 2 sections
with tab_sec:
    st.write("Each row is one narration line. **Slot** is the seconds that line has. Edit freely.")
    if not project.sections:
        st.info("No sections yet. Find them from the video, or upload a slots JSON in step 1.")
    rows = [{"start (s)": s.start, "end (s)": s.end, "title": s.title, "on screen": s.description}
            for s in project.sections]
    edited = st.data_editor(rows, num_rows="dynamic", key="sec_editor",
                            column_config={"start (s)": st.column_config.NumberColumn(min_value=0.0, format="%.1f"),
                                           "end (s)": st.column_config.NumberColumn(min_value=0.0, format="%.1f")})
    if st.button("Save sections"):
        old = {round(s.start, 2): s.script for s in project.sections}
        raw = [{"start": r.get("start (s)"), "end": r.get("end (s)"), "title": r.get("title") or "",
                "description": r.get("on screen") or "", "script": old.get(round(float(r.get("start (s)") or 0), 2), "")}
               for r in edited if r.get("start (s)") is not None]
        project.sections = normalize_sections(raw, project.duration or None, 0.5)
        project.clips, project.placements = [], []
        project.save()
        st.success(f"{len(project.sections)} sections saved.")
        st.rerun()

# --------------------------------------------------------------------------- 3 script
with tab_script:
    if not project.sections:
        st.info("Add sections first.")
    else:
        c1, c2 = st.columns(2)
        if c1.button("Write script with AI", type="primary"):
            project.persona = persona
            if run_step("Writing script", pl.write_script, project, services(), PERSONAS):
                st.rerun()
        if c2.button("Check script against documentation"):
            run_step("Checking", pl.verify_script, project, services())
        if project.issues:
            st.warning("Possible problems (check these against the documentation):")
            for i in project.issues:
                st.write(f"• **Section {i.section_index}:** {i.problem}")

        st.caption("Edit any line. Only changed lines are regenerated (unchanged ones come from the cache).")
        s0 = build_settings()
        changed = False
        for sec in project.sections:
            budget = sec.word_budget(s0.words_per_second, s0.tail_padding)
            # the key includes a checksum of the stored text so the box refreshes when the AI rewrites it
            key = f"script_{name}_{sec.index}_{sec.start}_{zlib.crc32(sec.script.encode())}"
            val = st.text_area(f"{sec.index}. {sec.title or 'Section'} · {sec.start:.1f}s · {sec.slot:.1f}s available",
                               sec.script, key=key, height=80)
            words = count_words(val)
            (st.caption if words <= budget * 1.2 else st.warning)(f"{words} words (about {budget} fit in the slot)")
            if val != sec.script:
                sec.script, changed = val, True
        if changed:
            project.save()

# --------------------------------------------------------------------------- 4 voice
with tab_voice:
    if not any(s.script.strip() for s in project.sections):
        st.info("Write the script first.")
    else:
        if st.button("Generate voice and fit to slots", type="primary"):
            svc = services()
            if run_step("Generating", pl.generate_clips, project, svc, PERSONAS):
                st.caption(f"Cache: {svc.cache.hits} clips reused, {svc.cache.misses} generated.")
        if project.clips:
            if project.flagged:
                st.error(f"{len(project.flagged)} line(s) could not be made to fit. Edit the script or widen the slot.")
            for sec, row in zip(project.sections, timeline_rows(project.sections, project.clips)):
                clip = project.clip_for(sec.index)
                with st.expander(f"{STATUS_ICON.get(row['status'], '')} {sec.index}. {sec.title}  ·  "
                                 f"{row['clip (s)']}s of {row['slot (s)']}s  ·  {row['status']}"):
                    st.write(clip.final_text if clip else "")
                    st.caption(row["note"])
                    if clip and clip.path and Path(clip.path).exists():
                        st.audio(clip.path)

# --------------------------------------------------------------------------- 5 result
with tab_out:
    if not project.clips:
        st.info("Generate the voice clips first.")
    else:
        if st.button("Assemble final audio", type="primary"):
            if run_step("Assembling", pl.assemble_audio, project, services()):
                st.rerun()
        if project.output and Path(project.output).exists():
            st.success("Narration ready: one audio file, as long as the video, with the gaps built in.")
            st.audio(project.output)
            d1, d2 = st.columns(2)
            d1.download_button("Download audio", Path(project.output).read_bytes(), Path(project.output).name)
            srt = project.output_dir / "narration.srt"
            if srt.exists():
                d2.download_button("Download subtitles (.srt)", srt.read_bytes(), "narration.srt")
            st.dataframe(timeline_rows(project.sections, project.clips))
            if project.video:
                if st.button("Build preview (audio over video)"):
                    p = run_step("Building preview", pl.make_preview, project)
                    if p and p is not True:
                        st.session_state["preview"] = str(p)
                if st.session_state.get("preview") and Path(st.session_state["preview"]).exists():
                    st.video(st.session_state["preview"])
