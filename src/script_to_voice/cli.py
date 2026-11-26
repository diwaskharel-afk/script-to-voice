"""Command line: `stv demo`, `stv run`, `stv build`, `stv voices`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings
from .docs import read_documents
from .llm import MockLLM, load_personas
from .media import video_duration
from .models import StvError
from . import pipeline as pl
from .pronunciation import Glossary
from .project import Project
from .sections import load_sections
from .assemble import timeline_rows


def _progress(msg: str, frac: float) -> None:
    print(f"[{frac * 100:3.0f}%] {msg}", flush=True)


def _report(project: Project) -> None:
    print("\nTimeline")
    print(f"{'#':>3} {'start':>8} {'slot':>6} {'clip':>6}  {'status':<10} note")
    for r in timeline_rows(project.sections, project.clips):
        clip = f"{r['clip (s)']:.1f}" if r["clip (s)"] is not None else "-"
        print(f"{r['#']:>3} {r['start']:>8} {r['slot (s)']:>6} {clip:>6}  {r['status']:<10} {r['note']}")
    if project.issues:
        print("\nPossible script problems (check against the documentation):")
        for i in project.issues:
            print(f"  section {i.section_index}: {i.problem}")
    if project.flagged:
        print(f"\n{len(project.flagged)} section(s) need a human: "
              + ", ".join(str(c.section_index) for c in project.flagged))
    if project.output:
        print(f"\nAudio: {project.output}")


def _settings(args) -> Settings:
    s = Settings.from_env()
    if getattr(args, "mock", False):
        s.llm_provider = s.tts_provider = "mock"
    for name in ("voice", "model"):
        v = getattr(args, name, None)
        if v:
            setattr(s, "elevenlabs_voice_id" if name == "voice" else "elevenlabs_model", v)
    if getattr(args, "format", None):
        s.output_format = args.format
    if getattr(args, "projects_dir", None):
        s.projects_dir = Path(args.projects_dir)
    return s


def cmd_demo(args) -> int:
    s = Settings()  # mock everything: no keys, no credits
    s.projects_dir = Path(args.projects_dir or "projects")
    example = Path(__file__).resolve().parents[2] / "examples" / "sample_sections.json"
    project = Project.create(s.projects_dir, "demo")
    project.duration = 60.0
    project.sections = load_sections(example, project.duration)
    project.notes = "Demo run with fake voices."
    svc = pl.Services.build(s)
    # lines of varying length, so the demo shows the fit loop: fits / sped up / shortened
    svc.llm = MockLLM(s, verbosity=(0.9, 1.1, 1.6, 0.9, 1.1))
    pl.run_all(project, svc, progress=_progress, skip_detect=True)
    _report(project)
    return 0


def cmd_run(args) -> int:
    s = _settings(args)
    glossary = Glossary.load(args.glossary)
    svc = pl.Services.build(s, glossary)
    project = Project.create(s.projects_dir, args.name)
    if args.video:
        project.video = str(Path(args.video).resolve())
        project.duration = video_duration(project.video)
    elif args.duration:
        project.duration = args.duration
    project.notes = args.notes or ""
    project.persona = args.persona
    project.document = read_documents(args.docs) if args.docs else ""
    skip = False
    if args.sections:
        project.sections = load_sections(args.sections, project.duration or None)
        skip = True
    elif not project.video:
        raise StvError("give --video, or --sections plus --duration")
    if svc.is_mock:
        print("(mock mode: fake AI and/or voices are being used)")
    personas = load_personas(Path("personas"))
    pl.run_all(project, svc, personas, _progress, skip_detect=skip, verify=not args.no_verify)
    if args.preview and project.video:
        print("Preview:", pl.make_preview(project, _progress))
    _report(project)
    return 1 if project.flagged else 0


def cmd_build(args) -> int:
    """Re-voice and re-assemble an existing project after editing its script."""
    s = _settings(args)
    project = Project.load(args.project)
    svc = pl.Services.build(s, Glossary.load(args.glossary))
    pl.generate_clips(project, svc, progress=_progress)
    pl.assemble_audio(project, svc, _progress)
    _report(project)
    return 1 if project.flagged else 0


def cmd_voices(args) -> int:
    from .tts import ElevenLabsTTS

    for vid, name in ElevenLabsTTS(_settings(args)).list_voices():
        print(f"{vid}  {name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stv", description="Script-to-voice: timed narration for training videos")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the whole pipeline on fake data (no keys, no credits)")
    d.add_argument("--projects-dir")
    d.set_defaults(fn=cmd_demo)

    def common(sp):
        sp.add_argument("--mock", action="store_true", help="fake AI and voices")
        sp.add_argument("--voice", help="ElevenLabs voice id")
        sp.add_argument("--model", help="ElevenLabs model id")
        sp.add_argument("--glossary", help="pronunciation glossary (.json or term=spoken lines)")
        sp.add_argument("--format", choices=["wav", "mp3"])
        sp.add_argument("--projects-dir")

    r = sub.add_parser("run", help="full pipeline: video + docs -> narration audio")
    r.add_argument("--name", required=True)
    r.add_argument("--video")
    r.add_argument("--docs", nargs="*", default=[])
    r.add_argument("--notes", default="")
    r.add_argument("--persona", default="Friendly trainer")
    r.add_argument("--sections", help="JSON of hand-written slots (skips video analysis)")
    r.add_argument("--duration", type=float, help="video length in seconds (when no --video)")
    r.add_argument("--no-verify", action="store_true")
    r.add_argument("--preview", action="store_true", help="also build a preview mp4")
    common(r)
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("build", help="re-voice and re-assemble an existing project")
    b.add_argument("project", help="project folder")
    common(b)
    b.set_defaults(fn=cmd_build)

    v = sub.add_parser("voices", help="list ElevenLabs voices")
    common(v)
    v.set_defaults(fn=cmd_voices)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except StvError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
