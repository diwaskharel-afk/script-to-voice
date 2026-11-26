"""A project is a folder holding everything for one video, so work can be resumed and edited."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import Clip, Issue, Placement, Section, StvError


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    return s or "project"


@dataclass
class Project:
    name: str
    root: Path
    video: str = ""
    duration: float = 0.0
    notes: str = ""
    persona: str = "Friendly trainer"
    document: str = ""
    sections: list[Section] = field(default_factory=list)
    clips: list[Clip] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    placements: list[Placement] = field(default_factory=list)
    output: str = ""

    # ---- paths
    @property
    def file(self) -> Path:
        return self.root / "project.json"

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    # ---- persistence
    @classmethod
    def create(cls, projects_dir: str | Path, name: str) -> "Project":
        root = Path(projects_dir) / slugify(name)
        (root / "clips").mkdir(parents=True, exist_ok=True)
        (root / "output").mkdir(parents=True, exist_ok=True)
        p = cls(name=name, root=root)
        if (root / "project.json").exists():
            return cls.load(root)
        p.save()
        return p

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "name": self.name, "video": self.video, "duration": self.duration, "notes": self.notes,
            "persona": self.persona, "output": self.output,
            "sections": [s.to_dict() for s in self.sections],
            "clips": [c.to_dict() for c in self.clips],
            "issues": [i.to_dict() for i in self.issues],
            "placements": [vars(p) for p in self.placements],
        }
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.file)
        (self.root / "document.txt").write_text(self.document, encoding="utf-8")

    @classmethod
    def load(cls, root: str | Path) -> "Project":
        root = Path(root)
        f = root / "project.json"
        if not f.exists():
            raise StvError(f"no project at {root}")
        d = json.loads(f.read_text(encoding="utf-8"))
        doc = root / "document.txt"
        return cls(
            name=d["name"], root=root, video=d.get("video", ""), duration=d.get("duration", 0.0),
            notes=d.get("notes", ""), persona=d.get("persona", "Friendly trainer"),
            document=doc.read_text(encoding="utf-8") if doc.exists() else "",
            sections=[Section.from_dict(s) for s in d.get("sections", [])],
            clips=[Clip.from_dict(c) for c in d.get("clips", [])],
            issues=[Issue(**i) for i in d.get("issues", [])],
            placements=[Placement(**p) for p in d.get("placements", [])],
            output=d.get("output", ""),
        )

    # ---- helpers
    def clip_for(self, index: int) -> Clip | None:
        return next((c for c in self.clips if c.section_index == index), None)

    @property
    def flagged(self) -> list[Clip]:
        return [c for c in self.clips if c.needs_human]
