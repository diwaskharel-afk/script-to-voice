"""Glossary that rewrites technical terms into how they should be spoken.

The script keeps the real spelling ("Kubernetes"); only the text sent to the voice
model is rewritten ("koo-ber-NET-eez"). ElevenLabs pronunciation dictionaries can be
used as well (see Settings.pronunciation_dictionary_ids).
"""
from __future__ import annotations

import json
import re
from pathlib import Path


class Glossary:
    def __init__(self, entries: dict[str, str] | None = None):
        self.entries = dict(entries or {})
        # longest first so "SQL Server" wins over "SQL"
        terms = sorted(self.entries, key=len, reverse=True)
        self._pattern = (
            re.compile(r"(?<!\w)(" + "|".join(re.escape(t) for t in terms) + r")(?!\w)", re.IGNORECASE)
            if terms else None
        )
        self._lower = {k.lower(): v for k, v in self.entries.items()}

    def apply(self, text: str) -> str:
        if not self._pattern:
            return text
        return self._pattern.sub(lambda m: self._lower[m.group(1).lower()], text)

    @classmethod
    def load(cls, path: str | Path | None) -> "Glossary":
        """Load JSON ({"term": "spoken"}) or a text file with `term = spoken` lines."""
        if not path:
            return cls()
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            return cls(json.loads(raw))
        entries = {}
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            entries[k.strip()] = v.strip()
        return cls(entries)

    @classmethod
    def from_text(cls, text: str) -> "Glossary":
        entries = {}
        for line in text.splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                if k.strip():
                    entries[k.strip()] = v.strip()
        return cls(entries)
