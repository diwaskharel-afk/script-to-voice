"""Content-addressed cache for generated clips, so unchanged lines cost no credits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ClipCache:
    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(**parts) -> str:
        blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> tuple[bytes, str] | None:
        for p in self.dir.glob(f"{key}.*"):
            self.hits += 1
            return p.read_bytes(), p.suffix.lstrip(".")
        self.misses += 1
        return None

    def put(self, key: str, data: bytes, ext: str) -> Path:
        p = self.dir / f"{key}.{ext}"
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
        return p
