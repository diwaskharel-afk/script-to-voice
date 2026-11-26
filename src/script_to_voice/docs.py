"""Read documentation (PDF, Markdown, text) into plain text for the script writer."""
from __future__ import annotations

from pathlib import Path

from .models import StvError

MAX_DOC_CHARS = 60_000


def read_document(path: str | Path) -> str:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(p))
            return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
        except Exception as e:  # noqa: BLE001
            raise StvError(f"could not read PDF {p.name}: {e}") from e
    if suffix in {".md", ".txt", ".rst", ""}:
        return p.read_text(encoding="utf-8", errors="replace")
    raise StvError(f"unsupported documentation type: {p.suffix} (use PDF, MD or TXT)")


def read_documents(paths: list[str | Path], limit: int = MAX_DOC_CHARS) -> str:
    parts = []
    for p in paths:
        parts.append(f"# Document: {Path(p).name}\n{read_document(p)}")
    text = "\n\n".join(parts)
    return text if len(text) <= limit else text[:limit] + "\n[...documentation truncated...]"
