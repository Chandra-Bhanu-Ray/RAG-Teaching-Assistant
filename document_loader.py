"""Load PDF / TXT / MD files and split them into overlapping chunks."""
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

import config


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: int


def load_pages(path: Path) -> list[tuple[int, str]]:
    """Return a list of (page_number, text). Text files are treated as one page."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if suffix in {".txt", ".md"}:
        return [(1, path.read_text(encoding="utf-8", errors="ignore"))]
    raise ValueError(f"Unsupported file type: {suffix} (use PDF, TXT or MD)")


def split_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Sliding-window splitter that prefers to cut at sentence boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end))
            if cut > start + size * 0.5:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_and_chunk(path: Path) -> list[Chunk]:
    path = Path(path)
    chunks: list[Chunk] = []
    cid = 0
    for page_no, page_text in load_pages(path):
        for piece in split_text(page_text):
            chunks.append(Chunk(text=piece, source=path.name, page=page_no, chunk_id=cid))
            cid += 1
    return chunks
