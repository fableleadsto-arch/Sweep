"""Pure text utilities for the ingestion pipeline.

Chunking mirrors the TS stack's ``chunkText`` (600 chars / 60 overlap) so the
embedding units stay consistent with what the brain already retrieves.
"""

from __future__ import annotations

import re

from .security import sanitize_content

CHUNK_SIZE = 600
CHUNK_OVERLAP = 60

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WS = re.compile(r"\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on common English boundaries."""
    text = sanitize_content(text, max_chars=0)
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def chunk_text(
    text: str,
    *,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character chunks with soft boundaries.

    Chunks break at the last sentence boundary inside ``size`` when one exists
    (keeping the overlap window intact), so retrieved chunks are readable.
    """
    cleaned = sanitize_content(text, max_chars=0)
    if not cleaned:
        return []
    if len(cleaned) <= size:
        return [cleaned]
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + size, len(cleaned))
        if end < len(cleaned):
            window = cleaned[start:end]
            # Prefer the last sentence boundary before `size` for a clean cut.
            for m in _SENTENCE_END.finditer(window):
                pass
            boundary = None
            for m in _SENTENCE_END.finditer(window):
                boundary = m.end()
            if boundary is not None and boundary >= size // 2:
                end = start + boundary
        chunks.append(cleaned[start:end].strip())
        if end >= len(cleaned):
            break
        start = end - overlap
    return [c for c in chunks if c]


def count_tokens(text: str) -> int:
    """Cheap token estimate (whitespace words) — used for chunk bookkeeping."""
    return len(_WS.split(text.strip())) if text.strip() else 0
