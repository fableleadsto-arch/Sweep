"""Evidence store — in-memory evidence collection with deduplication."""

from __future__ import annotations

import re
import uuid
from typing import Optional

from ..core.types import Evidence, Source


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


class EvidenceStore:
    """In-memory evidence collection with deduplication."""

    def __init__(self) -> None:
        self._items: list[Evidence] = []
        self._sources: dict[str, Source] = {}
        self._seen_excerpts: set[str] = set()

    def add(self, evidence: Evidence) -> bool:
        """Add evidence. Returns False if it's a near-duplicate."""
        norm = _normalize_text(evidence.excerpt)
        if len(norm) < 40:
            return False
        if norm in self._seen_excerpts:
            return False
        for seen in self._seen_excerpts:
            if seen and (seen in norm or norm in seen):
                return False

        if not evidence.id:
            evidence.id = uuid.uuid4().hex[:12]
        self._items.append(evidence)
        self._seen_excerpts.add(norm)
        return True

    def track_source(self, source: Source) -> None:
        self._sources[source.url] = source

    def all(self) -> list[Evidence]:
        return list(self._items)

    def count(self) -> int:
        return len(self._items)

    def sources(self) -> list[Source]:
        return list(self._sources.values())

    def clear(self) -> None:
        self._items.clear()
        self._sources.clear()
        self._seen_excerpts.clear()

    def to_list(self) -> list[dict]:
        return [e.model_dump() for e in self._items]

    def distinct_sources(self) -> int:
        return len(set(e.source_url for e in self._items))
