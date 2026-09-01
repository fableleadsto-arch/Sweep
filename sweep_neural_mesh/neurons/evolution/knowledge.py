"""
KnowledgeAcquisition — acquires and validates new knowledge.

Responsibilities:
  - Validate new information before adding it.
  - Deduplicate against existing knowledge.
  - Assign confidence based on source reliability.
  - Maintain a growing knowledge base from interactions.
"""
from __future__ import annotations

import re
from typing import Any


# Source reliability ratings (0.0 – 1.0)
TRUSTED_SOURCES: dict[str, float] = {
    "wikipedia": 0.9,
    "wikidata": 0.85,
    "user_feedback": 0.95,
    "self_discovery": 0.7,
}


class KnowledgeAcquisition:
    """Acquires new knowledge from external sources.

    Validates, deduplicates, and confidence-rates every new entry
    before adding it to the internal knowledge base.
    """

    def __init__(self) -> None:
        self._acquired: list[tuple[str, str, float, str]] = []

    def acquire(
        self,
        pattern: str,
        answer: str,
        source: str,
        confidence: float = 0.8,
    ) -> bool:
        """Acquire new knowledge.  Returns True if added, False if rejected."""
        if not self._validate(pattern, answer):
            return False
        if self._is_duplicate(pattern, answer):
            return False

        source_reliability = TRUSTED_SOURCES.get(source, 0.5)
        adjusted = confidence * source_reliability
        self._acquired.append((pattern, answer, adjusted, source))
        return True

    def get_knowledge(self, min_confidence: float = 0.5) -> list[tuple[str, str, float]]:
        return [(p, a, c) for p, a, c, _ in self._acquired if c >= min_confidence]

    def get_stats(self) -> dict[str, Any]:
        sources = set(s for _, _, _, s in self._acquired)
        return {
            "total_acquired": len(self._acquired),
            "by_source": {s: sum(1 for _, _, _, src in self._acquired if src == s) for s in sources},
        }

    # ── Internal ────────────────────────────────────────────

    def _validate(self, pattern: str, answer: str) -> bool:
        if not pattern or not answer or len(pattern) < 3 or len(answer) < 1:
            return False
        try:
            re.compile(pattern)
        except re.error:
            return False
        return True

    def _is_duplicate(self, pattern: str, answer: str) -> bool:
        return any(p == pattern and a == answer for p, a, _, _ in self._acquired)
