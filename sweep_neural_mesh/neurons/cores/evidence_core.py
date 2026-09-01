"""
EvidenceCore — evidence analysis and extraction.

Responsibilities:
  - Score evidence relevance against the query.
  - Extract definitions (X is Y) from evidence text.
  - Select the most relevant evidence item.
"""
from __future__ import annotations

import re
import time

from ..core_protocol import CoreResult, make_result, empty_result


class EvidenceCore:
    """Core C — Evidence analysis and extraction.

    Given a query and a list of evidence strings, this core:
      1. Computes word-overlap relevance between query and each evidence item.
      2. Tries to extract a definition pattern (X is/are/was Y).
      3. Returns the best-scoring evidence or a definition.
    """

    CORE_ID = "evidence"

    @property
    def core_id(self) -> str:
        return self.CORE_ID

    def process(self, query: str, evidence: list[str]) -> CoreResult:
        t0 = time.perf_counter()

        if not evidence:
            return empty_result(self.CORE_ID, t0, "No evidence provided")

        q = query.lower()
        query_words = set(re.findall(r"\b\w{3,}\b", q))

        best_match = ""
        best_score = 0.0

        for ev in evidence:
            ev_words = set(re.findall(r"\b\w{3,}\b", ev.lower()))
            overlap = len(query_words & ev_words)
            score = overlap / max(len(query_words), 1)
            if score > best_score:
                best_score = score
                best_match = ev

        if best_match:
            # Try to extract a definition
            def_match = re.search(
                r"(\w+(?:\s+\w+)*)\s+(?:is|are|was|were)\s+(.+?)(?:\.|$)",
                best_match,
            )
            if def_match:
                return make_result(
                    self.CORE_ID,
                    def_match.group(2).strip()[:200],
                    0.8,
                    "Definition extracted from evidence",
                    t0,
                    evidence_used=1,
                )

            return make_result(
                self.CORE_ID,
                best_match[:200],
                min(0.8, 0.5 + best_score * 0.3),
                f"Best evidence match (score: {best_score:.2f})",
                t0,
                evidence_used=1,
            )

        return make_result(
            self.CORE_ID,
            evidence[0][:200],
            0.4,
            "Returning first evidence item",
            t0,
            evidence_used=1,
        )
