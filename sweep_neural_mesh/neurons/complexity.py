"""
Complexity — adaptive pipeline depth based on query complexity.

Classifies queries into trivial/simple/moderate/deep and selects
which human reasoning modules to activate.
"""
from __future__ import annotations

import re


# ── Complexity patterns ────────────────────────────────────────

_SIMPLE_PATTERNS = [
    r"\b(what is|who is|when did|where is|how many)\b",
    r"\b(define|meaning of|abbreviation)\b",
]

_MODERATE_PATTERNS = [
    r"\b(compare|difference|why|how does|explain)\b",
    r"\b(best|worse|should|recommend)\b",
]

_DEEP_PATTERNS = [
    r"\b(counterfactual|what if|suppose|imagine)\b",
    r"\banalog(y|ous|ize)\b",
    r"\b(narrative|story|sequence of events)\b",
    r"\b(caus(al|e)|root cause|consequence)\b",
]


def classify_query_complexity(query: str, evidence_count: int) -> str:
    """Classify query complexity.

    Returns: 'trivial', 'simple', 'moderate', 'complex', or 'deep'.
    """
    q = query.lower().strip()
    word_count = len(q.split())

    # Trivial: greetings, single-word queries
    if word_count <= 2 and evidence_count == 0:
        return "trivial"

    # Simple: factual lookups
    if (word_count <= 8
            and any(re.search(p, q) for p in _SIMPLE_PATTERNS)
            and evidence_count < 3):
        return "simple"

    # Moderate: analysis questions
    if word_count <= 15 and evidence_count < 8:
        if any(re.search(p, q) for p in _MODERATE_PATTERNS):
            return "moderate"

    # Deep: complex multi-step reasoning
    has_deep = any(re.search(p, q) for p in _DEEP_PATTERNS)

    # Complex: evidence-heavy or multi-source
    if evidence_count >= 8 or (evidence_count >= 4 and word_count >= 15) or has_deep:
        return "deep"

    return "moderate"


def select_reasoning_modules(complexity: str, evidence_count: int) -> list[str]:
    """Select which human reasoning modules to activate.

    Adaptive pipeline depth:
      - trivial:  skip all reasoning
      - simple:   common_sense only
      - moderate: common_sense + abductive
      - complex:  + theory_of_mind + causal
      - deep:     all 7 modules
    """
    if complexity == "trivial":
        return []

    modules = ["common_sense"]

    if complexity in ("moderate", "complex", "deep"):
        modules.append("abductive")

    if complexity in ("complex", "deep"):
        modules.append("theory_of_mind")
        modules.append("causal")

    if complexity == "deep":
        modules.append("narrative")
        modules.append("analogical")
        modules.append("counterfactual")

    return modules
