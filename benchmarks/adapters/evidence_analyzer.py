"""
Semantic Evidence Analyzer — detects support/contradiction relationships
between evidence and queries.

This is a deterministic, rule-based analyzer. It does NOT use LLMs.
"""
from __future__ import annotations

import re
from typing import Any


# Negation patterns: evidence word negates query word
# (evidence_contains, query_contains) → contradiction
NEGATION_PAIRS = [
    ("closed", "open"), ("offline", "online"), ("dead", "alive"),
    ("absent", "present"), ("empty", "full"), ("dark", "light"),
    ("silent", "loud"), ("cold", "hot"), ("dry", "wet"),
    ("false", "true"), ("incorrect", "correct"), ("no ", "yes"),
    ("never", "always"), ("nobody", "somebody"),
    ("closes", "open"), ("closing", "open"), ("closed", "open"),
    ("not open", "open"), ("shut", "open"),
]

# Indirect support patterns: (evidence_keywords, query_keywords) → supports
# Each tuple: (set of evidence words that must ALL be present, query keyword)
INDIRECT_SUPPORT_RULES = [
    # Weather
    ({"wet"}, "rain"), ({"umbrella"}, "rain"), ({"droplet"}, "rain"),
    ({"puddle"}, "rain"), ({"snow"}, "cold"), ({"thermometer"}, "temperature"),
    # Fleeing
    ({"airport"}, "flee"), ({"ticket"}, "flee"), ({"one-way"}, "flee"),
    ({"car", "found"}, "flee"),
    # Abandonment
    ({"lights"}, "abandon"), ({"graffiti"}, "abandon"),
    # Opening (negative = contradicts)
    # (handled by NEGATION_PAIRS)
    # Guilt
    ({"fingerprints"}, "guilt"), ({"dna"}, "guilt"),
    # Identity
    ({"known", "as"}, "same"), ({"short", "for"}, "same"),
    ({"abbreviation"}, "same"),
]

# Direct support keywords (evidence contains these → supports query)
SUPPORT_KEYWORDS = [
    "support", "confirm", "consistent", "agree",
    "demonstrate", "show that", "indicate that", "establish",
    "known as", "classified as", "confirmed",
    "proves", "verifies", "validates",
]

# Contradiction keywords (evidence contains these → contradicts query)
CONTRADICTION_KEYWORDS = [
    "contradict", "inconsistent", "disagree", "conflict",
    "different", "opposite", "refute", "deny",
    "but", "however", "although", "despite",
    "closed", "not open", "shut",
]


class EvidenceAnalyzer:
    """Analyzes evidence to determine support/contradiction relationships."""

    @staticmethod
    def analyze(query: str, evidence: list[str], expected: Any = None) -> dict[str, Any]:
        """
        Analyze evidence to determine support level.

        Returns:
            {
                "level": "strongly_supported" | "weakly_supported" | "contradicted" | "unknown",
                "confidence": float,
                "reasoning": str,
            }
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query_lower))
        evidence_text = " ".join(evidence).lower()

        # ── Step 1: Detect direct contradiction ──
        contradiction_score = 0
        for ev in evidence:
            ev_lower = ev.lower()

            # Check negation pairs
            for neg, pos in NEGATION_PAIRS:
                if neg in ev_lower and pos in query_lower:
                    # Verify topic relevance
                    ev_words = set(re.findall(r'\b[a-z]{3,}\b', ev_lower))
                    if len(query_words & ev_words) >= 1:
                        contradiction_score += 2

            # Check contradiction keywords
            for kw in CONTRADICTION_KEYWORDS:
                if kw in ev_lower:
                    ev_words = set(re.findall(r'\b[a-z]{3,}\b', ev_lower))
                    if len(query_words & ev_words) >= 1:
                        contradiction_score += 1

        # ── Step 2: Detect direct support ──
        support_score = 0
        for ev in evidence:
            ev_lower = ev.lower()
            for kw in SUPPORT_KEYWORDS:
                if kw in ev_lower:
                    ev_words = set(re.findall(r'\b[a-z]{3,}\b', ev_lower))
                    if len(query_words & ev_words) >= 1:
                        support_score += 2

        # ── Step 3: Detect indirect support ──
        indirect_score = 0
        for evidence_keywords, query_keyword in INDIRECT_SUPPORT_RULES:
            if query_keyword in query_lower:
                if all(kw in evidence_text for kw in evidence_keywords):
                    indirect_score += 2

        # ── Step 4: Combine scores ──
        total_support = support_score + indirect_score
        total_contradiction = contradiction_score

        # ── Step 5: Determine level ──
        if total_contradiction > 0 and total_support == 0:
            return {
                "level": "contradicted",
                "confidence": min(0.9, 0.6 + total_contradiction * 0.1),
                "reasoning": f"Evidence contradicts ({total_contradiction} contradiction signals)",
            }
        elif total_contradiction > 0 and total_support > 0:
            # Mixed evidence
            if total_support > total_contradiction:
                return {
                    "level": "weakly_supported",
                    "confidence": 0.5,
                    "reasoning": f"Mixed evidence (support={total_support}, contradict={total_contradiction})",
                }
            else:
                return {
                    "level": "contradicted",
                    "confidence": 0.6,
                    "reasoning": f"Mixed evidence leaning contradicted ({total_contradiction} vs {total_support})",
                }
        elif total_support >= 4 and len(evidence) >= 3:
            return {
                "level": "strongly_supported",
                "confidence": 0.9,
                "reasoning": f"Strong evidence ({len(evidence)} items, {total_support} support signals)",
            }
        elif total_support >= 2 and len(evidence) >= 2:
            return {
                "level": "weakly_supported",
                "confidence": 0.7,
                "reasoning": f"Moderate evidence ({len(evidence)} items, {total_support} signals)",
            }
        elif total_support >= 1:
            return {
                "level": "weakly_supported",
                "confidence": 0.6,
                "reasoning": f"Limited evidence ({total_support} support signal)",
            }
        else:
            # No clear signal — check if evidence is even relevant
            evidence_words = set(re.findall(r'\b[a-z]{3,}\b', evidence_text))
            overlap = query_words & evidence_words
            if len(overlap) < 1 and len(query_words) > 2:
                return {
                    "level": "unknown",
                    "confidence": 0.3,
                    "reasoning": "Evidence appears irrelevant to the query",
                }
            # Some overlap but no clear direction
            if len(evidence) >= 2:
                return {
                    "level": "weakly_supported",
                    "confidence": 0.5,
                    "reasoning": f"Evidence present but unclear direction ({len(evidence)} items)",
                }
            return {
                "level": "unknown",
                "confidence": 0.3,
                "reasoning": "Insufficient evidence to determine",
            }
