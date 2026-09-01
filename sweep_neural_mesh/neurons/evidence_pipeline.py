"""
Evidence Pipeline — cross-referencing and corroboration logic.

Boosts evidence supported by multiple centers, suppresses isolated claims,
and detects contradictions across processing centers.
"""
from __future__ import annotations

import re
from typing import Any

from .signal import Signal


def cross_reference_evidence(
    evidence_signals: list[Signal],
    credibility_signals: list[Signal],
    causal_signals: list[Signal],
    contradiction_signals: list[Signal],
) -> tuple[list[Signal], list[Signal]]:
    """Cross-reference evidence across processing centers.

    Returns (boosted_signals, suppressed_signals).

    Logic:
      - Evidence supported by multiple centers → boost
      - Evidence contradicted by high-confidence contradictions → suppress
      - Evidence with causal links to other evidence → slight boost
      - Evidence with no corroboration → suppress
    """
    if not evidence_signals:
        return [], []

    evidence_texts = [s.data.get("evidence_text", "") for s in evidence_signals]
    corroboration: dict[int, int] = {i: 0 for i in range(len(evidence_signals))}
    contradiction_map: dict[int, float] = {i: 0.0 for i in range(len(evidence_signals))}

    # Credibility corroboration
    for cs in credibility_signals:
        cred_text = cs.data.get("evidence_text", "")
        for i, ev_text in enumerate(evidence_texts):
            if cred_text and ev_text and _texts_overlap(cred_text, ev_text):
                corroboration[i] += 1

    # Causal corroboration
    for cl in causal_signals:
        ev_a = cl.data.get("evidence_a", "")
        ev_b = cl.data.get("evidence_b", "")
        link_strength = cl.confidence
        for i, ev_text in enumerate(evidence_texts):
            if ev_text and (_texts_overlap(ev_text, ev_a) or _texts_overlap(ev_text, ev_b)):
                corroboration[i] += 1 if link_strength > 0.3 else 0

    # Contradiction suppression
    for ct in contradiction_signals:
        ev_a = ct.data.get("evidence_a", "")
        ev_b = ct.data.get("evidence_b", "")
        contra_strength = ct.confidence
        for i, ev_text in enumerate(evidence_texts):
            if ev_text and _texts_overlap(ev_text, ev_a):
                contradiction_map[i] = max(contradiction_map[i], contra_strength)
            if ev_text and _texts_overlap(ev_text, ev_b):
                contradiction_map[i] = max(contradiction_map[i], contra_strength)

    boosted: list[Signal] = []
    suppressed: list[Signal] = []

    for i, sig in enumerate(evidence_signals):
        corrob = corroboration[i]
        contra = contradiction_map[i]

        if corrob >= 2:
            boosted.append(sig)
        elif contra > 0.5:
            suppressed.append(sig)
        elif corrob == 0 and sig.confidence < 0.5:
            suppressed.append(sig)

    return boosted, suppressed


def apply_xref_adjustments(
    evidence_signals: list[Signal],
    boosted: list[Signal],
    suppressed: list[Signal],
) -> list[Signal]:
    """Apply cross-reference adjustments to evidence signals.

    Returns a new list with boosted/confidence-adjusted signals.
    """
    if not boosted and not suppressed:
        return evidence_signals

    boosted_ids = {id(s) for s in boosted}
    suppressed_ids = {id(s) for s in suppressed}
    adjusted: list[Signal] = []

    for es in evidence_signals:
        if id(es) in boosted_ids:
            adjusted.append(Signal(
                data={**es.data, "_xref_boosted": True},
                signal_type=es.signal_type,
                confidence=min(1.0, es.confidence * 1.15),
                source_center=es.source_center,
                metadata={**es.metadata, "xref_action": "boosted"},
                history=list(es.history),
            ))
        elif id(es) in suppressed_ids:
            adjusted.append(Signal(
                data={**es.data, "_xref_suppressed": True},
                signal_type=es.signal_type,
                confidence=max(0.0, es.confidence * 0.80),
                source_center=es.source_center,
                metadata={**es.metadata, "xref_action": "suppressed"},
                history=list(es.history),
            ))
        else:
            adjusted.append(es)

    return adjusted


def semantic_contradiction_score(text_a: str, text_b: str) -> float:
    """Detect semantic contradiction between two texts using embeddings.

    Returns a score 0.0-1.0 where higher = more contradictory.
    Uses cosine similarity in embedding space: semantically opposite
    statements have low similarity despite sharing topic words.
    """
    if not text_a or not text_b:
        return 0.0

    try:
        from .semantic_embeddings import get_embedder
        embedder = get_embedder()
        if embedder.backend not in ("minilm", "potion"):
            return 0.0

        result = embedder.similarity(text_a, text_b)

        # If texts are about the same topic (moderate similarity)
        # but have opposing meanings, the embedding similarity will be
        # lower than expected for paraphrases
        topic_words_a = set(re.findall(r'\b[a-z]{4,}\b', text_a.lower()))
        topic_words_b = set(re.findall(r'\b[a-z]{4,}\b', text_b.lower()))
        word_overlap = len(topic_words_a & topic_words_b) / max(len(topic_words_a | topic_words_b), 1)

        # If they share many words but have low embedding similarity,
        # they likely contradict each other
        if word_overlap > 0.2 and result.score < 0.5:
            return min(1.0, (1.0 - result.score) * word_overlap * 2)

        return 0.0
    except Exception:
        return 0.0


def _texts_overlap(text_a: str, text_b: str, threshold: float = 0.3) -> bool:
    """Check if two texts are semantically related using pretrained embeddings.

    Falls back to word overlap if embeddings are unavailable.
    """
    if not text_a or not text_b:
        return False

    # Try semantic similarity first
    try:
        from .semantic_embeddings import get_embedder
        embedder = get_embedder()
        result = embedder.similarity(text_a, text_b)
        if result.backend in ("minilm", "potion"):
            return result.score >= threshold
    except Exception:
        pass

    # Fallback: word overlap
    stop = {"the", "and", "for", "are", "but", "not", "you", "all",
            "can", "was", "this", "that", "with"}
    words_a = set(re.findall(r"\b[a-z]{4,}\b", text_a.lower())) - stop
    words_b = set(re.findall(r"\b[a-z]{4,}\b", text_b.lower())) - stop
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b)
    union = len(words_a | words_b)
    return (overlap / union if union else 0.0) >= threshold
