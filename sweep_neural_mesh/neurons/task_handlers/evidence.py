"""
Evidence Handler — evidence analysis, corroboration, and reasoning.

Handles:
  - Corroboration: do multiple sources agree?
  - Contradiction detection: find conflicting claims
  - Source ranking: assess reliability
  - Entity resolution: match entities across records
  - Evidence scoring: relevance and strength assessment
  - Claim verification: check claims against evidence
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any


@dataclass(frozen=True, slots=True)
class EvidenceResult:
    """Structured result from evidence analysis."""
    answer: str
    confidence: float
    method: str
    details: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class EvidenceItem:
    """A piece of evidence with metadata."""
    text: str
    source: str = "unknown"
    reliability: float = 0.5
    timestamp: str | None = None
    entities: list[str] = field(default_factory=list)


class EvidenceHandler:
    """Handles evidence analysis and reasoning tasks."""

    # ── Reliability scores for common source types ──────
    SOURCE_RELIABILITY = {
        "official": 0.95,
        "government": 0.95,
        "academic": 0.90,
        "peer-reviewed": 0.90,
        "journal": 0.85,
        "news": 0.75,
        "wikipedia": 0.80,
        "reputable": 0.80,
        "blog": 0.50,
        "social": 0.40,
        "anonymous": 0.30,
        "unknown": 0.50,
        "fabricated": 0.10,
        "fake": 0.10,
    }

    def process(self, query: str, evidence: list[str] | None = None) -> EvidenceResult:
        t0 = time.perf_counter()
        ev = evidence or []

        # Try simple claim first (works with 1+ evidence items)
        result = self._try_simple_claim(query, ev, t0)
        if result:
            return result

        result = self._try_corroboration(query, ev, t0)
        if result:
            return result

        result = self._try_contradiction(query, ev, t0)
        if result:
            return result

        result = self._try_source_ranking(query, ev, t0)
        if result:
            return result

        result = self._try_entity_resolution(query, ev, t0)
        if result:
            return result

        result = self._try_claim_verification(query, ev, t0)
        if result:
            return result

        result = self._try_evidence_scoring(query, ev, t0)
        if result:
            return result

        return EvidenceResult(
            answer="", confidence=0.0, method="none",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # -- Simple Claim Verification (single evidence) ----------------------

    def _try_simple_claim(self, query: str, ev: list[str], t0: float) -> EvidenceResult | None:
        """Handle simple claim queries like 'Does X improve Y?' with 1+ evidence items."""
        q_lower = query.lower()
        if not ev:
            return None

        # Match patterns: "Is X good?", "Does X improve Y?", "Is exercise beneficial?",
        # "Is smoking harmful?", "Is X real?", "Does X work?"
        claim_match = re.search(
            r'\b(is|does|do|can|will|should|has|have)\s+(.+?)\s+(good|bad|harmful|beneficial|benefit|'
            r'effective|work|safe|real|true|possible|necessary|important|improve|enhance|help|'
            r'cause|prevent|reduce|risk|increase|decrease|better|worse|healthy|dangerous|'
            r'efficient|reliable|accurate|correct|supported|proven)\b',
            q_lower,
        )
        if not claim_match:
            return None

        topic = claim_match.group(2).strip()
        keyword = claim_match.group(3).strip()
        topic_words = set(re.findall(r'\b\w{3,}\b', topic))
        topic_words.add(keyword)

        support = 0
        refute = 0
        for e in ev:
            e_lower = e.lower()
            e_words = set(re.findall(r'\b\w{3,}\b', e_lower))
            overlap = len(topic_words & e_words) / max(len(topic_words), 1)
            if overlap < 0.15:
                continue

            negation_words = {'not', 'no', 'never', 'fail', 'refute', 'contradict', 'deny', 'false', 'ineffective', 'harmful', 'dangerous'}
            has_neg = any(w in e_lower for w in negation_words)
            # Also check for negation via word-level
            has_neg_phrase = any(phrase in e_lower for phrase in ['not support', 'no evidence', 'is not', 'are not', 'does not', 'do not'])
            if has_neg or has_neg_phrase:
                refute += overlap
            else:
                support += overlap

        total = support + refute
        if total == 0:
            return None

        if support > refute:
            direction = 'supports' if keyword not in ('harmful', 'bad', 'dangerous', 'worse') else 'confirms'
            return EvidenceResult(
                answer=direction,
                confidence=min(0.85, 0.5 + support * 0.3),
                method='simple_claim',
                details={'support': support, 'refute': refute, 'topic': topic},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        elif refute > support:
            return EvidenceResult(
                answer='refutes',
                confidence=min(0.85, 0.5 + refute * 0.3),
                method='simple_claim',
                details={'support': support, 'refute': refute, 'topic': topic},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        else:
            return EvidenceResult(
                answer='mixed',
                confidence=0.50,
                method='simple_claim',
                details={'support': support, 'refute': refute, 'topic': topic},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    # ── Corroboration ────────────────────────────────────

    def _try_corroboration(self, query: str, ev: list[str], t0: float) -> EvidenceResult | None:
        """Check if multiple sources corroborate the same claim."""
        if len(ev) < 2:
            return None

        q_lower = query.lower()
        q_words = set(re.findall(r"\b\w{3,}\b", q_lower))

        # Score each evidence item against the query
        scored = []
        for e in ev:
            e_words = set(re.findall(r"\b\w{3,}\b", e.lower()))
            overlap = len(q_words & e_words) / max(len(q_words), 1)
            scored.append((e, overlap))

        # Group by similarity (corroboration clusters)
        clusters = []
        used = set()
        for i, (e1, s1) in enumerate(scored):
            if i in used:
                continue
            cluster = [e1]
            used.add(i)
            for j, (e2, s2) in enumerate(scored):
                if j in used:
                    continue
                # Check if e1 and e2 are similar enough
                sim = SequenceMatcher(None, e1.lower(), e2.lower()).ratio()
                if sim > 0.4:
                    cluster.append(e2)
                    used.add(j)
            clusters.append(cluster)

        # Find the best corroboration cluster
        best_cluster = max(clusters, key=len) if clusters else []
        num_sources = len(best_cluster)

        if num_sources >= 2:
            confidence = min(0.95, 0.5 + num_sources * 0.12)
            chain = [
                f"Found {num_sources} corroborating sources",
                f"Cluster: {[e[:60] for e in best_cluster[:3]]}",
            ]
            return EvidenceResult(
                answer=f"Corroborated by {num_sources} sources: {best_cluster[0][:200]}",
                confidence=confidence,
                method="corroboration",
                details={"clusters": len(clusters), "best_size": num_sources},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        return None

    # ── Contradiction Detection ──────────────────────────

    def _try_contradiction(self, query: str, ev: list[str], t0: float) -> EvidenceResult | None:
        """Find contradictions between evidence items."""
        if len(ev) < 2:
            return None

        contradictions = []
        negation_words = {"not", "no", "never", "neither", "nor", "nobody", "nothing",
                          "nowhere", "hardly", "barely", "scarcely", "seldom", "rarely"}
        affirmation_words = {"always", "certainly", "definitely", "clearly", "obviously",
                             "undoubtedly", "indeed", "fact", "proven", "confirmed"}

        for i, e1 in enumerate(ev):
            for j, e2 in enumerate(ev):
                if j <= i:
                    continue
                e1_lower = e1.lower()
                e2_lower = e2.lower()

                # Check for direct negation patterns
                # "X is Y" vs "X is not Y"
                for pattern in [
                    (r"(\w+)\s+(is|are|was|were)\s+(.+)", r"(\w+)\s+(is|are|was|were)\s+not\s+(.+)"),
                    (r"(.+)\s+(increases?|rises?|grows?)", r"(.+)\s+(decreases?|falls?|drops?|declines?)"),
                ]:
                    m1 = re.search(pattern[0], e1_lower)
                    m2 = re.search(pattern[1], e2_lower)
                    if m1 and m2 and m1.group(1) == m2.group(1):
                        contradictions.append((e1, e2, f"Direct contradiction on '{m1.group(1)}'"))

                # Check for affirmation vs negation of same topic
                e1_has_neg = any(w in e1_lower.split() for w in negation_words)
                e2_has_neg = any(w in e2_lower.split() for w in negation_words)
                e1_has_aff = any(w in e1_lower.split() for w in affirmation_words)
                e2_has_aff = any(w in e2_lower.split() for w in affirmation_words)

                if e1_has_neg and e2_has_aff and not e2_has_neg:
                    # Check topic overlap
                    e1_words = set(re.findall(r"\b\w{3,}\b", e1_lower))
                    e2_words = set(re.findall(r"\b\w{3,}\b", e2_lower))
                    if len(e1_words & e2_words) > len(e1_words) * 0.3:
                        contradictions.append((e1, e2, "Negation vs affirmation"))

        if contradictions:
            best = contradictions[0]
            return EvidenceResult(
                answer=f"Contradiction found: {best[2]}",
                confidence=0.80,
                method="contradiction",
                details={
                    "contradiction_count": len(contradictions),
                    "pairs": [(c[0][:100], c[1][:100], c[2]) for c in contradictions[:5]],
                },
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        return None

    # ── Source Ranking ───────────────────────────────────

    def _try_source_ranking(self, query: str, ev: list[str], t0: float) -> EvidenceResult | None:
        """Rank evidence by source reliability."""
        if not ev:
            return None

        q_lower = query.lower()

        # Check if the query asks about sources or reliability
        if not any(w in q_lower for w in ("source", "reliable", "trust", "credible", "rank", "best")):
            return None

        scored = []
        for e in ev:
            e_lower = e.lower()
            # Estimate source reliability
            reliability = 0.5  # Default
            for keyword, score in self.SOURCE_RELIABILITY.items():
                if keyword in e_lower:
                    reliability = max(reliability, score)

            # Boost for specific evidence markers
            if re.search(r"\b\d{4}\b", e):  # Has a date
                reliability = min(1.0, reliability + 0.05)
            if any(w in e_lower for w in ["study", "research", "data", "evidence"]):
                reliability = min(1.0, reliability + 0.10)

            scored.append((e, reliability))

        scored.sort(key=lambda x: x[1], reverse=True)

        ranking = [
            {"text": e[:100], "reliability": round(r, 2)}
            for e, r in scored[:5]
        ]

        return EvidenceResult(
            answer=f"Most reliable: {scored[0][0][:200]}",
            confidence=0.75,
            method="source_ranking",
            details={"ranking": ranking},
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ── Entity Resolution ────────────────────────────────

    def _try_entity_resolution(self, query: str, ev: list[str], t0: float) -> EvidenceResult | None:
        """Resolve entity references across multiple records."""
        if len(ev) < 2:
            return None

        q_lower = query.lower()
        if not any(w in q_lower for w in ("same", "different", "entity", "person", "match", "duplicate")):
            return None

        # Extract potential entity references
        entities = []
        for e in ev:
            # Simple entity extraction: capitalized words, email-like patterns
            names = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", e)
            emails = re.findall(r"[\w.]+@[\w.]+", e)
            phones = re.findall(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", e)
            entities.append({
                "text": e,
                "names": names,
                "emails": emails,
                "phones": phones,
            })

        # Check for matching entities
        matches = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1, e2 = entities[i], entities[j]

                # Email match
                if e1["emails"] and e2["emails"]:
                    if set(e1["emails"]) & set(e2["emails"]):
                        matches.append((i, j, "email match"))

                # Phone match
                if e1["phones"] and e2["phones"]:
                    if set(e1["phones"]) & set(e2["phones"]):
                        matches.append((i, j, "phone match"))

                # Name similarity
                for n1 in e1["names"]:
                    for n2 in e2["names"]:
                        sim = SequenceMatcher(None, n1.lower(), n2.lower()).ratio()
                        if sim > 0.8:
                            matches.append((i, j, f"name similarity ({n1} ~ {n2}, {sim:.0%})"))

        if matches:
            return EvidenceResult(
                answer=f"Found {len(matches)} entity matches",
                confidence=0.80,
                method="entity_resolution",
                details={"matches": [(m[0], m[1], m[2]) for m in matches[:10]]},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Check if query asks "are they the same?"
        if "same" in q_lower or "match" in q_lower:
            return EvidenceResult(
                answer="insufficient evidence to determine",
                confidence=0.40,
                method="entity_resolution",
                details={"matches": []},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        return None

    # ── Claim Verification ───────────────────────────────

    def _try_claim_verification(self, query: str, ev: list[str], t0: float) -> EvidenceResult | None:
        """Verify a claim against provided evidence."""
        q_lower = query.lower()

        if not any(w in q_lower for w in ("verify", "check", "true", "correct", "supported", "claim")):
            return None

        if not ev:
            return EvidenceResult(
                answer="insufficient evidence to verify",
                confidence=0.30,
                method="claim_verification",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Extract the claim from the query
        claim_match = re.search(
            r"(?:verify|check|is|does)\s+(.+?)(?:\s+(?:based|using|with)\s+evidence)?\??$",
            q_lower,
        )
        if not claim_match:
            return None

        claim = claim_match.group(1).strip()
        claim_words = set(re.findall(r"\b\w{3,}\b", claim))

        # Check evidence support
        support_score = 0
        refute_score = 0
        for e in ev:
            e_lower = e.lower()
            e_words = set(re.findall(r"\b\w{3,}\b", e_lower))
            overlap = len(claim_words & e_words) / max(len(claim_words), 1)

            if overlap > 0.3:
                # Check for negation
                negation_words = {"not", "no", "never", "contradicts", "refutes", "denies", "false"}
                has_negation = any(w in e_lower for w in negation_words)
                if has_negation:
                    refute_score += overlap
                else:
                    support_score += overlap

        total = support_score + refute_score
        if total == 0:
            return EvidenceResult(
                answer="insufficient evidence to verify",
                confidence=0.30,
                method="claim_verification",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        if support_score > refute_score * 2:
            return EvidenceResult(
                answer="supported by evidence",
                confidence=min(0.90, 0.5 + support_score * 0.3),
                method="claim_verification",
                details={"support": support_score, "refute": refute_score},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        elif refute_score > support_score * 2:
            return EvidenceResult(
                answer="refuted by evidence",
                confidence=min(0.90, 0.5 + refute_score * 0.3),
                method="claim_verification",
                details={"support": support_score, "refute": refute_score},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        else:
            return EvidenceResult(
                answer="mixed evidence — cannot determine",
                confidence=0.50,
                method="claim_verification",
                details={"support": support_score, "refute": refute_score},
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    # ── Evidence Scoring ─────────────────────────────────

    def _try_evidence_scoring(self, query: str, ev: list[str], t0: float) -> EvidenceResult | None:
        """Score evidence relevance and strength."""
        if not ev:
            return None

        q_lower = query.lower()
        if not any(w in q_lower for w in ("score", "relevance", "strength", "quality", "rate")):
            return None

        q_words = set(re.findall(r"\b\w{3,}\b", q_lower))

        scored = []
        for e in ev:
            e_words = set(re.findall(r"\b\w{3,}\b", e.lower()))
            overlap = len(q_words & e_words) / max(len(q_words), 1)

            # Length bonus (longer evidence is usually more informative)
            length_score = min(1.0, len(e) / 200)

            # Specificity bonus (more specific = better)
            specificity = len(e_words) / max(len(e.split()), 1)

            total = overlap * 0.5 + length_score * 0.25 + specificity * 0.25
            scored.append((e, round(total, 3)))

        scored.sort(key=lambda x: x[1], reverse=True)

        return EvidenceResult(
            answer=f"Best evidence: {scored[0][0][:200]}",
            confidence=scored[0][1],
            method="evidence_scoring",
            details={"scores": [(e[:80], s) for e, s in scored[:5]]},
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
