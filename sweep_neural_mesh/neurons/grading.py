"""
Evidence Grading — multi-dimensional assessment replacing confidence scores.

Instead of a single confidence score (like every other system),
Sweep uses a multi-dimensional grading system that provides
detailed, actionable assessments across multiple axes.

This is what makes Sweep DIFFERENT:
- Not just "confidence: 72%"
- But: "Depth: A, Breadth: B+, Reliability: A-, Coherence: C+, Actionability: B"

Each dimension captures a different aspect of evidence quality:

    DEPTH: How thoroughly was the evidence analyzed?
        - Did we examine it from multiple angles?
        - Did we check internal consistency?
        - Did we cross-reference with other evidence?

    BREADTH: How many perspectives were considered?
        - How many sources were consulted?
        - Did we consider opposing viewpoints?
        - Did we cover the full scope of the query?

    NOVELTY: How new/unique is this information?
        - Is this freshly discovered or well-established?
        - Does it contradict existing knowledge?
        - Does it fill a gap in understanding?

    RELIABILITY: How trustworthy is the source?
        - Source reputation and authority
        - Internal consistency of the evidence
        - Corroboration with other sources

    COHERENCE: How well do the evidence items fit together?
        - Do they support a consistent narrative?
        - Are there contradictions?
        - Do the pieces form a logical chain?

    ACTIONABILITY: How actionable is this information?
        - Can we make a decision based on this?
        - Is there a clear path forward?
        - What are the implications?

Each dimension is graded on a 0-100 scale with a letter grade:

    93-100: A+    87-92: A     80-86: A-
    77-79: B+     73-76: B     70-72: B-
    67-69: C+     63-66: C     60-62: C-
    57-59: D+     53-56: D     50-52: D-
    Below 50: F

The OVERALL GRADE is a weighted combination, not a simple average:
- Depth and Reliability are weighted more heavily (30% each)
- Breadth and Coherence are moderate (15% each)
- Novelty and Actionability are lighter (5% each)

This means a system that is deep and reliable will score higher
than one that is broad but shallow.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("sweep.neurons.grading")


# Letter grade boundaries
GRADE_BOUNDARIES: list[tuple[int, str]] = [
    (93, "A+"),
    (87, "A"),
    (80, "A-"),
    (77, "B+"),
    (73, "B"),
    (70, "B-"),
    (67, "C+"),
    (63, "C"),
    (60, "C-"),
    (57, "D+"),
    (53, "D"),
    (50, "D-"),
    (0, "F"),
]


def score_to_grade(score: float) -> str:
    """Convert a 0-100 score to a letter grade."""
    s = max(0, min(100, score * 100))  # normalize 0.0-1.0 to 0-100
    for threshold, grade in GRADE_BOUNDARIES:
        if s >= threshold:
            return grade
    return "F"


def score_to_percentage(score: float) -> str:
    """Convert a 0.0-1.0 score to a formatted percentage."""
    return f"{max(0, min(100, score * 100)):.0f}%"


@dataclass
class DimensionGrade:
    """A single grading dimension."""
    name: str
    score: float              # 0.0–1.0
    letter_grade: str         # A+, A, A-, etc.
    percentage: str           # "85%"
    reasoning: str            # why this grade
    evidence_count: int = 0   # how many evidence items contributed
    factors: list[str] = field(default_factory=list)  # specific factors

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "letter_grade": self.letter_grade,
            "percentage": self.percentage,
            "reasoning": self.reasoning,
            "evidence_count": self.evidence_count,
            "factors": self.factors,
        }


@dataclass
class EvidenceGrade:
    """
    The complete multi-dimensional grade for evidence evaluation.

    This replaces the single confidence score with a rich,
    multi-faceted assessment that tells you exactly WHY
    the evidence was evaluated this way.
    """
    # Individual dimensions
    depth: DimensionGrade
    breadth: DimensionGrade
    novelty: DimensionGrade
    reliability: DimensionGrade
    coherence: DimensionGrade
    actionability: DimensionGrade

    # Overall
    overall_score: float
    overall_grade: str
    overall_percentage: str
    overall_reasoning: str

    # Metadata
    total_evidence_count: int
    processing_phase: str     # "novice", "practice", "mastery"
    grade_dimensions: list[DimensionGrade] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth.to_dict(),
            "breadth": self.breadth.to_dict(),
            "novelty": self.novelty.to_dict(),
            "reliability": self.reliability.to_dict(),
            "coherence": self.coherence.to_dict(),
            "actionability": self.actionability.to_dict(),
            "overall": {
                "score": round(self.overall_score, 4),
                "grade": self.overall_grade,
                "percentage": self.overall_percentage,
                "reasoning": self.overall_reasoning,
            },
            "metadata": {
                "total_evidence_count": self.total_evidence_count,
                "processing_phase": self.processing_phase,
            },
        }

    def __str__(self) -> str:
        lines = [
            f"OVERALL: {self.overall_grade} ({self.overall_percentage})",
            f"  Depth:          {self.depth.letter_grade} ({self.depth.percentage})",
            f"  Breadth:        {self.breadth.letter_grade} ({self.breadth.percentage})",
            f"  Novelty:        {self.novelty.letter_grade} ({self.novelty.percentage})",
            f"  Reliability:    {self.reliability.letter_grade} ({self.reliability.percentage})",
            f"  Coherence:      {self.coherence.letter_grade} ({self.coherence.percentage})",
            f"  Actionability:  {self.actionability.letter_grade} ({self.actionability.percentage})",
        ]
        return "\n".join(lines)


class EvidenceGrader:
    """
    Multi-dimensional evidence grader.

    Replaces single-confidence scoring with six independent
    dimensions that each capture a different aspect of evidence quality.

    The grader receives processed signals from the neuronal reasoning
    system and produces a detailed grade that tells the user
    exactly how and why the evidence was evaluated.
    """

    # Dimension weights for overall score
    WEIGHTS: dict[str, float] = {
        "depth": 0.30,
        "breadth": 0.15,
        "novelty": 0.05,
        "reliability": 0.30,
        "coherence": 0.15,
        "actionability": 0.05,
    }

    def grade(
        self,
        evidence_signals: list[Any],
        credibility_signals: list[Any],
        temporal_signals: list[Any],
        causal_signals: list[Any],
        contradiction_signals: list[Any],
        integrated_confidence: float,
        processing_phase: str = "novice",
    ) -> EvidenceGrade:
        """
        Compute multi-dimensional grades from processed signals.
        """
        # ── DEPTH: How thoroughly was the evidence analyzed? ──
        depth = self._grade_depth(
            evidence_signals, credibility_signals, causal_signals
        )

        # ── BREADTH: How many perspectives were considered? ──
        breadth = self._grade_breadth(
            evidence_signals, credibility_signals, temporal_signals
        )

        # ── NOVELTY: How new/unique is this information? ──
        novelty = self._grade_novelty(temporal_signals, evidence_signals)

        # ── RELIABILITY: How trustworthy is the source? ──
        reliability = self._grade_reliability(
            credibility_signals, evidence_signals
        )

        # ── COHERENCE: How well do evidence items fit together? ──
        coherence = self._grade_coherence(
            causal_signals, contradiction_signals, evidence_signals
        )

        # ── ACTIONABILITY: How actionable is this information? ──
        actionability = self._grade_actionability(
            evidence_signals, contradiction_signals, integrated_confidence
        )

        # ── OVERALL: Weighted combination ──
        overall_score = (
            depth.score * self.WEIGHTS["depth"]
            + breadth.score * self.WEIGHTS["breadth"]
            + novelty.score * self.WEIGHTS["novelty"]
            + reliability.score * self.WEIGHTS["reliability"]
            + coherence.score * self.WEIGHTS["coherence"]
            + actionability.score * self.WEIGHTS["actionability"]
        )

        overall_reasoning = self._build_overall_reasoning(
            depth, breadth, novelty, reliability, coherence, actionability
        )

        grade = EvidenceGrade(
            depth=depth,
            breadth=breadth,
            novelty=novelty,
            reliability=reliability,
            coherence=coherence,
            actionability=actionability,
            overall_score=overall_score,
            overall_grade=score_to_grade(overall_score),
            overall_percentage=score_to_percentage(overall_score),
            overall_reasoning=overall_reasoning,
            total_evidence_count=len(evidence_signals),
            processing_phase=processing_phase,
            grade_dimensions=[
                depth, breadth, novelty, reliability, coherence, actionability,
            ],
        )
        logger.info(f"Graded: {grade.overall_grade} ({grade.overall_percentage}) "
                     f"depth={depth.letter_grade} breadth={breadth.letter_grade} "
                     f"reliability={reliability.letter_grade} coherence={coherence.letter_grade}")
        return grade

    def _grade_depth(
        self,
        evidence: list,
        credibility: list,
        causal: list,
    ) -> DimensionGrade:
        """Grade how thoroughly evidence was analyzed."""
        score = 0.3  # base

        # More evidence = deeper analysis
        if len(evidence) > 10:
            score += 0.3
        elif len(evidence) > 5:
            score += 0.2
        elif len(evidence) > 2:
            score += 0.1

        # Credibility assessment adds depth
        if credibility:
            score += 0.15

        # Causal analysis adds depth
        if causal:
            score += 0.15

        # Cross-referencing (multiple sources) adds depth
        sources = set()
        for e in evidence:
            if hasattr(e, 'data') and isinstance(e.data, dict):
                src = e.data.get("source", "")
                if src:
                    sources.add(src)
        if len(sources) > 2:
            score += 0.1
        elif len(sources) > 1:
            score += 0.05

        score = min(1.0, score)

        factors = []
        if len(evidence) > 5:
            factors.append(f"{len(evidence)} evidence items analyzed")
        if credibility:
            factors.append(f"{len(credibility)} credibility assessments")
        if causal:
            factors.append(f"{len(causal)} causal links found")
        if len(sources) > 1:
            factors.append(f"{len(sources)} distinct sources")

        return DimensionGrade(
            name="depth",
            score=score,
            letter_grade=score_to_grade(score),
            percentage=score_to_percentage(score),
            reasoning=f"Analyzed {len(evidence)} items with {len(credibility)} credibility checks and {len(causal)} causal links",
            evidence_count=len(evidence),
            factors=factors,
        )

    def _grade_breadth(
        self,
        evidence: list,
        credibility: list,
        temporal: list,
    ) -> DimensionGrade:
        """Grade how many perspectives were considered."""
        score = 0.3  # base

        # Source diversity
        sources = set()
        for e in evidence:
            if hasattr(e, 'data') and isinstance(e.data, dict):
                src = e.data.get("source", "")
                if src:
                    sources.add(src)
        source_diversity = min(1.0, len(sources) / 5.0)
        score += source_diversity * 0.3

        # Temporal coverage
        if temporal:
            score += 0.15

        # Evidence volume
        if len(evidence) > 8:
            score += 0.15
        elif len(evidence) > 4:
            score += 0.1

        score = min(1.0, score)

        factors = []
        if sources:
            factors.append(f"{len(sources)} distinct sources")
        if temporal:
            factors.append(f"{len(temporal)} temporal perspectives")
        factors.append(f"{len(evidence)} evidence items")

        return DimensionGrade(
            name="breadth",
            score=score,
            letter_grade=score_to_grade(score),
            percentage=score_to_percentage(score),
            reasoning=f"Covered {len(sources)} sources and {len(temporal)} temporal dimensions",
            evidence_count=len(evidence),
            factors=factors,
        )

    def _grade_novelty(
        self,
        temporal: list,
        evidence: list,
    ) -> DimensionGrade:
        """Grade how new/unique the information is."""
        score = 0.5  # neutral base

        # Recent evidence is more novel
        recent_count = 0
        for t in temporal:
            if hasattr(t, 'data') and isinstance(t.data, dict):
                dr = t.data.get("date_relevance", 0)
                if dr > 0.8:
                    recent_count += 1

        if recent_count > 0:
            score += min(0.3, recent_count * 0.1)

        # Some novelty from diverse evidence types
        if len(evidence) > 5:
            score += 0.1

        score = min(1.0, score)

        factors = []
        if recent_count > 0:
            factors.append(f"{recent_count} recent evidence items")
        if len(evidence) > 5:
            factors.append("diverse evidence base")

        return DimensionGrade(
            name="novelty",
            score=score,
            letter_grade=score_to_grade(score),
            percentage=score_to_percentage(score),
            reasoning=f"Found {recent_count} recent items among {len(evidence)} total",
            evidence_count=len(evidence),
            factors=factors,
        )

    def _grade_reliability(
        self,
        credibility: list,
        evidence: list,
    ) -> DimensionGrade:
        """Grade how trustworthy the source is."""
        score = 0.4  # neutral base

        if not credibility:
            return DimensionGrade(
                name="reliability",
                score=score,
                letter_grade=score_to_grade(score),
                percentage=score_to_percentage(score),
                reasoning="No credibility assessment performed",
                evidence_count=len(evidence),
                factors=["no credibility data"],
            )

        # Average credibility
        cred_scores = [c.confidence for c in credibility if hasattr(c, 'confidence')]
        if cred_scores:
            avg_cred = sum(cred_scores) / len(cred_scores)
            score = avg_cred

        # High-credibility sources boost score
        high_trust = sum(1 for c in credibility if c.confidence > 0.7)
        if high_trust > 0:
            score = min(1.0, score + high_trust * 0.05)

        # Source identification
        identified = 0
        for c in credibility:
            if hasattr(c, 'data') and isinstance(c.data, dict):
                src = c.data.get("source", "")
                if src and src != "unknown":
                    identified += 1
        if identified > 0:
            score = min(1.0, score + identified * 0.03)

        factors = []
        if cred_scores:
            factors.append(f"avg credibility: {sum(cred_scores)/len(cred_scores):.0%}")
        if high_trust > 0:
            factors.append(f"{high_trust} high-trust sources")
        if identified > 0:
            factors.append(f"{identified} identified sources")

        return DimensionGrade(
            name="reliability",
            score=score,
            letter_grade=score_to_grade(score),
            percentage=score_to_percentage(score),
            reasoning=f"Average credibility {score:.0%} across {len(credibility)} assessments",
            evidence_count=len(evidence),
            factors=factors,
        )

    def _grade_coherence(
        self,
        causal: list,
        contradictions: list,
        evidence: list,
    ) -> DimensionGrade:
        """Grade how well evidence items fit together."""
        score = 0.5  # neutral base

        # Causal links improve coherence
        if causal:
            avg_causal = sum(c.confidence for c in causal if hasattr(c, 'confidence')) / len(causal)
            score += avg_causal * 0.3

        # Contradictions reduce coherence
        if contradictions:
            contra_impact = len(contradictions) * 0.15
            score -= contra_impact

        # Consistent evidence (low contradiction rate) improves coherence
        if evidence and not contradictions:
            score += 0.15

        score = max(0.0, min(1.0, score))

        factors = []
        if causal:
            factors.append(f"{len(causal)} causal connections")
        if contradictions:
            factors.append(f"{len(contradictions)} contradictions detected")
        if evidence and not contradictions:
            factors.append("no contradictions found")

        reasoning = f"{len(causal)} causal links, {len(contradictions)} contradictions"
        return DimensionGrade(
            name="coherence",
            score=score,
            letter_grade=score_to_grade(score),
            percentage=score_to_percentage(score),
            reasoning=reasoning,
            evidence_count=len(evidence),
            factors=factors,
        )

    def _grade_actionability(
        self,
        evidence: list,
        contradictions: list,
        confidence: float,
    ) -> DimensionGrade:
        """Grade how actionable the information is."""
        score = 0.3  # base

        # Enough evidence to act on
        if len(evidence) >= 5:
            score += 0.3
        elif len(evidence) >= 3:
            score += 0.2
        elif len(evidence) >= 1:
            score += 0.1

        # Low contradiction rate
        if not contradictions:
            score += 0.2
        elif len(contradictions) <= 1:
            score += 0.1

        # High confidence enables action
        if confidence > 0.7:
            score += 0.2
        elif confidence > 0.5:
            score += 0.1

        score = min(1.0, score)

        factors = []
        if len(evidence) >= 5:
            factors.append("sufficient evidence volume")
        if not contradictions:
            factors.append("no contradictions")
        if confidence > 0.5:
            factors.append(f"confidence: {confidence:.0%}")

        return DimensionGrade(
            name="actionability",
            score=score,
            letter_grade=score_to_grade(score),
            percentage=score_to_percentage(score),
            reasoning=f"{len(evidence)} items, {len(contradictions)} contradictions, {confidence:.0%} confidence",
            evidence_count=len(evidence),
            factors=factors,
        )

    def _build_overall_reasoning(
        self,
        depth: DimensionGrade,
        breadth: DimensionGrade,
        novelty: DimensionGrade,
        reliability: DimensionGrade,
        coherence: DimensionGrade,
        actionability: DimensionGrade,
    ) -> str:
        """Build a human-readable reasoning for the overall grade."""
        strengths = []
        weaknesses = []

        dims = [
            ("depth", depth),
            ("breadth", breadth),
            ("novelty", novelty),
            ("reliability", reliability),
            ("coherence", coherence),
            ("actionability", actionability),
        ]

        for name, dim in dims:
            if dim.score >= 0.7:
                strengths.append(name)
            elif dim.score < 0.5:
                weaknesses.append(name)

        parts = []
        if strengths:
            parts.append(f"strengths: {', '.join(strengths)}")
        if weaknesses:
            parts.append(f"weaknesses: {', '.join(weaknesses)}")
        if not parts:
            parts.append("balanced performance across all dimensions")

        return "; ".join(parts)

    # ════════════════════════════════════════════════════════════════
    # PER-EVIDENCE GRADING + FEEDBACK LOOP
    # ════════════════════════════════════════════════════════════════

    def grade_per_evidence(
        self,
        evidence_signals: list[Any],
        credibility_signals: list[Any],
        decision_outcome: str = "unknown",
    ) -> list[dict[str, Any]]:
        """
        Grade each evidence item individually on depth, reliability, and coherence.

        Returns a list of per-evidence grade dicts with feedback signals
        that can be fed back to the plasticity system.
        """
        grades = []
        for i, sig in enumerate(evidence_signals):
            ev_data = sig.data if hasattr(sig, 'data') else {}
            text = ev_data.get("text", ev_data.get("evidence_text", ""))
            conf = sig.confidence if hasattr(sig, 'confidence') else 0.5

            # Individual depth: how much information this evidence carries
            depth_score = min(1.0, 0.3 + len(text) / 500.0 + conf * 0.3)

            # Individual reliability: based on confidence
            reliability_score = conf

            # Individual coherence: how well it fits with other evidence
            coherence_score = 0.5  # baseline
            if len(evidence_signals) > 1:
                # Check if text overlaps with other evidence
                other_texts = [
                    (s.data.get("text", s.data.get("evidence_text", ""))
                     if hasattr(s, 'data') else "")
                    for j, s in enumerate(evidence_signals) if j != i
                ]
                if any(text[:20] in ot for ot in other_texts if ot):
                    coherence_score = 0.8
                elif text and any(text.split()[k] in ot for ot in other_texts for k in range(min(3, len(text.split())))):
                    coherence_score = 0.65

            overall = (depth_score * 0.4 + reliability_score * 0.35 + coherence_score * 0.25)

            # Feedback signal: was this evidence useful for the final decision?
            useful = decision_outcome in ("approved", "proceed", "go", "increased_confidence")
            feedback_strength = overall * (1.2 if useful else 0.6)

            grades.append({
                "evidence_index": i,
                "text_preview": text[:80] if text else "",
                "depth_score": round(depth_score, 4),
                "reliability_score": round(reliability_score, 4),
                "coherence_score": round(coherence_score, 4),
                "overall_score": round(overall, 4),
                "overall_grade": score_to_grade(overall),
                "feedback_useful": useful,
                "feedback_strength": round(feedback_strength, 4),
            })

        return grades

    def compute_feedback_from_grades(
        self,
        per_evidence_grades: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Aggregate per-evidence grades into a feedback signal for plasticity.

        Returns a feedback dict with LTP/LTD signals that the plasticity
        system can use to strengthen or weaken synaptic connections.
        """
        if not per_evidence_grades:
            return {"ltp_strength": 0.0, "ltd_strength": 0.0, "avg_quality": 0.5}

        scores = [g["overall_score"] for g in per_evidence_grades]
        avg_quality = sum(scores) / len(scores)
        useful_count = sum(1 for g in per_evidence_grades if g["feedback_useful"])
        total = len(per_evidence_grades)

        # LTP strength: proportional to how many evidence items were useful
        ltp_strength = (useful_count / total) * avg_quality if total > 0 else 0.0

        # LTD strength: proportional to how many were NOT useful
        ltd_strength = ((total - useful_count) / total) * (1.0 - avg_quality) if total > 0 else 0.0

        return {
            "ltp_strength": round(ltp_strength, 4),
            "ltd_strength": round(ltd_strength, 4),
            "avg_quality": round(avg_quality, 4),
            "useful_evidence_count": useful_count,
            "total_evidence_count": total,
            "usefulness_ratio": round(useful_count / max(1, total), 4),
        }
