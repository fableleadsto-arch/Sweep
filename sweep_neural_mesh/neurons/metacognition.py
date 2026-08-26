"""
Metacognition — the Forebrain's self-monitoring system.

Metacognition is "thinking about thinking": the ability to monitor,
evaluate, and regulate one's own cognitive processes. This is the
prefrontal cortex's highest function — it's what makes us aware
of what we know, what we don't know, and how confident we should be.

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │                METACOGNITIVE SYSTEM                  │
    │                                                     │
    │  ┌─────────────────┐  ┌─────────────────────────┐  │
    │  │ Monitoring       │  │ Evaluation              │  │
    │  │ - Confidence     │  │ - Calibration check     │  │
    │  │   tracking       │  │ - Reasoning quality     │  │
    │  │ - Process        │  │ - Knowledge boundary    │  │
    │  │   awareness      │  │   detection             │  │
    │  └────────┬────────┘  └───────────┬─────────────┘  │
    │           ↓                       ↓                 │
    │  ┌─────────────────────────────────────────────┐   │
    │  │         REGULATION                           │   │
    │  │  - Adjust confidence based on calibration    │   │
    │  │  - Flag uncertainty for escalation           │   │
    │  │  - Detect when to seek more evidence          │   │
    │  │  - Monitor for reasoning biases              │   │
    │  └─────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────┘

Key metacognitive functions:

1. MONITORING: Track what we know and how we know it
   - Confidence tracking: are we consistently over/under-confident?
   - Process awareness: which centers contributed most?
   - Evidence tracking: how much evidence supports our conclusion?

2. EVALUATION: Assess the quality of our reasoning
   - Calibration: does our confidence match actual accuracy?
   - Coherence: does our reasoning form a logical chain?
   - Completeness: are we missing important perspectives?

3. REGULATION: Adjust our reasoning strategy
   - Confidence adjustment: correct over/under-confidence
   - Uncertainty flagging: know when we don't know
   - Escalation: know when to seek more evidence
   - Bias detection: catch systematic reasoning errors
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CalibrationRecord:
    """A record of a confidence prediction and its outcome."""
    predicted_confidence: float     # what we said our confidence was
    actual_outcome: float           # 0.0-1.0: how accurate was it
    timestamp: float = field(default_factory=time.time)


@dataclass
class UncertaintySignal:
    """A signal that we're uncertain about something."""
    uncertainty_type: str           # "knowledge_gap", "conflicting_evidence", etc.
    description: str
    severity: float                 # 0.0-1.0: how serious is this uncertainty
    suggested_action: str           # what to do about it
    timestamp: float = field(default_factory=time.time)


@dataclass
class MetacognitiveAssessment:
    """The complete metacognitive assessment of a reasoning pass."""
    # Monitoring
    confidence_history_length: int
    avg_confidence: float
    confidence_trend: str           # "improving", "declining", "stable"

    # Evaluation
    calibration_score: float        # 0.0-1.0: how well-calibrated are we
    reasoning_quality: float        # 0.0-1.0: estimated quality of reasoning
    knowledge_boundary: str         # what we know vs don't know

    # Regulation
    should_adjust_confidence: bool
    confidence_adjustment: float    # how much to adjust confidence
    uncertainty_signals: list[UncertaintySignal]
    escalation_recommended: bool
    escalation_reason: str

    # Overall metacognitive awareness
    awareness_score: float          # 0.0-1.0: how aware are we of our own state

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitoring": {
                "confidence_history_length": self.confidence_history_length,
                "avg_confidence": round(self.avg_confidence, 4),
                "confidence_trend": self.confidence_trend,
            },
            "evaluation": {
                "calibration_score": round(self.calibration_score, 4),
                "reasoning_quality": round(self.reasoning_quality, 4),
                "knowledge_boundary": self.knowledge_boundary,
            },
            "regulation": {
                "should_adjust_confidence": self.should_adjust_confidence,
                "confidence_adjustment": round(self.confidence_adjustment, 4),
                "uncertainty_count": len(self.uncertainty_signals),
                "escalation_recommended": self.escalation_recommended,
                "escalation_reason": self.escalation_reason,
            },
            "awareness_score": round(self.awareness_score, 4),
        }


class MetacognitiveSystem:
    """
    Monitor, evaluate, and regulate Sweep's own reasoning.

    Like the prefrontal cortex's metacognitive functions, this system:

    1. MONITORS reasoning quality in real-time
    2. EVALUATES whether our confidence matches actual accuracy
    3. REGULATES our reasoning strategy based on self-assessment
    4. DETECTS when we should seek more evidence or escalate

    This is what separates Sweep from simple confidence-score systems:
    we don't just produce a confidence number — we know HOW CALIBRATED
    that number is, and we adjust it accordingly.

    The metacognitive system learns over time:
    - If we're consistently over-confident, it will learn to dampen
    - If we're under-confident about certain topics, it will learn to boost
    - It tracks which types of reasoning produce reliable outcomes
    """

    def __init__(self) -> None:
        # Calibration history: predicted confidence → actual outcome
        self._calibration_history: list[CalibrationRecord] = []
        # Confidence history: tracking confidence over time
        self._confidence_history: list[float] = []
        # Uncertainty signals accumulated during reasoning
        self._uncertainty_signals: list[UncertaintySignal] = []
        # Knowledge gaps detected
        self._knowledge_gaps: list[str] = []
        # Reasoning quality estimates
        self._quality_estimates: list[float] = []

        # Calibration buckets: for computing calibration score
        self._calibration_buckets: dict[str, dict[str, float]] = {}
        # Learning rate for calibration
        self._calibration_lr = 0.1

    def monitor_reasoning(
        self,
        confidence: float,
        evidence_count: int,
        center_outputs: dict[str, int],
        contradictions: int,
        processing_phase: str,
    ) -> MetacognitiveAssessment:
        """
        Perform a full metacognitive assessment of the current reasoning.

        Called after processing centers complete but before final output.
        Monitors, evaluates, and generates regulatory signals.
        """
        # ── MONITORING ──
        self._confidence_history.append(confidence)
        avg_conf = self._compute_avg_confidence()
        trend = self._compute_confidence_trend()

        # ── EVALUATION ──
        calibration = self._compute_calibration_score()
        quality = self._estimate_reasoning_quality(
            evidence_count, center_outputs, contradictions, confidence
        )
        boundary = self._assess_knowledge_boundary(
            evidence_count, confidence, center_outputs
        )

        # ── REGULATION ──
        adjustment, should_adjust = self._compute_confidence_adjustment(
            confidence, calibration
        )

        # Detect uncertainties
        uncertainties = self._detect_uncertainties(
            evidence_count, contradictions, confidence, center_outputs
        )
        self._uncertainty_signals.extend(uncertainties)

        # Check if escalation is needed
        escalation, esc_reason = self._check_escalation_needed(
            confidence, evidence_count, contradictions, uncertainties
        )

        # ── AWARENESS ──
        awareness = self._compute_awareness_score(
            calibration, quality, len(uncertainties)
        )

        return MetacognitiveAssessment(
            confidence_history_length=len(self._confidence_history),
            avg_confidence=avg_conf,
            confidence_trend=trend,
            calibration_score=calibration,
            reasoning_quality=quality,
            knowledge_boundary=boundary,
            should_adjust_confidence=should_adjust,
            confidence_adjustment=adjustment,
            uncertainty_signals=uncertainties,
            escalation_recommended=escalation,
            escalation_reason=esc_reason,
            awareness_score=awareness,
        )

    def record_outcome(
        self,
        predicted_confidence: float,
        actual_outcome: float,
    ) -> None:
        """
        Record the actual outcome of a reasoning pass.

        This drives calibration learning: if our confidence was
        0.8 but the outcome was 0.4, we were over-confident.
        """
        record = CalibrationRecord(
            predicted_confidence=predicted_confidence,
            actual_outcome=actual_outcome,
        )
        self._calibration_history.append(record)

        # Update calibration buckets
        bucket = self._confidence_to_bucket(predicted_confidence)
        if bucket not in self._calibration_buckets:
            self._calibration_buckets[bucket] = {"predicted_sum": 0.0, "actual_sum": 0.0, "count": 0}
        self._calibration_buckets[bucket]["predicted_sum"] += predicted_confidence
        self._calibration_buckets[bucket]["actual_sum"] += actual_outcome
        self._calibration_buckets[bucket]["count"] += 1

        # Keep history bounded
        if len(self._calibration_history) > 500:
            self._calibration_history = self._calibration_history[-500:]

    def _compute_avg_confidence(self) -> float:
        """Compute average confidence over recent history."""
        recent = self._confidence_history[-20:]
        if not recent:
            return 0.5
        return sum(recent) / len(recent)

    def _compute_confidence_trend(self) -> str:
        """Determine if confidence is trending up, down, or stable."""
        if len(self._confidence_history) < 3:
            return "stable"

        recent = self._confidence_history[-10:]
        if len(recent) < 3:
            return "stable"

        # Simple linear trend
        first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)

        diff = second_half - first_half
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        return "stable"

    def _compute_calibration_score(self) -> float:
        """
        Compute how well-calibrated our confidence predictions are.

        Perfect calibration: when we say 70% confidence, we're right 70% of the time.
        Score of 1.0 = perfect calibration, 0.0 = completely miscalibrated.
        """
        if not self._calibration_history:
            return 0.5  # no data, assume neutral

        # Use calibration buckets
        total_error = 0.0
        bucket_count = 0

        for bucket, data in self._calibration_buckets.items():
            if data["count"] < 2:
                continue
            avg_predicted = data["predicted_sum"] / data["count"]
            avg_actual = data["actual_sum"] / data["count"]
            error = abs(avg_predicted - avg_actual)
            total_error += error
            bucket_count += 1

        if bucket_count == 0:
            return 0.5

        avg_error = total_error / bucket_count
        # Convert error to score: 0 error = 1.0, 1.0 error = 0.0
        return max(0.0, 1.0 - avg_error)

    def _estimate_reasoning_quality(
        self,
        evidence_count: int,
        center_outputs: dict[str, int],
        contradictions: int,
        confidence: float,
    ) -> float:
        """Estimate the quality of the current reasoning pass."""
        score = 0.5  # base

        # More evidence generally means better reasoning
        if evidence_count > 10:
            score += 0.15
        elif evidence_count > 5:
            score += 0.10
        elif evidence_count < 2:
            score -= 0.15

        # Multiple centers contributing means thorough analysis
        active_centers = sum(1 for v in center_outputs.values() if v > 0)
        if active_centers >= 4:
            score += 0.15
        elif active_centers >= 2:
            score += 0.08

        # Contradictions handled well (found and resolved)
        if contradictions > 0:
            score += 0.05  # finding contradictions is good
        if contradictions > 3:
            score -= 0.10  # too many contradictions = confusion

        # Confidence in reasonable range
        if 0.3 < confidence < 0.8:
            score += 0.05  # appropriately uncertain
        elif confidence > 0.9:
            score -= 0.05  # potentially over-confident

        return max(0.0, min(1.0, score))

    def _assess_knowledge_boundary(
        self,
        evidence_count: int,
        confidence: float,
        center_outputs: dict[str, int],
    ) -> str:
        """Assess what we know vs what we don't know."""
        if evidence_count < 2:
            return "severe knowledge gap: very limited evidence"
        if evidence_count < 5:
            return "moderate knowledge gap: some evidence available"
        if confidence < 0.3:
            return "significant uncertainty despite evidence"
        if center_outputs.get("contradiction_detector", 0) > 2:
            return "conflicting evidence present"
        return "adequate knowledge base for this query"

    def _compute_confidence_adjustment(
        self,
        current_confidence: float,
        calibration_score: float,
    ) -> tuple[float, bool]:
        """
        Compute how much to adjust confidence based on calibration.

        If we're over-confident (calibration shows we predict higher
        than actual outcomes), reduce confidence. Vice versa.
        """
        if not self._calibration_history:
            return 0.0, False

        # Recent predictions vs outcomes
        recent = self._calibration_history[-10:]
        avg_predicted = sum(r.predicted_confidence for r in recent) / len(recent)
        avg_actual = sum(r.actual_outcome for r in recent) / len(recent)

        # Compute miscalibration direction
        miscalibration = avg_predicted - avg_actual

        # Only adjust if miscalibration is significant
        if abs(miscalibration) < 0.1:
            return 0.0, False

        # Adjustment: dampen confidence in the direction of miscalibration
        adjustment = -miscalibration * self._calibration_lr
        adjustment = max(-0.3, min(0.3, adjustment))  # cap adjustment

        return adjustment, True

    def _detect_uncertainties(
        self,
        evidence_count: int,
        contradictions: int,
        confidence: float,
        center_outputs: dict[str, int],
    ) -> list[UncertaintySignal]:
        """Detect specific uncertainty signals during reasoning."""
        signals = []

        # Knowledge gap
        if evidence_count < 3:
            signals.append(UncertaintySignal(
                uncertainty_type="knowledge_gap",
                description=f"Only {evidence_count} evidence items available",
                severity=1.0 - (evidence_count / 3.0),
                suggested_action="seek additional evidence sources",
            ))

        # Contradictory evidence
        if contradictions > 0:
            signals.append(UncertaintySignal(
                uncertainty_type="conflicting_evidence",
                description=f"{contradictions} contradictions detected between evidence items",
                severity=min(1.0, contradictions * 0.3),
                suggested_action="investigate contradictions before deciding",
            ))

        # Low credibility
        cred_count = center_outputs.get("credibility_assessor", 0)
        if evidence_count > 3 and cred_count == 0:
            signals.append(UncertaintySignal(
                uncertainty_type="unverified_sources",
                description="Evidence not assessed for credibility",
                severity=0.4,
                suggested_action="run credibility assessment on sources",
            ))

        # Over-confidence warning
        if confidence > 0.9 and evidence_count < 10:
            signals.append(UncertaintySignal(
                uncertainty_type="potential_overconfidence",
                description=f"High confidence ({confidence:.0%}) with limited evidence ({evidence_count} items)",
                severity=0.5,
                suggested_action="verify confidence calibration",
            ))

        # Low confidence
        if confidence < 0.2:
            signals.append(UncertaintySignal(
                uncertainty_type="low_confidence",
                description=f"Very low confidence ({confidence:.0%}) in current assessment",
                severity=0.7,
                suggested_action="gather more evidence or accept uncertainty",
            ))

        return signals

    def _check_escalation_needed(
        self,
        confidence: float,
        evidence_count: int,
        contradictions: int,
        uncertainties: list[UncertaintySignal],
    ) -> tuple[bool, str]:
        """Check if escalation to more processing is needed."""
        # High uncertainty signals → escalate
        high_severity = [u for u in uncertainties if u.severity > 0.5]
        if high_severity:
            return True, f"{len(high_severity)} high-severity uncertainties detected"

        # Many contradictions → escalate
        if contradictions > 2:
            return True, f"{contradictions} contradictions need resolution"

        # Very low confidence with limited evidence → escalate
        if confidence < 0.3 and evidence_count < 5:
            return True, "low confidence with limited evidence"

        return False, ""

    def _compute_awareness_score(
        self,
        calibration: float,
        quality: float,
        uncertainty_count: int,
    ) -> float:
        """
        Compute overall metacognitive awareness score.

        High awareness = we know how well we're doing and what we don't know.
        """
        score = 0.3  # base awareness

        # Good calibration = high awareness
        score += calibration * 0.3

        # Good reasoning quality = high awareness
        score += quality * 0.2

        # Detecting uncertainties = high awareness (it's good to know what you don't know)
        if uncertainty_count > 0:
            score += min(0.2, uncertainty_count * 0.05)

        return max(0.0, min(1.0, score))

    def _confidence_to_bucket(self, confidence: float) -> str:
        """Map confidence to a bucket for calibration tracking."""
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "very_low"

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "calibration_history_size": len(self._calibration_history),
            "confidence_history_size": len(self._confidence_history),
            "uncertainty_signals_total": len(self._uncertainty_signals),
            "calibration_score": self._compute_calibration_score(),
            "avg_confidence": self._compute_avg_confidence(),
            "confidence_trend": self._compute_confidence_trend(),
            "calibration_buckets": {
                k: v["count"] for k, v in self._calibration_buckets.items()
            },
        }
