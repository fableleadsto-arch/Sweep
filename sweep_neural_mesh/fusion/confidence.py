"""
ConfidenceEngine — evaluates and calibrates confidence across the Mesh.

The ConfidenceEngine answers: "How sure is the Mesh about its output?"

It aggregates confidence from individual nodes, applies calibration,
and produces a final confidence estimate with uncertainty bounds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConfidenceReport:
    """A structured confidence assessment."""
    score: float = 0.0
    uncertainty: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    node_confidences: dict[str, float] = field(default_factory=dict)
    calibration_applied: bool = False
    agreement_score: float = 0.0
    quality_label: str = "unknown"

    @property
    def quality_tier(self) -> str:
        if self.score >= 0.9:
            return "high"
        if self.score >= 0.7:
            return "medium"
        if self.score >= 0.4:
            return "low"
        return "very_low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "uncertainty": self.uncertainty,
            "quality_tier": self.quality_tier,
            "agreement": self.agreement_score,
            "nodes": self.node_confidences,
            "calibrated": self.calibration_applied,
        }


class ConfidenceEngine:
    """
    Aggregates and calibrates confidence across the Mesh.

    Supports:
    - Simple averaging
    - Geometric mean (penalizes low confidence more)
    - Confidence-weighted average
    - Agreement scoring between multiple nodes
    """

    def __init__(self, method: str = "weighted_average") -> None:
        self.method = method

    def evaluate(
        self,
        node_confidences: dict[str, float],
        agreement_scores: list[float] | None = None,
    ) -> ConfidenceReport:
        """Produce a ConfidenceReport from per-node confidence scores."""
        if not node_confidences:
            return ConfidenceReport(quality_label="no_data")

        confs = list(node_confidences.values())
        report = ConfidenceReport(
            node_confidences=dict(node_confidences),
            min_confidence=min(confs),
            max_confidence=max(confs),
        )

        # Aggregate
        if self.method == "simple_average":
            report.score = sum(confs) / len(confs)
        elif self.method == "geometric_mean":
            # Clamp to avoid log(0)
            clamped = [max(c, 1e-10) for c in confs]
            log_sum = sum(math.log(c) for c in clamped)
            report.score = math.exp(log_sum / len(clamped))
        elif self.method == "weighted_average":
            # Weight by confidence itself (higher confidence gets more say)
            total = sum(confs)
            if total > 0:
                report.score = sum(c * c for c in confs) / total
            else:
                report.score = 0.0
        else:
            report.score = sum(confs) / len(confs)

        # Uncertainty (std dev of confidences)
        mean = report.score
        variance = sum((c - mean) ** 2 for c in confs) / len(confs)
        report.uncertainty = math.sqrt(variance)

        # Agreement
        if agreement_scores:
            report.agreement_score = sum(agreement_scores) / len(agreement_scores)
        elif len(confs) > 1:
            # Self-agreement: how close are the confidences to each other
            spread = max(confs) - min(confs)
            report.agreement_score = 1.0 - spread

        report.quality_label = report.quality_tier
        return report

    def calibrate(self, raw_score: float, temperature: float = 1.0) -> float:
        """Temperature-scaled calibration of a single score."""
        if temperature == 1.0:
            return raw_score
        # Sigmoid-like scaling
        logit = math.log(max(raw_score, 1e-10) / max(1 - raw_score, 1e-10))
        scaled = 1.0 / (1.0 + math.exp(-logit / temperature))
        return scaled
