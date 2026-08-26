"""
VerificationEngine — cross-validates outputs from independent models.

When multiple models can solve the same problem, the VerificationEngine
checks whether they agree. Agreement increases confidence; disagreement
triggers investigation.

    Model A → output_a
    Model B → output_b
              ↓
    VerificationEngine
              ↓
    VerificationResult { agreement, confidence, recommended_action }
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VerificationAction(Enum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"
    ESCALATE = "escalate"


@dataclass
class VerificationResult:
    """Result of cross-validating two or more model outputs."""
    agreement: float = 0.0
    confidence: float = 0.0
    disagreement: float = 0.0
    uncertainty: float = 0.0
    model_sources: list[str] = field(default_factory=list)
    recommended_action: VerificationAction = VerificationAction.ACCEPT
    details: dict[str, Any] = field(default_factory=dict)


class VerificationEngine:
    """
    Cross-validates outputs from independent models.

    Metrics supported:
    - cosine similarity (for embeddings)
    - classification agreement (for discrete outputs)
    - numerical closeness (for regression)
    - embedding distance (for vectors)
    """

    def __init__(self, agreement_threshold: float = 0.7) -> None:
        self.agreement_threshold = agreement_threshold

    def verify(
        self,
        outputs: list[Any],
        sources: list[str] | None = None,
        metric: str = "auto",
    ) -> VerificationResult:
        """Cross-validate outputs from multiple models."""
        if not outputs:
            return VerificationResult(model_sources=sources or [])
        if len(outputs) == 1:
            return VerificationResult(
                agreement=1.0,
                confidence=1.0,
                model_sources=sources or [],
                recommended_action=VerificationAction.ACCEPT,
            )

        sources = sources or [f"model_{i}" for i in range(len(outputs))]
        result = VerificationResult(model_sources=sources)

        if metric == "auto":
            metric = self._detect_metric(outputs[0])

        if metric == "classification":
            result = self._verify_classification(outputs, sources)
        elif metric == "cosine":
            result = self._verify_cosine(outputs, sources)
        elif metric == "numerical":
            result = self._verify_numerical(outputs, sources)
        else:
            result = self._verify_exact(outputs, sources)

        # Decision
        if result.agreement >= self.agreement_threshold:
            result.recommended_action = VerificationAction.ACCEPT
        elif result.agreement >= self.agreement_threshold * 0.5:
            result.recommended_action = VerificationAction.REVIEW
        else:
            result.recommended_action = VerificationAction.REJECT

        return result

    def _detect_metric(self, sample: Any) -> str:
        if isinstance(sample, (int, str)):
            return "classification"
        if isinstance(sample, list) and all(isinstance(x, (int, float)) for x in sample):
            if len(sample) > 10:
                return "cosine"
            return "numerical"
        return "exact"

    def _verify_classification(
        self, outputs: list[Any], sources: list[str]
    ) -> VerificationResult:
        """Check if discrete outputs agree."""
        first = outputs[0]
        matches = sum(1 for o in outputs if o == first)
        agreement = matches / len(outputs)
        return VerificationResult(
            agreement=agreement,
            confidence=agreement,
            disagreement=1.0 - agreement,
            model_sources=sources,
            details={"first_output": first, "matches": matches, "total": len(outputs)},
        )

    def _verify_cosine(
        self, outputs: list[Any], sources: list[str]
    ) -> VerificationResult:
        """Compute pairwise cosine similarity of vector outputs."""
        if len(outputs) < 2:
            return VerificationResult(agreement=1.0, model_sources=sources)

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        sims = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                if isinstance(outputs[i], list) and isinstance(outputs[j], list):
                    sims.append(cosine(outputs[i], outputs[j]))

        avg_sim = sum(sims) / len(sims) if sims else 0.0
        return VerificationResult(
            agreement=avg_sim,
            confidence=avg_sim,
            disagreement=1.0 - avg_sim,
            model_sources=sources,
            details={"pairwise_similarities": sims},
        )

    def _verify_numerical(
        self, outputs: list[Any], sources: list[str]
    ) -> VerificationResult:
        """Check numerical closeness."""
        nums = [float(o) for o in outputs if isinstance(o, (int, float))]
        if not nums:
            return VerificationResult(agreement=0.0, model_sources=sources)
        mean = sum(nums) / len(nums)
        spread = max(nums) - min(nums)
        max_range = max(abs(mean), 1.0)
        agreement = 1.0 - min(spread / max_range, 1.0)
        return VerificationResult(
            agreement=agreement,
            confidence=agreement,
            disagreement=spread / max_range,
            model_sources=sources,
            details={"mean": mean, "spread": spread, "values": nums},
        )

    def _verify_exact(
        self, outputs: list[Any], sources: list[str]
    ) -> VerificationResult:
        """Exact match check."""
        first = outputs[0]
        matches = sum(1 for o in outputs if o == first)
        agreement = matches / len(outputs)
        return VerificationResult(
            agreement=agreement,
            confidence=agreement,
            disagreement=1.0 - agreement,
            model_sources=sources,
        )
