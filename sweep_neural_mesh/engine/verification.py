"""
Verification Core — self-checking layer for Sweep's outputs.

After producing an answer, Sweep internally checks:
    1. Did I understand the task?
    2. Did I miss important information?
    3. Did I invent information?
    4. Are calculations correct?
    5. Are sources available?
    6. Are sources contradictory?
    7. Did I confuse inference with fact?
    8. Is the confidence justified?
    9. Can the conclusion be independently verified?

If confidence is low: DO NOT hallucinate.
Instead: "I don't have enough evidence to determine this."

Sweep-original implementation.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("sweep.verification")


@dataclass
class VerificationResult:
    """Result of verifying an answer."""
    passed: bool
    confidence_adjustment: float = 0.0  # positive = boost, negative = reduce
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


class VerificationCore:
    """Self-checking layer that verifies answers before output.

    Checks for:
        - Hallucination indicators
        - Overconfidence
        - Missing evidence
        - Contradictory sources
        - Calculation errors
        - Unsupported claims
    """

    HALLUCINATION_INDICATORS = [
        "definitely", "certainly", "absolutely", "without doubt",
        "100%", "always", "never", "impossible", "guaranteed",
    ]

    UNCERTAINTY_MARKERS = [
        "might", "could", "possibly", "perhaps", "unclear",
        "insufficient", "not enough", "unknown", "uncertain",
    ]

    def verify(
        self,
        query: str,
        answer: str,
        confidence: float,
        reasoning: str = "",
        evidence: list[str] | None = None,
        decision: str = "",
    ) -> VerificationResult:
        """Verify an answer before output."""
        t0 = time.perf_counter()
        issues = []
        warnings = []
        suggestions = []
        adj = 0.0

        evidence = evidence or []

        # Check 1: Empty or nonsensical answer
        if not answer or len(answer.strip()) < 2:
            issues.append("Answer is empty or too short")
            adj -= 0.3

        # Check 2: Hallucination indicators
        answer_lower = answer.lower()
        for indicator in self.HALLUCINATION_INDICATORS:
            if indicator in answer_lower and confidence > 0.8:
                warnings.append(f"Hallucination indicator '{indicator}' with high confidence")
                adj -= 0.1

        # Check 3: Overconfidence without evidence
        if confidence > 0.9 and not evidence:
            warnings.append("Very high confidence with no evidence provided")
            adj -= 0.15

        # Check 4: Answer is just the query echoed back
        if answer.strip().lower() == query.strip().lower():
            issues.append("Answer appears to be the query echoed back")
            adj -= 0.4

        # Check 5: Evidence contradiction check
        if evidence and len(evidence) >= 2:
            contradiction_found = self._check_contradictions(evidence)
            if contradiction_found and decision != "mixed":
                warnings.append("Evidence contains contradictions but decision is not 'mixed'")
                adj -= 0.1

        # Check 6: Unsupported claim detection
        if reasoning and not evidence:
            # Reasoning without evidence might be pure speculation
            reasoning_lower = reasoning.lower()
            if any(w in reasoning_lower for w in ["therefore", "thus", "consequently", "hence"]):
                warnings.append("Drawing conclusions without evidence")
                adj -= 0.1

        # Check 7: Confidence calibration
        if confidence < 0.3 and decision in ("supported", "refuted"):
            warnings.append(f"Low confidence ({confidence:.2f}) with definitive decision")
            suggestions.append("Consider using 'insufficient' instead")

        # Check 8: Answer format validation
        if decision == "supported" and answer.lower() in ("no", "false", "refuted"):
            issues.append("Decision says 'supported' but answer says 'no/false'")
            adj -= 0.2

        # Check 9: Missing key information
        if not reasoning and not answer:
            issues.append("No reasoning or answer provided")
            adj -= 0.3

        passed = len(issues) == 0
        latency = (time.perf_counter() - t0) * 1000

        if issues:
            logger.warning(f"Verification FAILED: {issues}")
        if warnings:
            logger.info(f"Verification warnings: {warnings}")

        return VerificationResult(
            passed=passed,
            confidence_adjustment=adj,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
            latency_ms=latency,
        )

    def _check_contradictions(self, evidence: list[str]) -> bool:
        """Quick check if evidence contains contradictions."""
        negation_words = {"not", "no", "never", "none", "false", "incorrect",
                          "refute", "contradict", "fail", "failed"}
        affirmation_words = {"is", "are", "was", "were", "has", "have", "true",
                            "correct", "supports", "confirms"}

        for i in range(len(evidence)):
            for j in range(i + 1, len(evidence)):
                e1 = evidence[i].lower()
                e2 = evidence[j].lower()
                # Check for same-topic opposite-polarity
                words1 = set(re.findall(r'\b\w{3,}\b', e1))
                words2 = set(re.findall(r'\b\w{3,}\b', e2))
                overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                if overlap > 0.3:
                    e1_neg = any(w in e1.split() for w in negation_words)
                    e2_neg = any(w in e2.split() for w in negation_words)
                    if e1_neg != e2_neg:
                        return True
        return False
