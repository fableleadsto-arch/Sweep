"""
Adversarial Critique — Actively attempts to disprove candidate answers.

§13: Critique stage must actively attempt to DISPROVE the candidate answer.
Searches for: invalid inference, missing premise, contradictory evidence,
unsupported assumption, logical leap, ambiguity, circular reasoning, overconfidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sweep_neural_mesh.neurons.cortex import ReasoningCortex
from sweep_neural_mesh.training.task_generator import Task
from sweep_neural_mesh.training.solver import Candidate


@dataclass
class CritiqueResult:
    """Result of adversarial critique."""
    candidate_id: str
    answer: str
    is_valid: bool
    issues_found: list[str]
    confidence_after: float
    critique_reasoning: str
    suggestion: str


class Critique:
    """
    Adversarial critique stage.

    §13: After producing an answer, activate critique stage.
    Searches specifically for:
    - invalid inference
    - missing premise
    - contradictory evidence
    - unsupported assumption
    - incorrect relationship
    - logical leap
    - ambiguity
    - hallucinated information
    - circular reasoning
    - overconfidence
    """

    CRITIQUE_PROMPTS = [
        "Check if this answer assumes facts not in evidence.",
        "Check if this answer contains a logical leap.",
        "Check if this answer contradicts any given information.",
        "Check if this answer is overconfident given the evidence.",
        "Check if this answer has a missing premise.",
    ]

    def __init__(self) -> None:
        self._cortex = ReasoningCortex()

    def critique(
        self,
        task: Task,
        candidate: Candidate,
    ) -> CritiqueResult:
        """Perform adversarial critique of a candidate answer."""
        issues = []
        total_confidence = 0.0

        for prompt in self.CRITIQUE_PROMPTS:
            critique_query = (
                f"Given this problem: {task.input[:500]}\n\n"
                f"Proposed answer: {candidate.answer}\n"
                f"Reasoning: {candidate.reasoning[:300]}\n\n"
                f"{prompt}\n"
                f"Answer YES if the issue exists, NO if not."
            )

            result = self._cortex.reason(
                query=critique_query,
                evidence=[task.input],
            )

            total_confidence += result.confidence
            if "support" in result.decision.lower():
                issue_name = prompt.replace("Check if this answer ", "").replace(".", "")
                issues.append(issue_name)

        avg_confidence = total_confidence / len(self.CRITIQUE_PROMPTS)
        is_valid = len(issues) == 0
        confidence_after = candidate.confidence * (0.8 if issues else 1.0)

        suggestion = ""
        if issues:
            suggestion = f"Found {len(issues)} issues: {'; '.join(issues[:3])}. Consider revising."

        return CritiqueResult(
            candidate_id=candidate.id,
            answer=candidate.answer,
            is_valid=is_valid,
            issues_found=issues,
            confidence_after=confidence_after,
            critique_reasoning=f"Checked {len(self.CRITIQUE_PROMPTS)} critique dimensions. Found {len(issues)} issues.",
            suggestion=suggestion,
        )
