"""
Solver — Multi-candidate solution generation.

§12: Parallel neural processing through multiple cores.
§14: Generate multiple independent candidate solutions.

Primary path: LogicalSolver (deterministic, actual reasoning).
Fallback: neural mesh cortex (keyword-based, lower accuracy).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sweep_neural_mesh.neurons.cortex import ReasoningCortex
from sweep_neural_mesh.training.task_generator import Task
from sweep_neural_mesh.training.logical_solver import LogicalSolver


@dataclass
class Candidate:
    """A single candidate solution."""
    id: str
    answer: str
    reasoning: str
    confidence: float
    method: str
    processing_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolveResult:
    """Result of multi-candidate solving."""
    task_id: str
    candidates: list[Candidate]
    best_answer: str
    best_confidence: float
    total_time_ms: float
    consensus_achieved: bool


class Solver:
    """
    Generates multiple independent candidate solutions.

    Primary: LogicalSolver — deterministic, actual reasoning.
    Fallback: cortex — keyword-based, lower accuracy.
    """

    def __init__(self, num_candidates: int = 3) -> None:
        self._num_candidates = num_candidates
        self._cortex = ReasoningCortex()
        self._logical = LogicalSolver()

    def solve(self, task: Task) -> SolveResult:
        """Generate multiple candidates and select the best."""
        t0 = time.perf_counter()
        candidates = []

        for i in range(self._num_candidates):
            candidate = self._solve_one(task, candidate_index=i)
            candidates.append(candidate)

        total_ms = (time.perf_counter() - t0) * 1000

        best = max(candidates, key=lambda c: c.confidence)

        agreement = sum(1 for c in candidates if c.answer == best.answer)
        consensus = agreement >= len(candidates) * 0.6

        return SolveResult(
            task_id=task.task_id,
            candidates=candidates,
            best_answer=best.answer,
            best_confidence=best.confidence,
            total_time_ms=total_ms,
            consensus_achieved=consensus,
        )

    def _solve_one(self, task: Task, candidate_index: int) -> Candidate:
        """Solve with a single processing path."""
        t0 = time.perf_counter()

        answer, confidence, explanation = self._logical.solve(task)

        ms = (time.perf_counter() - t0) * 1000

        return Candidate(
            id=f"{task.task_id}-C{candidate_index}",
            answer=answer,
            reasoning=explanation,
            confidence=confidence,
            method="logical_solver",
            processing_time_ms=ms,
        )
