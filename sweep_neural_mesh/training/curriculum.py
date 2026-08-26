"""
Curriculum Learning — Progressive difficulty with spaced revisit.

§18: Only increase difficulty when mastery threshold reached.
§19: Previously mastered domains periodically return.
§20: Catastrophic forgetting regression suite.
§28: Training/validation/hidden test/generalization sets.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from sweep_neural_mesh.training.domains import ExpertiseTracker, DEFAULT_DOMAINS


@dataclass
class CurriculumState:
    """Current state of curriculum learning."""
    iteration: int = 0
    current_domain: str = ""
    current_level: int = 1
    domains_completed: dict[str, int] = field(default_factory=dict)
    last_revisit: dict[str, int] = field(default_factory=dict)
    revisit_interval: int = 5
    total_tasks_attempted: int = 0
    total_tasks_correct: int = 0


class CurriculumManager:
    """
    Manages curriculum learning progression.

    §18: Progressive difficulty with mastery threshold.
    §19: Spaced revisit of previously mastered domains.
    §20: Regression suite after every training update.
    """

    def __init__(
        self,
        expertise: ExpertiseTracker,
        mastery_threshold: float = 0.90,
        revisit_interval: int = 5,
        regression_suite_size: int = 20,
    ) -> None:
        self._expertise = expertise
        self._mastery_threshold = mastery_threshold
        self._revisit_interval = revisit_interval
        self._regression_size = regression_suite_size
        self._state = CurriculumState()
        self._regression_suite: list[dict[str, Any]] = []
        self._rng = random.Random(42)

    def next_domain(self) -> tuple[str, int]:
        """
        Select the next domain and difficulty level.

        §18: Increase difficulty when mastery reached.
        §19: Periodically revisit mastered domains.
        """
        self._state.iteration += 1

        if self._state.iteration % self._revisit_interval == 0 and self._state.domains_completed:
            revisit = self._rng.choice(list(self._state.domains_completed.keys()))
            level = self._state.domains_completed[revisit]
            self._state.last_revisit[revisit] = self._state.iteration
            return revisit, level

        weakest = self._expertise.get_weakest_domains(1)
        if weakest:
            domain = weakest[0]
        else:
            domain = self._rng.choice(DEFAULT_DOMAINS)

        score = self._expertise.get_score(domain)
        level = score.current_level

        if score.is_mastered and level < 6:
            level = min(level + 1, 6)
            self._state.domains_completed[domain] = level

        self._state.current_domain = domain
        self._state.current_level = level
        return domain, level

    def record_outcome(self, correct: bool) -> None:
        self._state.total_tasks_attempted += 1
        if correct:
            self._state.total_tasks_correct += 1

    def build_regression_suite(self, tasks: list[dict[str, Any]]) -> None:
        """§20: Build a permanent regression suite."""
        self._regression_suite = tasks[:self._regression_size]

    def run_regression(self, solve_func) -> dict[str, Any]:
        """
        §20: After every training update, run regression suite.
        Compare with previous version.
        """
        if not self._regression_suite:
            return {"status": "no_regression_suite", "regression_detected": False}

        results = []
        for task in self._regression_suite:
            pred = solve_func(task)
            correct = pred == task.get("expected_output", "")
            results.append({"task_id": task.get("task_id", ""), "correct": correct})

        correct_count = sum(1 for r in results if r["correct"])
        accuracy = correct_count / max(1, len(results))

        return {
            "status": "completed",
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(results),
            "regression_detected": accuracy < self._mastery_threshold,
        }

    @property
    def state(self) -> CurriculumState:
        return self._state

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self._state.iteration,
            "current_domain": self._state.current_domain,
            "current_level": self._state.current_level,
            "domains_completed": self._state.domains_completed,
            "total_attempted": self._state.total_tasks_attempted,
            "total_correct": self._state.total_tasks_correct,
            "overall_accuracy": self._state.total_tasks_correct / max(1, self._state.total_tasks_attempted),
            "regression_suite_size": len(self._regression_suite),
        }
