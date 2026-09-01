"""
SelfEvolutionCoordinator — orchestrates all self-evolution capabilities.

Signal flow:
    Input → Learn → Evolve → Acquire → Optimize → Output

Responsibilities:
  1. Record all interactions for learning.
  2. Evolve patterns based on failures.
  3. Acquire new knowledge from successes/failures.
  4. Optimise based on performance metrics.
"""
from __future__ import annotations

from typing import Any

from .learning import LearningModule, LearningEvent
from .engine import EvolutionEngine
from .knowledge import KnowledgeAcquisition
from .tracker import PerformanceTracker


class SelfEvolutionCoordinator:
    """Coordinates learning, evolution, knowledge acquisition, and performance tracking.

    Usage::

        coord = SelfEvolutionCoordinator()
        result = coord.process_feedback(
            query="What is X?",
            expected="Y",
            actual="Z",
            confidence=0.8,
            source="factual",
        )
        print(result)  # {'correct': False, 'learned': True, ...}
    """

    def __init__(self) -> None:
        self._learning = LearningModule()
        self._evolution = EvolutionEngine()
        self._acquisition = KnowledgeAcquisition()
        self._performance = PerformanceTracker()

    def process_feedback(
        self,
        query: str,
        expected: str,
        actual: str,
        confidence: float,
        source: str,
    ) -> dict[str, Any]:
        """Process feedback from an interaction and update all modules."""
        correct = expected.lower().strip() == actual.lower().strip()

        # Record learning event
        event = LearningEvent(
            query=query, expected=expected, actual=actual,
            correct=correct, confidence=confidence, source=source,
        )
        self._learning.record_event(event)

        # Record performance metrics
        self._performance.record_metric("accuracy", 1.0 if correct else 0.0)
        self._performance.record_core_performance(source, 1.0 if correct else 0.0, 0.0)
        self._performance.record_calibration(confidence, correct)

        evolved = False
        if not correct:
            # Learn from failure
            self._learning._learn_from_failure(event)

            # Evolve failing patterns if failure rate is high
            failure_rate = self._learning.get_failure_rate(source)
            if failure_rate > 0.3:
                mutation = self._evolution.mutate_pattern(
                    pattern=query,
                    failure_count=int(failure_rate * 100),
                    context=f"Expected: {expected}, Got: {actual}",
                )
                self._acquisition.acquire(
                    pattern=mutation.mutated,
                    answer=expected,
                    source="self_discovery",
                    confidence=0.8,
                )
                evolved = True

        return {
            "correct": correct,
            "learned": not correct,
            "evolved": evolved,
            "stats": self._learning.get_stats(),
        }

    def get_adaptive_patterns(self) -> list[tuple[str, str, float]]:
        """Get all learned/evolved patterns for integration into cores."""
        patterns = list(self._learning.get_learned_facts())
        patterns.extend(self._acquisition.get_knowledge())
        return patterns

    def get_optimization_suggestions(self) -> list[dict[str, Any]]:
        return self._performance.suggest_optimizations()

    def get_full_stats(self) -> dict[str, Any]:
        return {
            "learning": self._learning.get_stats(),
            "evolution": {
                "generation": self._evolution._generation,
                "mutations": len(self._evolution._mutations),
            },
            "acquisition": self._acquisition.get_stats(),
            "performance": self._performance.get_stats(),
        }

    def save_state(self) -> None:
        self._learning._save_state()

    def load_state(self) -> None:
        self._learning._load_state()
