"""
EvolutionEngine — mutates and evolves patterns based on performance.

Responsibilities:
  - Propose mutations for failing patterns (broaden, alternatives, fix).
  - Select best-performing variants.
  - Cross-pollinate successful patterns.
  - Track generation count and fitness history.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PatternMutation:
    """A proposed mutation to a pattern."""
    pattern_id: str
    original: str
    mutated: str
    reason: str
    expected_improvement: float


class EvolutionEngine:
    """Evolves patterns based on performance feedback.

    Strategies:
      - Broaden:      make pattern more general (high failure count).
      - Alternatives: add | alternatives (medium failure count).
      - Fix:          correct common regex issues (low failure count).
    """

    def __init__(self) -> None:
        self._mutations: list[PatternMutation] = []
        self._generation = 0
        self._fitness_history: list[float] = []

    def mutate_pattern(
        self,
        pattern: str,
        failure_count: int,
        context: str = "",
    ) -> PatternMutation:
        """Propose a mutation for a failing pattern."""
        if failure_count > 5:
            strategy = self._broaden_pattern
        elif failure_count > 2:
            strategy = self._add_alternatives
        else:
            strategy = self._fix_common_issues

        mutated = strategy(pattern, context)
        mutation = PatternMutation(
            pattern_id=f"mut_{self._generation}_{len(self._mutations)}",
            original=pattern,
            mutated=mutated,
            reason=f"Strategy: {strategy.__name__}",
            expected_improvement=0.1,
        )
        self._mutations.append(mutation)
        return mutation

    def select_best(self, variants: list[tuple[str, float]]) -> str:
        """Select the best-performing variant."""
        if not variants:
            return ""
        return max(variants, key=lambda x: x[1])[0]

    def cross_pollinate(self, pattern_a: str, pattern_b: str) -> str:
        """Combine elements from two successful patterns."""
        parts_a = pattern_a.split("|")
        parts_b = pattern_b.split("|")
        if len(parts_a) > 1 and len(parts_b) > 1:
            return f"{parts_a[0]}|{parts_b[-1]}"
        return pattern_a

    def record_fitness(self, fitness: float) -> None:
        self._fitness_history.append(fitness)
        self._generation += 1

    # ── Mutation strategies ─────────────────────────────────

    @staticmethod
    def _broaden_pattern(pattern: str, _ctx: str) -> str:
        broadened = pattern.replace("\\b", "")
        return re.sub(r"\{[^}]+\\}", ".*", broadened)

    @staticmethod
    def _add_alternatives(pattern: str, _ctx: str) -> str:
        alternatives = {
            "what is": "what (is|are|was|were)",
            "how does": "how (does|do|did)",
            "why do": "why (do|does|did)",
            "when was": "when (was|were|did)",
        }
        mutated = pattern
        for original, replacement in alternatives.items():
            if original in mutated.lower():
                mutated = re.sub(re.escape(original), replacement, mutated, flags=re.IGNORECASE)
                break
        return mutated

    @staticmethod
    def _fix_common_issues(pattern: str, _ctx: str) -> str:
        fixed = pattern.replace(".", "\\.")
        fixed = fixed.replace("+", "\\+")
        return fixed
