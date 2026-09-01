"""
LearningModule — tracks interactions and learns from feedback.

Responsibilities:
  - Record all query/answer interactions with outcomes.
  - Identify patterns that frequently fail.
  - Suggest improvements based on failure analysis.
  - Maintain a growing knowledge base from interactions.
  - Persist state to disk for cross-session learning.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LearningEvent:
    """A single learning event from an interaction."""
    query: str
    expected: str
    actual: str
    correct: bool
    confidence: float
    source: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class LearningModule:
    """Tracks successes/failures and learns from feedback.

    Features:
      - Records all interactions with outcomes.
      - Identifies patterns that frequently fail.
      - Learns new facts from failures.
      - Persists state to disk.
    """

    def __init__(self, memory_size: int = 10_000) -> None:
        self._events: list[LearningEvent] = []
        self._memory_size = memory_size
        self._failure_patterns: dict[str, int] = {}
        self._success_patterns: dict[str, int] = {}
        self._learned_facts: list[tuple[str, str, float]] = []
        self._load_state()

    def record_event(self, event: LearningEvent) -> None:
        """Record a learning event and update internal statistics."""
        self._events.append(event)

        bucket = self._success_patterns if event.correct else self._failure_patterns
        bucket[event.source] = bucket.get(event.source, 0) + 1

        if not event.correct and event.confidence > 0.7:
            self._learn_from_failure(event)

        if len(self._events) > self._memory_size:
            self._events = self._events[-self._memory_size:]

        self._save_state()

    def _learn_from_failure(self, event: LearningEvent) -> None:
        """Extract a new fact from a high-confidence failure."""
        words = re.findall(r"\b\w{3,}\b", event.query.lower())
        if len(words) >= 2:
            pattern = ".*".join(words[:3])
            self._learned_facts.append((pattern, event.expected, 0.8))

    def get_failure_rate(self, source: str) -> float:
        s = self._success_patterns.get(source, 0)
        f = self._failure_patterns.get(source, 0)
        return f / (s + f) if (s + f) > 0 else 0.0

    def get_worst_performing(self, top_k: int = 5) -> list[tuple[str, float]]:
        sources = set(self._success_patterns) | set(self._failure_patterns)
        rates = [(s, self.get_failure_rate(s)) for s in sources]
        rates.sort(key=lambda x: x[1], reverse=True)
        return rates[:top_k]

    def get_learned_facts(self) -> list[tuple[str, str, float]]:
        return list(self._learned_facts)

    def get_stats(self) -> dict[str, Any]:
        total = len(self._events)
        correct = sum(1 for e in self._events if e.correct)
        return {
            "total_events": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
            "learned_facts": len(self._learned_facts),
            "failure_patterns": len(self._failure_patterns),
            "success_patterns": len(self._success_patterns),
        }

    # ── Persistence ─────────────────────────────────────────

    def _state_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), ".learning_state.json")

    def _load_state(self) -> None:
        path = self._state_path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    state = json.load(f)
                self._learned_facts = [tuple(x) for x in state.get("learned_facts", [])]
                self._failure_patterns = state.get("failure_patterns", {})
                self._success_patterns = state.get("success_patterns", {})
            except Exception:
                pass

    def _save_state(self) -> None:
        try:
            state = {
                "learned_facts": [list(x) for x in self._learned_facts[-1000:]],
                "failure_patterns": self._failure_patterns,
                "success_patterns": self._success_patterns,
            }
            with open(self._state_path(), "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass
