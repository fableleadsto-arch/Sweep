"""
Forgetting Curve — Ebbinghaus-inspired memory decay with spaced repetition.

Human memory doesn't decay linearly. It follows an exponential curve:
    R = e^(-t/S)

Where:
    R = retention (0.0–1.0)
    t = time since last review
    S = stability (strength of memory)

Key properties:
    - New memories decay fast (S is small)
    - Reviewed memories decay slower (S increases with each review)
    - Spaced repetition optimizes review timing
    - Emotional memories decay slower (amygdala boost)

Architecture:

    Memory Item
        ↓
    ┌─────────────────────────────────────┐
    │  FORGETTING CURVE                   │
    │                                     │
    │  Initial encoding: S = 1.0          │
    │  After 1st review: S = 3.0          │
    │  After 2nd review: S = 9.0          │
    │  After nth review: S = 3^n          │
    │                                     │
    │  Retention at time t:               │
    │  R(t) = e^(-t/S)                    │
    │                                     │
    │  Review when R(t) < threshold:      │
    │  optimal_review_interval = S * ln(2)│
    └─────────────────────────────────────┘
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryTrace:
    """A memory with forgetting curve parameters."""
    memory_id: str
    content: str                   # what is remembered
    stability: float = 1.0         # S parameter (grows with reviews)
    retention: float = 1.0         # R parameter (decays with time)
    created_at: float = field(default_factory=time.time)
    last_reviewed: float = field(default_factory=time.time)
    review_count: int = 0          # number of times reviewed
    emotional_boost: float = 0.0   # amygdala boost (0.0–1.0)
    importance: float = 0.5        # inherent importance (0.0–1.0)

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600

    @property
    def hours_since_review(self) -> float:
        return (time.time() - self.last_reviewed) / 3600

    @property
    def optimal_review_hours(self) -> float:
        """When should this memory be reviewed next?"""
        return self.stability * math.log(2)

    @property
    def is_forgotten(self) -> bool:
        """Has this memory been forgotten (retention below threshold)?"""
        return self.retention < 0.1

    @property
    def needs_review(self) -> bool:
        """Does this memory need review?"""
        return self.retention < 0.5


class ForgettingCurve:
    """
    Ebbinghaus-inspired memory decay with spaced repetition.

    Implements the exponential forgetting curve:
        R(t) = e^(-t/S)

    And spaced repetition:
        - Each review increases stability (S)
        - Optimal review interval = S * ln(2)
        - Emotional memories decay slower (amygdala boost)

    This replaces the simple linear decay in the existing system
    with a biologically accurate model.

    Key behaviors:
    - New memories: S=1.0, half-life = 0.69 hours (~42 minutes)
    - After 1 review: S=3.0, half-life = 2.08 hours
    - After 3 reviews: S=27.0, half-life = 18.7 hours
    - After 5 reviews: S=243.0, half-life = 6.9 days
    - With emotional boost (0.5): S doubled, all intervals doubled
    """

    def __init__(self) -> None:
        self._traces: dict[str, MemoryTrace] = {}
        self._next_id = 0
        # Configuration
        self._stability_growth_rate = 3.0   # multiply S by this per review
        self._min_stability = 0.5           # minimum stability
        self._max_stability = 1000.0        # cap stability
        self._review_threshold = 0.5        # review when retention drops below this
        self._forget_threshold = 0.1        # forget when retention drops below this
        self._max_memories = 2000

    def encode(
        self,
        content: str,
        importance: float = 0.5,
        emotional_boost: float = 0.0,
    ) -> MemoryTrace:
        """
        Encode a new memory into the forgetting curve system.

        Args:
            content: what to remember
            importance: inherent importance (0.0–1.0)
            emotional_boost: amygdala boost (0.0–1.0), slows decay
        """
        self._next_id += 1
        trace = MemoryTrace(
            memory_id=f"mem_{self._next_id}",
            content=content[:500],
            stability=max(self._min_stability, 1.0 + emotional_boost * 2.0),
            retention=1.0,
            emotional_boost=emotional_boost,
            importance=importance,
        )
        self._traces[trace.memory_id] = trace
        self._enforce_capacity()
        return trace

    def review(self, memory_id: str) -> MemoryTrace | None:
        """
        Review a memory, increasing its stability.

        Like re-reading flashcards: each review makes the memory
        more resistant to forgetting.
        """
        trace = self._traces.get(memory_id)
        if not trace:
            return None

        # Increase stability (spaced repetition multiplier)
        effective_growth = self._stability_growth_rate * (1.0 + trace.emotional_boost * 0.5)
        trace.stability = min(
            self._max_stability,
            trace.stability * effective_growth,
        )

        # Reset retention to 1.0 (just reviewed)
        trace.retention = 1.0
        trace.review_count += 1
        trace.last_reviewed = time.time()

        return trace

    def update_retention(self, memory_id: str) -> MemoryTrace | None:
        """
        Update a memory's retention based on time since last review.

        R(t) = e^(-t/S) where t = hours since review, S = stability
        """
        trace = self._traces.get(memory_id)
        if not trace:
            return None

        hours = trace.hours_since_review
        # Ebbinghaus formula with emotional boost
        effective_stability = trace.stability * (1.0 + trace.emotional_boost)
        trace.retention = math.exp(-hours / effective_stability)

        return trace

    def update_all_retention(self) -> int:
        """Update retention for all memories. Returns count of forgotten memories."""
        forgotten = 0
        for trace in self._traces.values():
            self.update_retention(trace.memory_id)
            if trace.is_forgotten:
                forgotten += 1
        return forgotten

    def get_memories_needing_review(self, top_k: int = 10) -> list[MemoryTrace]:
        """
        Get memories that need review, sorted by urgency.

        Urgency = how close to forgetting they are × importance.
        """
        needing_review: list[tuple[float, MemoryTrace]] = []
        for trace in self._traces.values():
            self.update_retention(trace.memory_id)
            if trace.needs_review:
                # Urgency: lower retention = more urgent, higher importance = more urgent
                urgency = (1.0 - trace.retention) * trace.importance
                needing_review.append((urgency, trace))

        needing_review.sort(key=lambda x: x[0], reverse=True)
        return [trace for _, trace in needing_review[:top_k]]

    def get_forgotten_memories(self) -> list[MemoryTrace]:
        """Get memories that have been forgotten."""
        forgotten = []
        for trace in self._traces.values():
            self.update_retention(trace.memory_id)
            if trace.is_forgotten:
                forgotten.append(trace)
        return forgotten

    def strengthen(self, memory_id: str, amount: float = 0.1) -> bool:
        """Manually strengthen a memory (e.g., from emotional encoding)."""
        trace = self._traces.get(memory_id)
        if not trace:
            return False
        trace.stability = min(self._max_stability, trace.stability + amount)
        trace.retention = min(1.0, trace.retention + amount)
        return True

    def prune_forgotten(self) -> int:
        """Remove memories that have been forgotten. Returns count removed."""
        forgotten_ids = [
            mid for mid, trace in self._traces.items()
            if trace.is_forgotten
        ]
        for mid in forgotten_ids:
            del self._traces[mid]
        return len(forgotten_ids)

    def _enforce_capacity(self) -> None:
        """Remove lowest-importance memories if over capacity."""
        if len(self._traces) <= self._max_memories:
            return
        # Sort by importance × retention (keep most valuable)
        sorted_traces = sorted(
            self._traces.items(),
            key=lambda x: x[1].importance * x[1].retention,
        )
        # Remove lowest
        to_remove = len(self._traces) - self._max_memories
        for mid, _ in sorted_traces[:to_remove]:
            del self._traces[mid]

    @property
    def memory_count(self) -> int:
        return len(self._traces)

    @property
    def stats(self) -> dict[str, Any]:
        if not self._traces:
            return {"total_memories": 0}
        retentions = [t.retention for t in self._traces.values()]
        stabilities = [t.stability for t in self._traces.values()]
        needing_review = sum(1 for t in self._traces.values() if t.needs_review)
        forgotten = sum(1 for t in self._traces.values() if t.is_forgotten)
        return {
            "total_memories": len(self._traces),
            "avg_retention": round(sum(retentions) / len(retentions), 4),
            "avg_stability": round(sum(stabilities) / len(stabilities), 4),
            "needing_review": needing_review,
            "forgotten": forgotten,
            "avg_review_count": round(
                sum(t.review_count for t in self._traces.values()) / len(self._traces), 2
            ),
        }
