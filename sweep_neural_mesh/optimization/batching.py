"""
Model optimization utilities for the Neural Mesh.

Provides:
- Pruning analysis (identify redundant computations)
- Batching strategies (group similar requests)
- Caching policies (LRU, TTL, frequency-based)
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PruningReport:
    """Analysis of which nodes/features can be pruned."""
    total_features: int = 0
    features_kept: int = 0
    features_pruned: int = 0
    sparsity: float = 0.0
    estimated_speedup: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)


class Pruner:
    """
    Analyzes and prunes redundant computations in the Mesh.

    Does NOT modify models directly. Instead, produces reports
    that the router and resource manager use to skip unnecessary
    nodes or reduce computation.
    """

    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold

    def analyze_sparsity(self, data: list[float]) -> PruningReport:
        """Analyze how many values in a vector are near zero."""
        if not data:
            return PruningReport()
        near_zero = sum(1 for x in data if abs(x) < self.threshold)
        total = len(data)
        kept = total - near_zero
        sparsity = near_zero / total
        return PruningReport(
            total_features=total,
            features_kept=kept,
            features_pruned=near_zero,
            sparsity=sparsity,
            estimated_speedup=1.0 / max(1.0 - sparsity, 0.1),
        )

    def analyze_node_redundancy(
        self, node_latencies: dict[str, list[float]]
    ) -> dict[str, Any]:
        """Identify nodes whose output variance is very low (low information)."""
        results = {}
        for name, latencies in node_latencies.items():
            if not latencies:
                continue
            mean = sum(latencies) / len(latencies)
            variance = sum((l - mean) ** 2 for l in latencies) / len(latencies)
            cv = (variance ** 0.5) / mean if mean > 0 else 0  # coefficient of variation
            results[name] = {
                "mean_latency_ms": mean,
                "cv_latency": cv,
                "high_variance": cv > 0.5,
                "suggestion": "consider caching" if cv > 0.5 else "stable",
            }
        return results


class BatchingStrategy:
    """
    Groups similar requests for efficient batch processing.

    This is a scheduling optimization — it does not change model
    behavior, only groups compatible inputs for vectorised execution.
    """

    def __init__(self, max_batch_size: int = 32, max_wait_ms: float = 10.0) -> None:
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._pending: deque[tuple[str, Any, float]] = deque()

    def add_request(self, request_id: str, data: Any) -> None:
        self._pending.append((request_id, data, time.time()))

    def get_batch(self) -> list[tuple[str, Any]]:
        """Extract a batch of requests ready for processing."""
        batch = []
        now = time.time()
        while self._pending and len(batch) < self.max_batch_size:
            req_id, data, added_at = self._pending[0]
            if len(batch) >= self.max_batch_size:
                break
            if (now - added_at) * 1000 >= self.max_wait_ms or len(batch) == 0:
                self._pending.popleft()
                batch.append((req_id, data))
            else:
                break
        return batch

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def clear(self) -> None:
        self._pending.clear()


class CachePolicy:
    """
    Adaptive caching policy for NeuralPacket data.

    Supports LRU, TTL, and frequency-based eviction.
    """

    def __init__(
        self,
        max_entries: int = 1000,
        ttl_seconds: float = 3600.0,
        policy: str = "lru",
    ) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.policy = policy
        self._store: dict[str, tuple[Any, float, int]] = {}  # key -> (value, timestamp, hits)

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts, hits = entry
        if time.time() - ts > self.ttl_seconds:
            del self._store[key]
            return None
        self._store[key] = (value, ts, hits + 1)
        return value

    def put(self, key: str, value: Any) -> None:
        if len(self._store) >= self.max_entries:
            self._evict()
        self._store[key] = (value, time.time(), 0)

    def _evict(self) -> None:
        if not self._store:
            return
        if self.policy == "lru":
            oldest = min(self._store, key=lambda k: self._store[k][1])
        elif self.policy == "lfu":
            oldest = min(self._store, key=lambda k: self._store[k][2])
        else:
            oldest = min(self._store, key=lambda k: self._store[k][1])
        del self._store[oldest]

    @property
    def size(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()

    def __repr__(self) -> str:
        return f"CachePolicy(policy={self.policy}, entries={self.size})"
