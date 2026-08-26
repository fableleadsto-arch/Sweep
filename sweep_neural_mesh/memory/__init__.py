"""
FeatureCache — caches NeuralPacket embeddings and intermediate results.

Avoids redundant computation when the same input is processed by
multiple nodes or re-processed across requests.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any


class FeatureCache:
    """Simple TTL-based cache for NeuralPacket data."""

    def __init__(self, max_entries: int = 1000, ttl_seconds: float = 3600.0) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[Any, float]] = {}
        self._hits = 0
        self._misses = 0

    def _key(self, data: Any) -> str:
        try:
            raw = json.dumps(data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            raw = str(data)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        value, created = entry
        if time.time() - created > self.ttl_seconds:
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        if len(self._store) >= self.max_entries:
            # Evict oldest
            oldest_key = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest_key]
        self._store[key] = (value, time.time())

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> None:
        self._store.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"FeatureCache(entries={self.size}, hit_rate={self.hit_rate:.2f})"
