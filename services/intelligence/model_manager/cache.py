"""Model cache — manages loaded models in memory with eviction."""

from __future__ import annotations

import gc
import sys
import time
from typing import Any


class ModelCache:
    """LRU cache for loaded models with memory-aware eviction."""

    def __init__(self, max_size: int = 4, max_memory_mb: int = 2000):
        self._max_size = max_size
        self._max_memory_mb = max_memory_mb
        self._cache: dict[str, Any] = {}
        self._access_order: list[str] = []
        self._size_mb: dict[str, float] = {}
        self._loaded_at: dict[str, float] = {}

    def get(self, key: str) -> Any | None:
        if key in self._cache:
            self._touch(key)
            return self._cache[key]
        return None

    def put(self, key: str, model: Any, size_mb: float = 0):
        if key in self._cache:
            self._touch(key)
            self._cache[key] = model
            return

        # Evict if at capacity
        while len(self._cache) >= self._max_size:
            self._evict_oldest()

        self._cache[key] = model
        self._access_order.append(key)
        self._size_mb[key] = size_mb
        self._loaded_at[key] = time.time()

    def remove(self, key: str) -> bool:
        if key not in self._cache:
            return False
        del self._cache[key]
        self._access_order.remove(key)
        self._size_mb.pop(key, None)
        self._loaded_at.pop(key, None)
        gc.collect()
        return True

    def clear(self):
        self._cache.clear()
        self._access_order.clear()
        self._size_mb.clear()
        self._loaded_at.clear()
        gc.collect()

    def _touch(self, key: str):
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict_oldest(self):
        if not self._access_order:
            return
        oldest = self._access_order[0]
        self.remove(oldest)

    def _evict_by_memory(self, needed_mb: float):
        """Evict models until we have room."""
        while (self.total_size_mb() + needed_mb > self._max_memory_mb
               and self._access_order):
            self._evict_oldest()

    @property
    def loaded_models(self) -> list[str]:
        return list(self._access_order)

    @property
    def count(self) -> int:
        return len(self._cache)

    def total_size_mb(self) -> float:
        return sum(self._size_mb.values())

    def stats(self) -> dict:
        return {
            "count": self.count,
            "max_size": self._max_size,
            "total_mb": round(self.total_size_mb(), 1),
            "max_mb": self._max_memory_mb,
            "models": self.loaded_models,
        }
