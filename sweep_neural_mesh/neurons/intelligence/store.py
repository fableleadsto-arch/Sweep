"""
IntelligenceStore — persists and retrieves organized intelligence.

Responsibilities:
  - Store intelligence items with metadata.
  - Retrieve by topic, source, confidence, or time range.
  - Merge new intelligence with existing.
  - Evict stale intelligence (TTL-based).
  - Provide statistics about stored intelligence.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .gatherer import GatheredIntel, IntelSource
from .organizer import OrganizedIntel, IntelCluster
from .analyzer import AnalyzedIntel


@dataclass
class StoredIntel:
    """A stored intelligence item with full metadata."""
    content: str
    source: str
    topic: str
    confidence: float
    timestamp: float
    entities: list[dict[str, str]]
    relations: list[dict[str, str]]
    query: str = ""
    content_id: str = ""
    access_count: int = 0
    last_accessed: float = 0.0


class IntelligenceStore:
    """Persistent store for organized intelligence.

    Usage::

        store = IntelligenceStore()
        store.store_intel(query="quantum computing", organized=org, analyzed=ana)
        results = store.retrieve(topic="physics", min_confidence=0.7)
        print(f"Stored: {store.stats()}")
    """

    def __init__(self, max_items: int = 10_000, ttl_seconds: float = 86400) -> None:
        self._items: list[StoredIntel] = []
        self._max_items = max_items
        self._ttl = ttl_seconds
        self._by_topic: dict[str, list[int]] = {}
        self._by_source: dict[str, list[int]] = {}

    def store_intel(
        self,
        query: str,
        organized: OrganizedIntel,
        analyzed: AnalyzedIntel | None = None,
    ) -> int:
        """Store organized and analyzed intelligence.

        Returns the number of new items stored.
        """
        stored = 0

        for cluster in organized.clusters:
            for item in cluster.items:
                if self._is_duplicate(item.content):
                    continue

                stored_item = StoredIntel(
                    content=item.content,
                    source=item.source.value,
                    topic=cluster.topic,
                    confidence=item.confidence,
                    timestamp=item.timestamp,
                    entities=item.entities,
                    relations=item.relations,
                    query=query,
                    content_id=item.content_id,
                )
                idx = len(self._items)
                self._items.append(stored_item)

                # Index by topic
                self._by_topic.setdefault(cluster.topic, []).append(idx)
                # Index by source
                self._by_source.setdefault(item.source.value, []).append(idx)

                stored += 1

        # Evict if over limit
        self._evict_stale()

        return stored

    def retrieve(
        self,
        topic: str | None = None,
        source: str | None = None,
        min_confidence: float = 0.0,
        max_items: int = 100,
    ) -> list[StoredIntel]:
        """Retrieve stored intelligence with optional filters."""
        candidates: list[int] | None = None

        if topic and topic in self._by_topic:
            candidates = set(self._by_topic[topic])
        if source and source in self._by_source:
            src_set = set(self._by_source[source])
            candidates = candidates & src_set if candidates else src_set

        if candidates is None:
            candidates = set(range(len(self._items)))

        results = []
        for idx in sorted(candidates):
            item = self._items[idx]
            if item.confidence < min_confidence:
                continue
            item.access_count += 1
            item.last_accessed = time.time()
            results.append(item)
            if len(results) >= max_items:
                break

        return results

    def search(
        self, query: str, max_results: int = 20,
    ) -> list[StoredIntel]:
        """Search stored intelligence by keyword relevance."""
        query_words = set(query.lower().split())

        scored: list[tuple[float, int]] = []
        for i, item in enumerate(self._items):
            item_words = set(item.content.lower().split())
            overlap = len(query_words & item_words)
            score = overlap * item.confidence
            scored.append((score, i))

        scored.sort(reverse=True)

        results = []
        for score, idx in scored[:max_results]:
            if score > 0:
                item = self._items[idx]
                item.access_count += 1
                item.last_accessed = time.time()
                results.append(item)

        return results

    def merge(self, other: "IntelligenceStore") -> int:
        """Merge another store into this one."""
        merged = 0
        for item in other._items:
            if not self._is_duplicate(item.content):
                idx = len(self._items)
                self._items.append(item)
                self._by_topic.setdefault(item.topic, []).append(idx)
                self._by_source.setdefault(item.source, []).append(idx)
                merged += 1
        return merged

    def get_topics(self) -> list[tuple[str, int]]:
        """Get all topics with item counts."""
        return sorted(
            [(topic, len(indices)) for topic, indices in self._by_topic.items()],
            key=lambda x: -x[1],
        )

    def get_sources(self) -> list[tuple[str, int]]:
        """Get all sources with item counts."""
        return sorted(
            [(source, len(indices)) for source, indices in self._by_source.items()],
            key=lambda x: -x[1],
        )

    def stats(self) -> dict[str, Any]:
        """Get store statistics."""
        if not self._items:
            return {"total_items": 0}

        confs = [i.confidence for i in self._items]
        return {
            "total_items": len(self._items),
            "topics": len(self._by_topic),
            "sources": len(self._by_source),
            "avg_confidence": sum(confs) / len(confs),
            "min_confidence": min(confs),
            "max_confidence": max(confs),
            "total_accesses": sum(i.access_count for i in self._items),
        }

    def clear(self) -> None:
        """Clear all stored intelligence."""
        self._items.clear()
        self._by_topic.clear()
        self._by_source.clear()

    def save(self, path: str) -> None:
        """Save store to disk."""
        data = {
            "items": [
                {
                    "content": i.content, "source": i.source, "topic": i.topic,
                    "confidence": i.confidence, "timestamp": i.timestamp,
                    "entities": i.entities, "relations": i.relations,
                    "query": i.query, "content_id": i.content_id,
                }
                for i in self._items
            ]
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> int:
        """Load store from disk. Returns number of items loaded."""
        if not os.path.exists(path):
            return 0

        with open(path) as f:
            data = json.load(f)

        loaded = 0
        for item_data in data.get("items", []):
            item = StoredIntel(**item_data)
            idx = len(self._items)
            self._items.append(item)
            self._by_topic.setdefault(item.topic, []).append(idx)
            self._by_source.setdefault(item.source, []).append(idx)
            loaded += 1

        return loaded

    # ── Internal ────────────────────────────────────────────

    def _is_duplicate(self, content: str) -> bool:
        """Check if content already exists in store."""
        # Simple hash-based dedup
        content_hash = hash(content[:200])
        for item in self._items[-100:]:  # Check recent items
            if hash(item.content[:200]) == content_hash:
                return True
        return False

    def _evict_stale(self) -> None:
        """Evict stale items if over max limit."""
        if len(self._items) <= self._max_items:
            return

        # Remove least recently accessed items
        now = time.time()
        for i in range(len(self._items) - 1, -1, -1):
            item = self._items[i]
            age = now - item.timestamp
            if age > self._ttl or item.access_count == 0:
                self._items.pop(i)
                # Rebuild indices (simplified)
                self._rebuild_indices()

            if len(self._items) <= self._max_items * 0.8:
                break

    def _rebuild_indices(self) -> None:
        """Rebuild topic and source indices."""
        self._by_topic.clear()
        self._by_source.clear()
        for i, item in enumerate(self._items):
            self._by_topic.setdefault(item.topic, []).append(i)
            self._by_source.setdefault(item.source, []).append(i)
