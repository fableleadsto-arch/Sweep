"""
Deduplication Engine — detect duplicate and near-duplicate content.

The internet contains massive amounts of duplicated information.
20 webpages ≠ 20 independent pieces of evidence. That's a critical distinction.

This module detects:
- Duplicate pages (identical content)
- Near-duplicate content (slight variations)
- Syndicated articles (same story from same source)
- Reposted images (same image, different platforms)
- Same claim from the same underlying source

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │           DEDUPLICATION ENGINE                       │
    │                                                     │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Content Hasher                              │  │
    │  │  (exact hash + SimHash for near-duplicates)  │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Similarity Detector                         │  │
    │  │  (cosine similarity on content features)     │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Source Tracker                              │  │
    │  │  (identify syndicated/copied content)        │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Independence Assessor                       │  │
    │  │  (count truly independent sources)           │  │
    │  └──────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContentItem:
    """A piece of content to check for duplicates."""
    item_id: str
    content: str
    source: str = ""
    url: str = ""
    content_type: str = "text"  # text, image, document
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """Exact content hash."""
        normalized = self._normalize(self.content)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    @property
    def simhash(self) -> int:
        """SimHash for near-duplicate detection."""
        return self._compute_simhash(self.content)

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text

    @staticmethod
    def _compute_simhash(text: str) -> int:
        """Compute SimHash fingerprint."""
        normalized = ContentItem._normalize(text)
        tokens = normalized.split()

        # 64-bit SimHash
        v = [0] * 64
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
            for i in range(64):
                if h & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1

        fingerprint = 0
        for i in range(64):
            if v[i] > 0:
                fingerprint |= (1 << i)
        return fingerprint

    @staticmethod
    def _hamming_distance(h1: int, h2: int) -> int:
        """Compute Hamming distance between two hashes."""
        xor = h1 ^ h2
        count = 0
        while xor:
            count += 1
            xor &= xor - 1
        return count


@dataclass
class DuplicateCluster:
    """A cluster of duplicate/near-duplicate items."""
    cluster_id: str
    items: list[ContentItem]
    representative: ContentItem  # The "canonical" version
    similarity_scores: dict[str, float] = field(default_factory=dict)
    duplication_type: str = "exact"  # exact, near_duplicate, syndicated

    @property
    def independent_count(self) -> int:
        """Count of truly independent sources."""
        sources = set()
        for item in self.items:
            # Normalize source domain
            source = item.source.lower()
            source = re.sub(r'^https?://(www\.)?', '', source)
            source = source.split('/')[0]
            sources.add(source)
        return len(sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "item_count": len(self.items),
            "independent_sources": self.independent_count,
            "duplication_type": self.duplication_type,
            "representative": self.representative.content[:100],
        }


@dataclass
class DeduplicationResult:
    """Result of deduplication analysis."""
    total_items: int
    unique_items: int
    duplicate_clusters: list[DuplicateCluster]
    exact_duplicates: int
    near_duplicates: int
    syndicated: int
    independence_ratio: float  # unique / total
    effective_evidence_count: int  # after deduplication
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_items": self.total_items,
            "unique_items": self.unique_items,
            "exact_duplicates": self.exact_duplicates,
            "near_duplicates": self.near_duplicates,
            "syndicated": self.syndicated,
            "independence_ratio": self.independence_ratio,
            "effective_evidence_count": self.effective_evidence_count,
            "clusters": [c.to_dict() for c in self.duplicate_clusters],
        }


class DeduplicationEngine:
    """
    Content deduplication engine.

    Detects exact duplicates, near-duplicates, and syndicated content.
    Computes effective evidence count after removing duplicates.
    """

    def __init__(
        self,
        simhash_threshold: int = 10,
        similarity_threshold: float = 0.85,
    ) -> None:
        self._simhash_threshold = simhash_threshold
        self._similarity_threshold = similarity_threshold
        self._items: list[ContentItem] = []
        self._hash_index: dict[str, list[int]] = {}  # hash → [indices]
        self._simhash_index: list[tuple[int, int]] = []  # (simhash, index)

    def add_item(self, item: ContentItem) -> None:
        """Add a content item for deduplication."""
        idx = len(self._items)
        self._items.append(item)

        # Index by exact hash
        h = item.content_hash
        self._hash_index.setdefault(h, []).append(idx)

        # Index by SimHash
        self._simhash_index.append((item.simhash, idx))

    def add_content(
        self,
        content: str,
        source: str = "",
        url: str = "",
        content_type: str = "text",
        item_id: str | None = None,
    ) -> ContentItem:
        """Convenience: create and add a content item."""
        if item_id is None:
            item_id = hashlib.md5(content[:100].encode()).hexdigest()[:8]
        item = ContentItem(
            item_id=item_id,
            content=content,
            source=source,
            url=url,
            content_type=content_type,
        )
        self.add_item(item)
        return item

    def _find_exact_duplicates(self) -> list[DuplicateCluster]:
        """Find exact duplicate clusters."""
        clusters: list[DuplicateCluster] = []

        for h, indices in self._hash_index.items():
            if len(indices) < 2:
                continue

            items = [self._items[i] for i in indices]
            cluster = DuplicateCluster(
                cluster_id=f"exact_{h[:8]}",
                items=items,
                representative=items[0],
                duplication_type="exact",
            )
            clusters.append(cluster)

        return clusters

    def _find_near_duplicates(self) -> list[DuplicateCluster]:
        """Find near-duplicate clusters using SimHash."""
        clusters: list[DuplicateCluster] = []
        used: set[int] = set()

        # Sort by SimHash for efficient comparison
        sorted_pairs = sorted(self._simhash_index, key=lambda x: x[0])

        for i, (h1, idx1) in enumerate(sorted_pairs):
            if idx1 in used:
                continue

            group = [idx1]
            for j in range(i + 1, len(sorted_pairs)):
                h2, idx2 = sorted_pairs[j]
                if idx2 in used:
                    continue

                # Early termination if hashes too far apart
                if abs(h1 - h2) > self._simhash_threshold * 1000:
                    break

                dist = ContentItem._hamming_distance(h1, h2)
                if dist <= self._simhash_threshold:
                    group.append(idx2)

            if len(group) >= 2:
                items = [self._items[idx] for idx in group]
                # Compute actual similarity for the cluster
                rep = items[0]
                sim_scores = {}
                for item in items[1:]:
                    sim = self._compute_similarity(rep.content, item.content)
                    sim_scores[item.item_id] = sim

                cluster = DuplicateCluster(
                    cluster_id=f"near_{rep.item_id}",
                    items=items,
                    representative=rep,
                    similarity_scores=sim_scores,
                    duplication_type="near_duplicate",
                )
                clusters.append(cluster)
                used.update(group)

        return clusters

    @staticmethod
    def _compute_similarity(text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / max(union, 1)

    def _detect_syndication(self) -> list[DuplicateCluster]:
        """Detect syndicated content (same source domain)."""
        source_groups: dict[str, list[int]] = defaultdict(list)

        for idx, item in enumerate(self._items):
            source = item.source.lower()
            source = re.sub(r'^https?://(www\.)?', '', source)
            source = source.split('/')[0]
            if source:
                source_groups[source].append(idx)

        clusters: list[DuplicateCluster] = []
        for source, indices in source_groups.items():
            if len(indices) < 2:
                continue

            items = [self._items[i] for i in indices]
            # Check if content is similar within same source
            rep = items[0]
            similar_items = [rep]
            for item in items[1:]:
                sim = self._compute_similarity(rep.content, item.content)
                if sim > 0.5:
                    similar_items.append(item)

            if len(similar_items) >= 2:
                cluster = DuplicateCluster(
                    cluster_id=f"synd_{source[:10]}",
                    items=similar_items,
                    representative=rep,
                    duplication_type="syndicated",
                )
                clusters.append(cluster)

        return clusters

    def deduplicate(self) -> DeduplicationResult:
        """Run full deduplication analysis."""
        t0 = time.perf_counter()

        exact = self._find_exact_duplicates()
        near = self._find_near_duplicates()
        syndicated = self._detect_syndication()

        # Merge all clusters
        all_clusters = exact + near + syndicated

        # Count unique items
        duplicated_ids: set[str] = set()
        for cluster in all_clusters:
            for item in cluster.items[1:]:  # Keep first as representative
                duplicated_ids.add(item.item_id)

        unique = len(self._items) - len(duplicated_ids)
        total = len(self._items)
        independence = unique / max(total, 1)

        # Effective evidence: sum of independent sources per cluster + standalone items
        effective = unique
        for cluster in all_clusters:
            effective += cluster.independent_count - 1  # Already counted one per cluster

        latency = (time.perf_counter() - t0) * 1000

        return DeduplicationResult(
            total_items=total,
            unique_items=unique,
            duplicate_clusters=all_clusters,
            exact_duplicates=sum(len(c.items) - 1 for c in exact),
            near_duplicates=sum(len(c.items) - 1 for c in near),
            syndicated=sum(len(c.items) for c in syndicated),
            independence_ratio=independence,
            effective_evidence_count=effective,
            latency_ms=latency,
        )
