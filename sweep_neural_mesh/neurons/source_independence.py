"""
Source Independence Tracker — understand when sources share origins.

Sweep should ideally understand that:

    Article A, Article B, Article C

might all originate from:

    Press Release X

Therefore:

    3 pages ≠ 3 independent confirmations.

This can make Sweep much more scientifically defensible.

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │         SOURCE INDEPENDENCE TRACKER                  │
    │                                                     │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Source Provenance Graph                     │  │
    │  │  (source → origin tracking)                  │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Content Similarity Analyzer                 │  │
    │  │  (detect shared content between sources)     │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Independence Scorer                         │  │
    │  │  (compute true independent source count)     │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Chain Tracker                               │  │
    │  │  (trace information back to original source) │  │
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
class SourceNode:
    """A source in the provenance graph."""
    source_id: str
    name: str
    source_type: str  # news, blog, government, academic, social, press_release, etc.
    domain: str = ""
    origin_source_id: str = ""  # What source this was derived from
    content_hash: str = ""
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "name": self.name,
            "type": self.source_type,
            "domain": self.domain,
            "origin": self.origin_source_id,
        }


@dataclass
class SourceGroup:
    """A group of sources sharing the same origin."""
    origin_source: SourceNode
    derived_sources: list[SourceNode]
    shared_content_ratio: float = 0.0
    independence_score: float = 0.0  # 0 = all from same origin, 1 = all independent

    @property
    def effective_count(self) -> int:
        """Effective independent source count."""
        return max(1, 1 + int(self.independence_score * len(self.derived_sources)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin_source.name,
            "derived_count": len(self.derived_sources),
            "independence_score": self.independence_score,
            "effective_count": self.effective_count,
        }


@dataclass
class IndependenceReport:
    """Report on source independence for evidence."""
    total_sources: int = 0
    independent_groups: list[SourceGroup] = field(default_factory=list)
    fully_independent: int = 0
    shared_origin_count: int = 0
    overall_independence_score: float = 0.0
    effective_source_count: int = 0
    provenance_chain: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_sources": self.total_sources,
            "fully_independent": self.fully_independent,
            "shared_origin_count": self.shared_origin_count,
            "overall_independence_score": self.overall_independence_score,
            "effective_source_count": self.effective_source_count,
            "groups": [g.to_dict() for g in self.independent_groups],
        }


class SourceIndependenceTracker:
    """
    Tracks source provenance and computes true independence.

    Detects when multiple sources share the same origin (press release,
    wire story, etc.) and computes a true independence score.
    """

    # ── Source type reliability weights ──────────────────────
    _SOURCE_RELIABILITY: dict[str, float] = {
        "government": 0.95,
        "academic": 0.90,
        "official_record": 0.92,
        "news_major": 0.85,
        "news_minor": 0.70,
        "press_release": 0.50,
        "wire_service": 0.60,
        "blog": 0.40,
        "social_media": 0.30,
        "forum": 0.25,
        "unknown": 0.50,
    }

    # ── Domain classification patterns ───────────────────────
    _DOMAIN_TYPES: list[tuple[str, str]] = [
        (r'(\.gov|\bgov\.)(\w+)?', "government"),
        (r'(\.edu|\bedu\.)(\w+)?', "academic"),
        (r'(\.org|\borg\.)(\w+)?', "organization"),
        (r'(reuters|ap|afp|press)', "wire_service"),
        (r'(bbc|cnn|nytimes|guardian|washingtonpost|aljazeera)', "news_major"),
        (r'(wordpress|blogspot|medium\.com|substack)', "blog"),
        (r'(twitter|facebook|instagram|reddit|tiktok)', "social_media"),
        # Also check source name for government indicators
        (r'(government|official|federal|ministry|department|agency|council)', "government"),
    ]

    def __init__(self) -> None:
        self._sources: dict[str, SourceNode] = {}
        self._content_hashes: dict[str, list[str]] = {}  # hash → [source_ids]
        self._groups: list[SourceGroup] = []

    def _classify_source(self, name: str, domain: str = "") -> str:
        """Classify a source by type."""
        combined = f"{name} {domain}".lower()
        for pattern, source_type in self._DOMAIN_TYPES:
            if re.search(pattern, combined):
                return source_type
        return "unknown"

    def _extract_domain(self, url_or_name: str) -> str:
        """Extract domain from URL or name."""
        url_or_name = url_or_name.lower()
        url_or_name = re.sub(r'^https?://(www\.)?', '', url_or_name)
        domain = url_or_name.split('/')[0]
        return domain

    def _compute_content_hash(self, text: str) -> str:
        """Compute normalized content hash."""
        normalized = text.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # Take first 500 chars for fingerprint
        return hashlib.md5(normalized[:500].encode()).hexdigest()[:12]

    def add_source(
        self,
        name: str,
        content: str = "",
        url: str = "",
        origin_source_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> SourceNode:
        """Add a source to the tracker."""
        source_id = hashlib.md5(f"{name}:{url[:50]}".encode()).hexdigest()[:10]
        domain = self._extract_domain(url or name)
        source_type = self._classify_source(name, domain)
        content_hash = self._compute_content_hash(content) if content else ""

        node = SourceNode(
            source_id=source_id,
            name=name,
            source_type=source_type,
            domain=domain,
            origin_source_id=origin_source_id,
            content_hash=content_hash,
            confidence=self._SOURCE_RELIABILITY.get(source_type, 0.5),
            metadata=metadata or {},
        )

        self._sources[source_id] = node

        # Index by content hash
        if content_hash:
            self._content_hashes.setdefault(content_hash, []).append(source_id)

        return node

    def _find_content_clusters(self) -> list[list[str]]:
        """Find sources with identical/near-identical content."""
        clusters: list[list[str]] = []
        for h, source_ids in self._content_hashes.items():
            if len(source_ids) >= 2:
                clusters.append(source_ids)
        return clusters

    def _compute_independence_score(self, group: SourceGroup) -> float:
        """Compute independence score for a group of sources."""
        if len(group.derived_sources) <= 1:
            return 1.0

        # Factors reducing independence:
        # 1. Same domain
        # 2. Same source type
        # 3. Shared content
        same_domain = sum(
            1 for s in group.derived_sources
            if s.domain == group.origin_source.domain
        )
        same_type = sum(
            1 for s in group.derived_sources
            if s.source_type == group.origin_source.source_type
        )

        domain_penalty = same_domain / max(len(group.derived_sources), 1) * 0.5
        type_penalty = same_type / max(len(group.derived_sources), 1) * 0.3
        content_penalty = group.shared_content_ratio * 0.2

        score = 1.0 - domain_penalty - type_penalty - content_penalty
        return max(0.1, min(1.0, score))

    def analyze(self) -> IndependenceReport:
        """Analyze source independence across all tracked sources."""
        t0 = time.perf_counter()

        if not self._sources:
            return IndependenceReport(latency_ms=(time.perf_counter() - t0) * 1000)

        # Find content clusters
        content_clusters = self._find_content_clusters()

        # Group sources by origin
        origin_groups: dict[str, list[SourceNode]] = defaultdict(list)
        standalone: list[SourceNode] = []

        for source in self._sources.values():
            if source.origin_source_id:
                origin_groups[source.origin_source_id].append(source)
            else:
                standalone.append(source)

        # Also group content-identical sources
        for cluster in content_clusters:
            nodes = [self._sources[sid] for sid in cluster if sid in self._sources]
            if len(nodes) >= 2:
                # Use oldest as origin
                origin = nodes[0]
                for n in nodes[1:]:
                    n.origin_source_id = origin.source_id
                    origin_groups[origin.source_id].append(n)

        # Build source groups
        groups: list[SourceGroup] = []
        for origin_id, derived in origin_groups.items():
            origin = self._sources.get(origin_id)
            if not origin:
                continue

            group = SourceGroup(
                origin_source=origin,
                derived_sources=derived,
                shared_content_ratio=0.5,  # Simplified
            )
            group.independence_score = self._compute_independence_score(group)
            groups.append(group)

        # Compute overall statistics
        total = len(self._sources)
        fully_independent = len(standalone)
        shared = sum(len(g.derived_sources) for g in groups)

        overall_score = 1.0
        if total > 0:
            independent_contributions = fully_independent
            for g in groups:
                independent_contributions += g.effective_count
            overall_score = independent_contributions / max(total, 1)

        effective_count = fully_independent + sum(g.effective_count for g in groups)

        # Provenance chain
        chain = []
        for source in self._sources.values():
            if source.origin_source_id:
                origin = self._sources.get(source.origin_source_id)
                if origin:
                    chain.append({
                        "source": source.name,
                        "derived_from": origin.name,
                        "type": source.source_type,
                    })

        latency = (time.perf_counter() - t0) * 1000

        return IndependenceReport(
            total_sources=total,
            independent_groups=groups,
            fully_independent=fully_independent,
            shared_origin_count=shared,
            overall_independence_score=min(1.0, overall_score),
            effective_source_count=effective_count,
            provenance_chain=chain,
            latency_ms=latency,
        )
