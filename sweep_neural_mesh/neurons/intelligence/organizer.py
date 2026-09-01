"""
IntelligenceOrganizer — structures, categorizes, and deduplicates intelligence.

Responsibilities:
  - Group intelligence by topic/domain.
  - Build knowledge graphs from relations.
  - Detect contradictions between items.
  - Merge related items into coherent summaries.
  - Maintain a structured knowledge representation.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict

from .gatherer import GatheredIntel, IntelSource


@dataclass
class IntelCluster:
    """A group of related intelligence items."""
    topic: str
    items: list[GatheredIntel]
    summary: str = ""
    confidence: float = 0.0
    contradictions: list[tuple[str, str]] = field(default_factory=list)
    relations: list[dict[str, str]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.items)

    @property
    def avg_confidence(self) -> float:
        if not self.items:
            return 0.0
        return sum(i.confidence for i in self.items) / len(self.items)


@dataclass
class OrganizedIntel:
    """The organized intelligence output."""
    query: str
    clusters: list[IntelCluster]
    contradictions: list[tuple[str, str, str]]  # (item_a, item_b, reason)
    knowledge_graph: dict[str, list[dict[str, str]]]  # entity -> relations
    total_items: int
    latency_ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def topics(self) -> list[str]:
        return [c.topic for c in self.clusters]

    @property
    def high_confidence_items(self) -> list[GatheredIntel]:
        return [
            item for cluster in self.clusters
            for item in cluster.items if item.confidence >= 0.8
        ]


class IntelligenceOrganizer:
    """Organizes gathered intelligence into structured knowledge.

    Usage::

        organizer = IntelligenceOrganizer()
        organized = organizer.organize(query="quantum computing", intel=items)
        for cluster in organized.clusters:
            print(f"{cluster.topic}: {cluster.size} items")
    """

    def __init__(self) -> None:
        self._knowledge_graph: dict[str, list[dict[str, str]]] = defaultdict(list)

    def organize(
        self,
        query: str,
        intel: list[GatheredIntel],
    ) -> OrganizedIntel:
        """Organize gathered intelligence into structured knowledge.

        Steps:
          1. Cluster by topic.
          2. Detect contradictions.
          3. Build knowledge graph.
          4. Generate summaries.
        """
        t0 = time.perf_counter()

        # 1. Cluster by topic
        clusters = self._cluster_by_topic(intel)

        # 2. Detect contradictions
        contradictions = self._detect_contradictions(intel)

        # 3. Build knowledge graph
        knowledge_graph = self._build_knowledge_graph(intel)

        # 4. Generate summaries
        for cluster in clusters:
            cluster.summary = self._generate_summary(cluster)
            cluster.relations = self._extract_cluster_relations(cluster)

        lat = (time.perf_counter() - t0) * 1000

        return OrganizedIntel(
            query=query,
            clusters=clusters,
            contradictions=contradictions,
            knowledge_graph=knowledge_graph,
            total_items=len(intel),
            latency_ms=lat,
        )

    def merge_clusters(self, clusters: list[IntelCluster]) -> IntelCluster:
        """Merge multiple clusters into one."""
        if not clusters:
            return IntelCluster(topic="empty", items=[])
        if len(clusters) == 1:
            return clusters[0]

        all_items = [item for c in clusters for item in c.items]
        topics = [c.topic for c in clusters]
        main_topic = max(set(topics), key=topics.count)

        return IntelCluster(
            topic=main_topic,
            items=all_items,
            summary="\n".join(c.summary for c in clusters if c.summary),
            confidence=sum(c.avg_confidence for c in clusters) / len(clusters),
        )

    # ── Internal ────────────────────────────────────────────

    def _cluster_by_topic(self, intel: list[GatheredIntel]) -> list[IntelCluster]:
        """Group intelligence items by topic."""
        by_topic: dict[str, list[GatheredIntel]] = defaultdict(list)

        for item in intel:
            topic = item.topic or "general"
            by_topic[topic].append(item)

        clusters = []
        for topic, items in sorted(by_topic.items(), key=lambda x: -len(x[1])):
            clusters.append(IntelCluster(
                topic=topic,
                items=items,
                confidence=sum(i.confidence for i in items) / len(items),
            ))

        return clusters

    def _detect_contradictions(
        self, intel: list[GatheredIntel],
    ) -> list[tuple[str, str, str]]:
        """Detect contradictions between intelligence items."""
        contradictions = []

        for i, a in enumerate(intel):
            for b in intel[i + 1:]:
                if a.overlaps_with(b, threshold=0.5):
                    # Check for contradictory relations
                    for rel_a in a.relations:
                        for rel_b in b.relations:
                            if (rel_a["subject"].lower() == rel_b["subject"].lower()
                                    and rel_a["predicate"] == rel_b["predicate"]
                                    and rel_a["object"].lower() != rel_b["object"].lower()):
                                contradictions.append((
                                    a.content[:100],
                                    b.content[:100],
                                    f"Conflicting: {rel_a['predicate']} {rel_a['object']} vs {rel_b['object']}",
                                ))

                    # Check for negation patterns
                    a_neg = bool(re.search(r'\b(not|never|no|cannot|false)\b', a.content.lower()))
                    b_neg = bool(re.search(r'\b(not|never|no|cannot|false)\b', b.content.lower()))
                    if a_neg != b_neg and a.topic == b.topic:
                        contradictions.append((
                            a.content[:100],
                            b.content[:100],
                            "One affirms, other negates the same topic",
                        ))

        return contradictions[:20]

    def _build_knowledge_graph(
        self, intel: list[GatheredIntel],
    ) -> dict[str, list[dict[str, str]]]:
        """Build a knowledge graph from relations."""
        graph: dict[str, list[dict[str, str]]] = defaultdict(list)

        for item in intel:
            for entity in item.entities:
                name = entity["name"]
                graph[name].append({
                    "type": "entity",
                    "category": entity.get("type", "unknown"),
                    "source": item.source.value,
                })

            for rel in item.relations:
                subj = rel["subject"]
                graph[subj].append({
                    "predicate": rel["predicate"],
                    "object": rel["object"],
                    "source": item.source.value,
                })

        # Update instance graph
        self._knowledge_graph.update(graph)

        return dict(graph)

    def _generate_summary(self, cluster: IntelCluster) -> str:
        """Generate a summary for a cluster."""
        if not cluster.items:
            return ""

        # Take the highest-confidence items
        sorted_items = sorted(cluster.items, key=lambda x: -x.confidence)
        top_items = sorted_items[:5]

        sentences = [item.content for item in top_items]
        return " ".join(sentences)

    def _extract_cluster_relations(self, cluster: IntelCluster) -> list[dict[str, str]]:
        """Extract all relations from a cluster."""
        relations = []
        for item in cluster.items:
            relations.extend(item.relations)
        return relations

    def get_knowledge_graph(self) -> dict[str, list[dict[str, str]]]:
        """Get the accumulated knowledge graph."""
        return dict(self._knowledge_graph)
