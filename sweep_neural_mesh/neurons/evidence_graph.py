"""
Evidence Graph — graph-based evidence storage and correlation.

Instead of storing flat lists of search results, Sweep constructs:

    PERSON → APPEARS_IN → IMAGE
    PERSON → MENTIONED_IN → ARTICLE
    PERSON → ASSOCIATED_WITH → ORGANIZATION
    PERSON → LOCATED_AT → LOCATION
    PERSON → ATTENDED → EVENT → PHOTO/VIDEO/ARTICLE

This makes complex investigations much easier to understand.

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │              EVIDENCE GRAPH                          │
    │                                                     │
    │  Evidence Nodes:                                     │
    │    - Person, Org, Location, Event, Image, etc.      │
    │    - Each has: content, source, timestamp, weight   │
    │                                                     │
    │  Correlation Edges:                                  │
    │    - appears_in, mentioned_in, associated_with, etc. │
    │    - Each has: strength, evidence_text, confidence  │
    │                                                     │
    │  Graph Algorithms:                                   │
    │    - PageRank for evidence importance                │
    │    - Community detection for evidence clusters       │
    │    - Shortest path for evidence chains               │
    └─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EvidenceType(str, Enum):
    CLAIM = "claim"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    WEBPAGE = "webpage"
    TESTIMONY = "testimony"
    DATA = "data"
    OFFICIAL_RECORD = "official_record"


class CorrelationType(str, Enum):
    APPEARS_IN = "appears_in"
    MENTIONED_IN = "mentioned_in"
    ASSOCIATED_WITH = "associated_with"
    LOCATED_AT = "located_at"
    ATTENDED = "attended"
    AUTHORED = "authored"
    OCCURRED_DURING = "occurred_during"
    CORROBORATES = "corroborates"
    CONTRADICTS = "contradicts"
    RELATED_TO = "related_to"
    CAUSED_BY = "caused_by"
    TEMPORAL_OVERLAP = "temporal_overlap"
    SAME_PERSON = "same_person"
    SAME_ORG = "same_organization"


@dataclass
class EvidenceNode:
    """A node in the evidence graph."""
    node_id: str
    evidence_type: EvidenceType
    content: str
    source: str = ""
    timestamp: str = ""
    confidence: float = 0.8
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "type": self.evidence_type.value,
            "content": self.content[:200],
            "source": self.source,
            "confidence": self.confidence,
            "weight": self.weight,
        }


@dataclass
class CorrelationEdge:
    """An edge connecting two evidence nodes."""
    source_id: str
    target_id: str
    correlation_type: CorrelationType
    strength: float = 0.8
    evidence_text: str = ""
    confidence: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.correlation_type.value,
            "strength": self.strength,
            "confidence": self.confidence,
        }


@dataclass
class EvidenceChain:
    """A chain of evidence connected through correlations."""
    nodes: list[EvidenceNode]
    edges: list[CorrelationEdge]
    overall_confidence: float
    chain_strength: float
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "length": len(self.nodes),
            "confidence": self.overall_confidence,
            "strength": self.chain_strength,
            "description": self.description,
            "node_types": [n.evidence_type.value for n in self.nodes],
        }


@dataclass
class GraphStats:
    """Statistics about the evidence graph."""
    node_count: int = 0
    edge_count: int = 0
    type_distribution: dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    avg_weight: float = 0.0
    connected_components: int = 0
    strongest_chain_length: int = 0


class EvidenceGraph:
    """
    Graph-based evidence storage and correlation engine.

    Stores evidence as a directed graph where:
    - Nodes represent pieces of evidence (claims, images, documents, etc.)
    - Edges represent correlations between evidence (corroborates, contradicts, etc.)

    Supports:
    - Adding evidence from multiple modalities
    - Automatic correlation discovery
    - Evidence chain building
    - PageRank for evidence importance
    - Community detection for evidence clusters
    """

    def __init__(self) -> None:
        self._nodes: dict[str, EvidenceNode] = {}
        self._edges: list[CorrelationEdge] = []
        self._adjacency: dict[str, list[str]] = {}
        self._next_id = 0

    def _make_id(self, content: str, source: str) -> str:
        key = f"{content[:100]}:{source}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def add_evidence(
        self,
        content: str,
        evidence_type: EvidenceType = EvidenceType.CLAIM,
        source: str = "",
        timestamp: str = "",
        confidence: float = 0.8,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceNode:
        """Add a piece of evidence to the graph."""
        node_id = self._make_id(content, source)

        # Check for duplicates
        if node_id in self._nodes:
            existing = self._nodes[node_id]
            if confidence > existing.confidence:
                existing.confidence = confidence
            return existing

        node = EvidenceNode(
            node_id=node_id,
            evidence_type=evidence_type,
            content=content,
            source=source,
            timestamp=timestamp,
            confidence=confidence,
            weight=weight,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        self._adjacency[node_id] = []

        # Auto-correlate with existing evidence
        self._auto_correlate(node)

        return node

    def add_correlation(
        self,
        source_id: str,
        target_id: str,
        correlation_type: CorrelationType,
        strength: float = 0.8,
        evidence_text: str = "",
        confidence: float = 0.8,
    ) -> CorrelationEdge | None:
        """Add a correlation edge between two evidence nodes."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return None

        edge = CorrelationEdge(
            source_id=source_id,
            target_id=target_id,
            correlation_type=correlation_type,
            strength=strength,
            evidence_text=evidence_text,
            confidence=confidence,
        )
        self._edges.append(edge)

        if target_id not in self._adjacency.get(source_id, []):
            self._adjacency.setdefault(source_id, []).append(target_id)
        if source_id not in self._adjacency.get(target_id, []):
            self._adjacency.setdefault(target_id, []).append(source_id)

        return edge

    def _auto_correlate(self, new_node: EvidenceNode) -> None:
        """Automatically find correlations with existing evidence."""
        new_content = new_node.content.lower()

        for existing_id, existing_node in self._nodes.items():
            if existing_id == new_node.node_id:
                continue

            existing_content = existing_node.content.lower()

            # Simple content overlap for auto-correlation
            new_words = set(new_content.split())
            existing_words = set(existing_content.split())
            if not new_words or not existing_words:
                continue

            overlap = len(new_words & existing_words) / max(len(new_words | existing_words), 1)

            if overlap > 0.3:
                # Determine correlation type
                corr_type = CorrelationType.RELATED_TO

                # Check for contradiction
                negation_words = {"not", "never", "no", "isn't", "wasn't", "aren't", "don't", "doesn't"}
                new_neg = bool(new_words & negation_words)
                existing_neg = bool(existing_words & negation_words)
                if new_neg != existing_neg and overlap > 0.5:
                    corr_type = CorrelationType.CONTRADICTS
                elif overlap > 0.5:
                    corr_type = CorrelationType.CORROBORATES

                self.add_correlation(
                    source_id=existing_id,
                    target_id=new_node.node_id,
                    correlation_type=corr_type,
                    strength=overlap,
                    confidence=overlap * 0.9,
                )

    def find_chains(
        self,
        start_id: str,
        end_id: str,
        max_length: int = 6,
    ) -> list[EvidenceChain]:
        """Find evidence chains between two nodes."""
        if start_id not in self._nodes or end_id not in self._nodes:
            return []

        # BFS to find paths
        visited: set[str] = {start_id}
        queue: list[tuple[str, list[str], list[str]]] = [(start_id, [start_id], [])]
        chains: list[EvidenceChain] = []

        while queue:
            current, path, edge_ids = queue.pop(0)

            if len(path) > max_length:
                continue

            if current == end_id and len(path) > 1:
                nodes = [self._nodes[nid] for nid in path if nid in self._nodes]
                edges = [
                    e for e in self._edges
                    if e.source_id in path and e.target_id in path
                ]
                if nodes:
                    min_conf = min(n.confidence for n in nodes)
                    avg_strength = sum(e.strength for e in edges) / max(len(edges), 1)
                    chains.append(EvidenceChain(
                        nodes=nodes,
                        edges=edges,
                        overall_confidence=min_conf,
                        chain_strength=avg_strength,
                    ))
                continue

            for neighbor in self._adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor], edge_ids))

        return sorted(chains, key=lambda c: c.overall_confidence, reverse=True)

    def get_evidence_importance(self) -> dict[str, float]:
        """Simple PageRank-like importance scoring."""
        if not self._nodes:
            return {}

        n = len(self._nodes)
        damping = 0.85
        iterations = 20

        # Initialize scores
        scores = {nid: 1.0 / n for nid in self._nodes}

        for _ in range(iterations):
            new_scores = {}
            for nid in self._nodes:
                incoming = sum(
                    e.strength for e in self._edges
                    if e.target_id == nid
                )
                neighbor_scores = 0.0
                for e in self._edges:
                    if e.target_id == nid and e.source_id in scores:
                        out_degree = len(self._adjacency.get(e.source_id, []))
                        if out_degree > 0:
                            neighbor_scores += scores[e.source_id] * e.strength / out_degree

                new_scores[nid] = (1 - damping) / n + damping * (incoming / max(n, 1) + neighbor_scores)

            scores = new_scores

        return scores

    def find_communities(self) -> list[list[str]]:
        """Simple community detection using connected components."""
        visited: set[str] = set()
        communities: list[list[str]] = []

        for node_id in self._nodes:
            if node_id in visited:
                continue

            component: list[str] = []
            stack = [node_id]
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                for neighbor in self._adjacency.get(current, []):
                    if neighbor not in visited:
                        stack.append(neighbor)

            if component:
                communities.append(component)

        return communities

    def get_stats(self) -> GraphStats:
        """Get comprehensive graph statistics."""
        if not self._nodes:
            return GraphStats()

        type_dist: dict[str, int] = {}
        for node in self._nodes.values():
            t = node.evidence_type.value
            type_dist[t] = type_dist.get(t, 0) + 1

        communities = self.find_communities()
        importance = self.get_evidence_importance()

        return GraphStats(
            node_count=len(self._nodes),
            edge_count=len(self._edges),
            type_distribution=type_dist,
            avg_confidence=sum(n.confidence for n in self._nodes.values()) / len(self._nodes),
            avg_weight=sum(n.weight for n in self._nodes.values()) / len(self._nodes),
            connected_components=len(communities),
            strongest_chain_length=max((len(c) for c in communities), default=0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Export the full graph as a dictionary."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
            "stats": self.get_stats().__dict__,
        }
