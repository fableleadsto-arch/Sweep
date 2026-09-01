"""
Recursive Investigation Engine — graph-based traversal that discovers new
investigation nodes from each finding.

The key insight: each discovery becomes a potential new investigation node.
Sweep discovers Person → Company → Website → Employee → Organization → Event → Photo.

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │          RECURSIVE INVESTIGATION ENGINE              │
    │                                                     │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Investigation Graph                         │  │
    │  │  Nodes: person, org, location, event, etc.   │  │
    │  │  Edges: works_at, located_in, attended, etc. │  │
    │  └──────────────────────────────────────────────┘  │
    │         ↓                                           │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Discovery Queue (BFS/DFS traversal)         │  │
    │  │  Each node → extract relationships → new nodes│  │
    │  └──────────────────────────────────────────────┘  │
    │         ↓                                           │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Depth Limiter + Confidence Decay            │  │
    │  │  Stops when confidence < threshold           │  │
    │  └──────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import re
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    WEBSITE = "website"
    USERNAME = "username"
    EMAIL = "email"
    PHONE = "phone"
    IMAGE = "image"
    DOCUMENT = "document"
    DATE = "date"
    CLAIM = "claim"
    UNKNOWN = "unknown"


class EdgeType(str, Enum):
    WORKS_AT = "works_at"
    LOCATED_IN = "located_in"
    ATTENDED = "attended"
    OWNS = "owns"
    MENTIONED_IN = "mentioned_in"
    ASSOCIATED_WITH = "associated_with"
    LINKED_TO = "linked_to"
    AUTHORED = "authored"
    OCCURRED_ON = "occurred_on"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    RELATED_TO = "related_to"


@dataclass
class InvNode:
    """A node in the investigation graph."""
    node_id: str
    node_type: NodeType
    label: str
    data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    depth: int = 0
    discovered_by: str = ""
    created_at: float = field(default_factory=time.time)
    visited: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "confidence": self.confidence,
            "depth": self.depth,
            "visited": self.visited,
            "data_keys": list(self.data.keys()),
        }


@dataclass
class InvEdge:
    """A directed edge in the investigation graph."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    confidence: float = 0.8
    evidence: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "edge_type": self.edge_type.value,
            "confidence": self.confidence,
        }


@dataclass
class Discovery:
    """A single discovery made during recursive investigation."""
    source_node_id: str
    discovered_type: NodeType
    label: str
    relationship: EdgeType
    evidence: str = ""
    confidence: float = 0.7


@dataclass
class InvestigationResult:
    """Result of a recursive investigation."""
    nodes: list[InvNode]
    edges: list[InvEdge]
    discoveries: list[Discovery]
    total_depth: int
    nodes_visited: int
    nodes_discovered: int
    confidence_range: tuple[float, float]
    investigation_path: list[str]
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "discovery_count": len(self.discoveries),
            "total_depth": self.total_depth,
            "nodes_visited": self.nodes_visited,
            "confidence_range": self.confidence_range,
            "investigation_path": self.investigation_path,
        }


class RecursiveInvestigationEngine:
    """
    Performs recursive graph-based investigation.

    Given an initial target (person, org, etc.), it:
    1. Creates an initial node
    2. Extracts relationships from evidence
    3. Creates new nodes for each discovered entity
    4. Recursively investigates new nodes up to a depth limit
    5. Builds an investigation graph showing all connections

    The engine uses confidence decay to prevent infinite recursion:
    each level of depth reduces confidence by a factor.
    """

    # ── Entity extraction patterns ────────────────────────────
    _PERSON_PATTERNS = [
        r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',           # John Smith
        r'\b([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+)\b',   # John M. Smith
        r'\b(Mr|Mrs|Ms|Dr|Prof)\.?\s+([A-Z][a-z]+ [A-Z][a-z]+)\b',
    ]

    _ORG_PATTERNS = [
        r'\b([A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC|Co|Company|Group|Foundation|Institute|University|Agency))\b',
        r'\b(The [A-Z][a-z]+ (?:Company|Group|Foundation|Institute))\b',
        r'\b([A-Z][a-z]+ [A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC))\b',
    ]

    _LOCATION_PATTERNS = [
        r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',           # in Delhi
        r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',           # at London
        r'\bnear\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',         # near Paris
        r'\b([A-Z][a-z]+),?\s+([A-Z][a-z]+)\b',               # Delhi, India
    ]

    _EVENT_PATTERNS = [
        r'\b(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:conference|summit|meeting|event|festival|ceremony|workshop|seminar))\b',
        r'\b(attended|participated in|spoke at|presented at)\s+(?:the\s+)?(.+?)(?:\.|,|$)',
    ]

    _DATE_PATTERNS = [
        r'\b(\d{4})\b',
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',
    ]

    _EMAIL_PATTERN = r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
    _URL_PATTERN = r'\b(https?://[^\s]+|www\.[^\s]+)\b'
    _USERNAME_PATTERN = r'@([a-zA-Z0-9_]{3,20})\b'

    def __init__(
        self,
        max_depth: int = 5,
        confidence_threshold: float = 0.25,
        confidence_decay: float = 0.85,
        max_nodes: int = 200,
    ) -> None:
        self._max_depth = max_depth
        self._confidence_threshold = confidence_threshold
        self._confidence_decay = confidence_decay
        self._max_nodes = max_nodes

        self._nodes: dict[str, InvNode] = {}
        self._edges: list[InvEdge] = []
        self._next_id = 0

    def _make_id(self, label: str, node_type: NodeType) -> str:
        key = f"{node_type.value}:{label.lower().strip()}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _get_or_create_node(
        self,
        label: str,
        node_type: NodeType,
        depth: int,
        confidence: float,
        discovered_by: str = "",
        data: dict[str, Any] | None = None,
    ) -> InvNode:
        node_id = self._make_id(label, node_type)
        if node_id in self._nodes:
            existing = self._nodes[node_id]
            if confidence > existing.confidence:
                existing.confidence = confidence
            return existing
        node = InvNode(
            node_id=node_id,
            node_type=node_type,
            label=label.strip(),
            data=data or {},
            confidence=confidence,
            depth=depth,
            discovered_by=discovered_by,
        )
        self._nodes[node_id] = node
        return node

    def _add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        confidence: float = 0.8,
        evidence: str = "",
    ) -> InvEdge:
        edge = InvEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            confidence=confidence,
            evidence=evidence,
        )
        self._edges.append(edge)
        return edge

    # ── Entity extraction from text ──────────────────────────

    def _extract_discoveries(
        self, text: str, source_node: InvNode
    ) -> list[Discovery]:
        """Extract entity discoveries from text near a source node."""
        discoveries: list[Discovery] = []

        # People
        for pat in self._PERSON_PATTERNS:
            for m in re.finditer(pat, text):
                name = m.group(0) if not m.lastindex else m.group(m.lastindex)
                if len(name) > 3:
                    discoveries.append(Discovery(
                        source_node_id=source_node.node_id,
                        discovered_type=NodeType.PERSON,
                        label=name,
                        relationship=EdgeType.ASSOCIATED_WITH,
                        evidence=m.group(0),
                        confidence=0.7,
                    ))

        # Organizations
        for pat in self._ORG_PATTERNS:
            for m in re.finditer(pat, text):
                discoveries.append(Discovery(
                    source_node_id=source_node.node_id,
                    discovered_type=NodeType.ORGANIZATION,
                    label=m.group(0),
                    relationship=EdgeType.ASSOCIATED_WITH,
                    evidence=m.group(0),
                    confidence=0.75,
                ))

        # Locations
        for pat in self._LOCATION_PATTERNS:
            for m in re.finditer(pat, text):
                loc = m.group(1) if m.lastindex else m.group(0)
                if len(loc) > 2:
                    discoveries.append(Discovery(
                        source_node_id=source_node.node_id,
                        discovered_type=NodeType.LOCATION,
                        label=loc,
                        relationship=EdgeType.LOCATED_IN,
                        evidence=m.group(0),
                        confidence=0.7,
                    ))

        # Events
        for pat in self._EVENT_PATTERNS:
            for m in re.finditer(pat, text):
                event = m.group(0)
                if len(event) > 5:
                    discoveries.append(Discovery(
                        source_node_id=source_node.node_id,
                        discovered_type=NodeType.EVENT,
                        label=event,
                        relationship=EdgeType.ATTENDED,
                        evidence=m.group(0),
                        confidence=0.6,
                    ))

        # Dates
        for pat in self._DATE_PATTERNS:
            for m in re.finditer(pat, text):
                discoveries.append(Discovery(
                    source_node_id=source_node.node_id,
                    discovered_type=NodeType.DATE,
                    label=m.group(0),
                    relationship=EdgeType.OCCURRED_ON,
                    evidence=m.group(0),
                    confidence=0.8,
                ))

        # Emails
        for m in re.finditer(self._EMAIL_PATTERN, text):
            discoveries.append(Discovery(
                source_node_id=source_node.node_id,
                discovered_type=NodeType.EMAIL,
                label=m.group(0),
                relationship=EdgeType.ASSOCIATED_WITH,
                evidence=m.group(0),
                confidence=0.9,
            ))

        # URLs
        for m in re.finditer(self._URL_PATTERN, text):
            discoveries.append(Discovery(
                source_node_id=source_node.node_id,
                discovered_type=NodeType.WEBSITE,
                label=m.group(0),
                relationship=EdgeType.LINKED_TO,
                evidence=m.group(0),
                confidence=0.8,
            ))

        # Usernames
        for m in re.finditer(self._USERNAME_PATTERN, text):
            discoveries.append(Discovery(
                source_node_id=source_node.node_id,
                discovered_type=NodeType.USERNAME,
                label=m.group(0),
                relationship=EdgeType.ASSOCIATED_WITH,
                evidence=m.group(0),
                confidence=0.85,
            ))

        # Deduplicate by (type, label)
        seen: set[tuple[str, str]] = set()
        unique: list[Discovery] = []
        for d in discoveries:
            key = (d.discovered_type.value, d.label.lower().strip())
            if key not in seen and d.source_node_id != self._make_id(d.label, d.discovered_type):
                seen.add(key)
                unique.append(d)

        return unique

    # ── Main investigation API ────────────────────────────────

    def investigate(
        self,
        target: str,
        target_type: NodeType = NodeType.PERSON,
        evidence_texts: list[str] | None = None,
    ) -> InvestigationResult:
        """
        Run recursive investigation starting from a target.

        Args:
            target: The investigation target (name, URL, etc.)
            target_type: Type of the target
            evidence_texts: Initial evidence to process

        Returns:
            InvestigationResult with full investigation graph
        """
        t0 = time.perf_counter()
        self._nodes.clear()
        self._edges.clear()
        self._next_id = 0

        all_evidence = evidence_texts or []
        investigation_path: list[str] = []

        # Create root node
        root = self._get_or_create_node(
            label=target,
            node_type=target_type,
            depth=0,
            confidence=1.0,
        )
        root.visited = True
        investigation_path.append(f"[0] {target_type.value}: {target}")

        # BFS queue: (node, depth, evidence_texts)
        queue: list[tuple[InvNode, int, list[str]]] = [(root, 0, all_evidence)]
        discoveries_all: list[Discovery] = []
        nodes_visited = 0

        while queue and len(self._nodes) < self._max_nodes:
            current, depth, evidence = queue.pop(0)

            if depth > self._max_depth:
                continue
            if current.confidence < self._confidence_threshold:
                continue

            nodes_visited += 1

            # Extract discoveries from evidence
            combined_text = " ".join(evidence) if evidence else ""
            discoveries = self._extract_discoveries(combined_text, current)

            for disc in discoveries:
                # Apply confidence decay
                decayed_conf = disc.confidence * (self._confidence_decay ** depth)

                if decayed_conf < self._confidence_threshold:
                    continue

                # Create discovered node
                new_node = self._get_or_create_node(
                    label=disc.label,
                    node_type=disc.discovered_type,
                    depth=depth + 1,
                    confidence=decayed_conf,
                    discovered_by=current.node_id,
                )

                # Create edge
                self._add_edge(
                    source_id=current.node_id,
                    target_id=new_node.node_id,
                    edge_type=disc.relationship,
                    confidence=decayed_conf,
                    evidence=disc.evidence,
                )

                discoveries_all.append(disc)

                # Enqueue for further investigation
                if not new_node.visited and depth + 1 < self._max_depth:
                    new_node.visited = True
                    investigation_path.append(
                        f"[{depth+1}] {disc.discovered_type.value}: {disc.label}"
                    )
                    queue.append((new_node, depth + 1, evidence))

        # Compute stats
        confs = [n.confidence for n in self._nodes.values()]
        conf_range = (min(confs), max(confs)) if confs else (0.0, 0.0)
        max_d = max((n.depth for n in self._nodes.values()), default=0)

        latency = (time.perf_counter() - t0) * 1000

        return InvestigationResult(
            nodes=list(self._nodes.values()),
            edges=self._edges,
            discoveries=discoveries_all,
            total_depth=max_d,
            nodes_visited=nodes_visited,
            nodes_discovered=len(self._nodes),
            confidence_range=conf_range,
            investigation_path=investigation_path,
            latency_ms=latency,
        )

    def get_subgraph(self, node_id: str, max_hops: int = 2) -> dict[str, Any]:
        """Get a subgraph around a specific node."""
        if node_id not in self._nodes:
            return {"error": "node not found"}

        relevant_edges = []
        relevant_node_ids = {node_id}

        for hop in range(max_hops):
            new_ids: set[str] = set()
            for edge in self._edges:
                if edge.source_id in relevant_node_ids:
                    new_ids.add(edge.target_id)
                    relevant_edges.append(edge)
                elif edge.target_id in relevant_node_ids:
                    new_ids.add(edge.source_id)
                    relevant_edges.append(edge)
            relevant_node_ids.update(new_ids)

        nodes = [
            self._nodes[nid].to_dict()
            for nid in relevant_node_ids
            if nid in self._nodes
        ]

        return {
            "center": self._nodes[node_id].to_dict(),
            "nodes": nodes,
            "edges": [e.to_dict() for e in relevant_edges],
        }

    def get_graph_stats(self) -> dict[str, Any]:
        """Get statistics about the investigation graph."""
        if not self._nodes:
            return {"node_count": 0, "edge_count": 0}

        type_counts: dict[str, int] = {}
        for node in self._nodes.values():
            t = node.node_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        edge_type_counts: dict[str, int] = {}
        for edge in self._edges:
            t = edge.edge_type.value
            edge_type_counts[t] = edge_type_counts.get(t, 0) + 1

        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "node_types": type_counts,
            "edge_types": edge_type_counts,
            "max_depth": max((n.depth for n in self._nodes.values()), default=0),
            "avg_confidence": sum(n.confidence for n in self._nodes.values()) / len(self._nodes),
        }
