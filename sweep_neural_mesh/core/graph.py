"""
MeshGraph — a directed computation graph of NeuralNodes.

The MeshGraph is the task-specific blueprint the ExecutionEngine
executes. It is constructed dynamically per analysis request:

    Image
      ↓
    VisionEncoder (Node A)
      ↓
    ObjectDetector (Node B) ──→ FusionNode (Node D)
                                  ↑
    AudioEncoder (Node C) ────────┘
      ↓
    Result

Nodes and edges are added dynamically. The graph validates that
all edges connect compatible modalities.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .node import Modality, NeuralNode, NodeResult


@dataclass
class Edge:
    """A directed edge between two nodes in the mesh."""
    from_node_id: str
    to_node_id: str
    from_modality: Modality
    to_modality: Modality
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __repr__(self) -> str:
        return (
            f"Edge({self.from_node_id} → {self.to_node_id}, "
            f"{self.from_modality.value}→{self.to_modality.value})"
        )


class MeshGraph:
    """
    A directed acyclic computation graph for the Neural Mesh.

    Constructed dynamically for each task. The graph specifies which
    nodes execute and how data flows between them. The ExecutionEngine
    topologically sorts and executes the graph.
    """

    def __init__(self, graph_id: str | None = None, name: str = "") -> None:
        self.graph_id = graph_id or str(uuid.uuid4())[:12]
        self.name = name or f"graph-{self.graph_id}"
        self._nodes: dict[str, NeuralNode] = {}
        self._edges: list[Edge] = []
        self._adjacency: dict[str, list[str]] = {}
        self._reverse_adj: dict[str, list[str]] = {}
        self._execution_order: list[str] | None = None
        self._results: dict[str, NodeResult] = {}

    # -- Graph construction --

    def add_node(self, node: NeuralNode) -> None:
        """Add a node to the graph."""
        self._nodes[node.node_id] = node
        self._adjacency.setdefault(node.node_id, [])
        self._reverse_adj.setdefault(node.node_id, [])
        self._execution_order = None  # invalidate cache

    def add_edge(self, from_node: NeuralNode, to_node: NeuralNode) -> Edge:
        """Add a directed edge. Validates modality compatibility."""
        edge = Edge(
            from_node_id=from_node.node_id,
            to_node_id=to_node.node_id,
            from_modality=from_node.schema.output_modalities[0] if from_node.schema.output_modalities else Modality.TENSOR,
            to_modality=to_node.schema.input_modalities[0] if to_node.schema.input_modalities else Modality.TENSOR,
        )
        self._edges.append(edge)
        self._adjacency.setdefault(from_node.node_id, []).append(to_node.node_id)
        self._reverse_adj.setdefault(to_node.node_id, []).append(from_node.node_id)
        self._execution_order = None
        return edge

    # -- Topology --

    def _compute_execution_order(self) -> list[str]:
        """Topological sort using Kahn's algorithm."""
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for nid, targets in self._adjacency.items():
            for t in targets:
                in_degree[t] = in_degree.get(t, 0) + 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[str] = []
        while queue:
            queue.sort()  # deterministic ordering
            nid = queue.pop(0)
            order.append(nid)
            for target in self._adjacency.get(nid, []):
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)

        if len(order) != len(self._nodes):
            raise ValueError(
                f"Cycle detected in graph: {self.graph_id}. "
                f"Order has {len(order)} nodes but graph has {len(self._nodes)}."
            )
        return order

    @property
    def execution_order(self) -> list[str]:
        if self._execution_order is None:
            self._execution_order = self._compute_execution_order()
        return list(self._execution_order)

    def roots(self) -> list[NeuralNode]:
        """Nodes with no incoming edges (entry points)."""
        root_ids = [
            nid for nid in self._nodes
            if not self._reverse_adj.get(nid, [])
        ]
        return [self._nodes[nid] for nid in root_ids]

    def remove_node(self, node_id: str) -> NeuralNode | None:
        """Remove a node and all its edges. Returns the removed node or None."""
        if node_id not in self._nodes:
            return None
        node = self._nodes.pop(node_id)
        # Remove all edges involving this node
        self._edges = [
            e for e in self._edges
            if e.from_node_id != node_id and e.to_node_id != node_id
        ]
        # Clean adjacency lists
        self._adjacency.pop(node_id, None)
        self._reverse_adj.pop(node_id, None)
        for targets in self._adjacency.values():
            while node_id in targets:
                targets.remove(node_id)
        for sources in self._reverse_adj.values():
            while node_id in sources:
                sources.remove(node_id)
        self._execution_order = None
        return node

    def leaves(self) -> list[NeuralNode]:
        """Nodes with no outgoing edges (exit points)."""
        leaf_ids = [
            nid for nid in self._nodes
            if not self._adjacency.get(nid, [])
        ]
        return [self._nodes[nid] for nid in leaf_ids]

    def predecessors(self, node_id: str) -> list[NeuralNode]:
        return [self._nodes[nid] for nid in self._reverse_adj.get(node_id, [])]

    def successors(self, node_id: str) -> list[NeuralNode]:
        return [self._nodes[nid] for nid in self._adjacency.get(node_id, [])]

    def store_result(self, node_id: str, result: NodeResult) -> None:
        self._results[node_id] = result

    def get_result(self, node_id: str) -> NodeResult | None:
        return self._results.get(node_id)

    @property
    def nodes(self) -> list[NeuralNode]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[Edge]:
        return list(self._edges)

    @property
    def size(self) -> int:
        return len(self._nodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [
                {"from": e.from_node_id, "to": e.to_node_id}
                for e in self._edges
            ],
            "execution_order": self.execution_order,
        }

    def __repr__(self) -> str:
        return (
            f"MeshGraph(id={self.graph_id}, name={self.name}, "
            f"nodes={self.size}, edges={len(self._edges)})"
        )
