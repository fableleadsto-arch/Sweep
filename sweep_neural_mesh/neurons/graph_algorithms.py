"""
Graph Algorithms — PageRank, shortest path, community detection.

Implements graph-based reasoning for Sweep's causal and evidence networks:

    PageRank:
        PR(v) = (1-d)/N + d * Σ PR(u)/L(u)
        where d=damping factor, N=nodes, L=outlinks

    Dijkstra's Shortest Path:
        dist[v] = min(dist[u] + weight(u,v))

    Betweenness Centrality:
        C_B(v) = Σ σ_st(v) / σ_st
        where σ_st = shortest paths from s to t

    Community Detection (Label Propagation):
        Each node adopts the most frequent label among neighbors

    Graph Density:
        D = 2|E| / (|V| * (|V|-1))

Used for:
- Causal chain analysis (shortest path between causes)
- Evidence network centrality (which evidence is most important)
- Community detection (which evidence items cluster together)
- PageRank for evidence importance ranking

All operations are logged.
"""
from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("sweep.math.graph")


@dataclass
class GraphNode:
    """A node in the reasoning graph."""
    name: str
    weight: float = 1.0
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """An edge in the reasoning graph."""
    from_node: str
    to_node: str
    weight: float = 1.0
    edge_type: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageRankResult:
    """Result of PageRank computation."""
    rankings: dict[str, float]  # node → rank
    iterations: int
    converged: bool
    damping_factor: float


@dataclass
class ShortestPathResult:
    """Result of shortest path computation."""
    path: list[str]
    total_weight: float
    found: bool
    hop_count: int


@dataclass
class CentralityResult:
    """Result of centrality computation."""
    centrality: dict[str, float]  # node → centrality score
    method: str


@dataclass
class CommunityResult:
    """Result of community detection."""
    communities: dict[str, int]  # node → community_id
    num_communities: int
    modularity: float


class ReasoningGraph:
    """
    Weighted directed graph for reasoning.

    Stores nodes and edges with support for:
    - PageRank importance ranking
    - Shortest path (causal chains)
    - Betweenness centrality (key evidence nodes)
    - Community detection (evidence clustering)
    - Topological sort (dependency ordering)
    """

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, list[GraphEdge]] = {}  # from_node → [edges]
        self._reverse_edges: dict[str, list[GraphEdge]] = {}
        self._total_operations = 0
        logger.info("ReasoningGraph initialized")

    def add_node(self, name: str, weight: float = 1.0, label: str = "", **metadata: Any) -> None:
        """Add a node to the graph."""
        self._nodes[name] = GraphNode(name=name, weight=weight, label=label, metadata=metadata)
        if name not in self._edges:
            self._edges[name] = []
        if name not in self._reverse_edges:
            self._reverse_edges[name] = []
        logger.debug(f"Added node '{name}' (weight={weight})")

    def add_edge(
        self, from_node: str, to_node: str, weight: float = 1.0,
        edge_type: str = "default", **metadata: Any,
    ) -> None:
        """Add a directed edge."""
        if from_node not in self._nodes:
            self.add_node(from_node)
        if to_node not in self._nodes:
            self.add_node(to_node)

        edge = GraphEdge(from_node=from_node, to_node=to_node,
                        weight=weight, edge_type=edge_type, metadata=metadata)
        self._edges[from_node].append(edge)
        self._reverse_edges[to_node].append(edge)
        logger.debug(f"Added edge {from_node} → {to_node} (w={weight})")

    # ════════════════════════════════════════════════════════════════
    # PAGERANK
    # ════════════════════════════════════════════════════════════════

    def pagerank(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tolerance: float = 1e-6,
    ) -> PageRankResult:
        """
        Compute PageRank for all nodes.

        PR(v) = (1-d)/N + d * Σ PR(u)/L(u)

        where d=damping, N=total nodes, L(u)=outlinks from u.
        """
        n = len(self._nodes)
        if n == 0:
            return PageRankResult({}, 0, True, damping)

        # Initialize uniform
        pr = {name: 1.0 / n for name in self._nodes}
        converged = False

        for iteration in range(max_iter):
            new_pr = {}
            diff = 0.0

            for name in self._nodes:
                # Sum of PR from incoming links
                incoming_sum = 0.0
                for edge in self._reverse_edges.get(name, []):
                    source = edge.from_node
                    outlinks = len(self._edges.get(source, []))
                    if outlinks > 0:
                        incoming_sum += pr[source] / outlinks

                # PageRank formula
                new_pr[name] = (1.0 - damping) / n + damping * incoming_sum
                diff += abs(new_pr[name] - pr[name])

            pr = new_pr
            self._total_operations += 1

            if diff < tolerance:
                converged = True
                logger.info(f"PageRank converged after {iteration+1} iterations")
                break

        result = PageRankResult(rankings=pr, iterations=iteration+1,
                               converged=converged, damping_factor=damping)

        # Log top 5
        sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"PageRank top 5: {[(n, round(r, 4)) for n, r in sorted_pr]}")
        return result

    # ════════════════════════════════════════════════════════════════
    # SHORTEST PATH (DIJKSTRA)
    # ════════════════════════════════════════════════════════════════

    def shortest_path(self, start: str, end: str) -> ShortestPathResult:
        """Find shortest path between two nodes using Dijkstra's algorithm."""
        if start not in self._nodes or end not in self._nodes:
            return ShortestPathResult([], 0.0, False, 0)

        distances = {name: float('inf') for name in self._nodes}
        distances[start] = 0.0
        previous: dict[str, str | None] = {name: None for name in self._nodes}
        visited: set[str] = set()

        while True:
            # Pick unvisited node with smallest distance
            unvisited = {n: d for n, d in distances.items() if n not in visited}
            if not unvisited:
                break
            current = min(unvisited, key=unvisited.get)
            if distances[current] == float('inf'):
                break
            if current == end:
                break

            visited.add(current)

            for edge in self._edges.get(current, []):
                neighbor = edge.to_node
                new_dist = distances[current] + edge.weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current

        # Reconstruct path
        path = []
        current: str | None = end
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()

        found = path[0] == start and path[-1] == end
        if found:
            logger.info(f"Shortest path {start}→{end}: {path} (w={distances[end]:.2f})")

        self._total_operations += 1
        return ShortestPathResult(
            path=path if found else [],
            total_weight=distances[end] if found else float('inf'),
            found=found,
            hop_count=max(0, len(path) - 1) if found else 0,
        )

    # ════════════════════════════════════════════════════════════════
    # BETWEENNESS CENTRALITY
    # ════════════════════════════════════════════════════════════════

    def betweenness_centrality(self) -> CentralityResult:
        """
        Compute betweenness centrality for all nodes.

        C_B(v) = Σ σ_st(v) / σ_st

        Nodes with high betweenness are bridges between clusters.
        """
        centrality = {name: 0.0 for name in self._nodes}

        for s in self._nodes:
            # BFS from s to find all shortest paths
            sp_count = {name: 0 for name in self._nodes}
            sp_count[s] = 1
            dist = {name: float('inf') for name in self._nodes}
            dist[s] = 0
            predecessors: dict[str, list[str]] = {name: [] for name in self._nodes}
            queue = deque([s])

            while queue:
                v = queue.popleft()
                for edge in self._edges.get(v, []):
                    w = edge.to_node
                    if dist[w] > dist[v] + 1:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                        predecessors[w] = [v]
                        sp_count[w] = sp_count[v]
                    elif dist[w] == dist[v] + 1:
                        predecessors[w].append(v)
                        sp_count[w] += sp_count[v]

            # Back-propagate dependency
            delta = {name: 0.0 for name in self._nodes}
            nodes_by_dist = sorted(self._nodes, key=lambda n: dist[n], reverse=True)

            for w in nodes_by_dist:
                for v in predecessors.get(w, []):
                    if sp_count[w] > 0:
                        delta[v] += (sp_count[v] / sp_count[w]) * (1 + delta[w])
                if w != s:
                    centrality[w] += delta[w]

        # Normalize
        n = len(self._nodes)
        if n > 2:
            for name in centrality:
                centrality[name] /= (n - 1) * (n - 2)

        self._total_operations += 1
        result = CentralityResult(centrality=centrality, method="betweenness")

        sorted_c = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Betweenness centrality top 5: {[(n, round(c, 4)) for n, c in sorted_c]}")
        return result

    # ════════════════════════════════════════════════════════════════
    # COMMUNITY DETECTION
    # ════════════════════════════════════════════════════════════════

    def label_propagation(
        self,
        max_iter: int = 100,
        seed: int = 42,
    ) -> CommunityResult:
        """
        Community detection via Label Propagation Algorithm.

        Each node adopts the most frequent label among its neighbors.
        Converges to communities where most edges are within-community.
        """
        rng = random.Random(seed)

        # Initialize each node with unique label
        labels = {name: i for i, name in enumerate(self._nodes)}
        self._total_operations += 1

        for iteration in range(max_iter):
            changed = False
            nodes_list = list(self._nodes.keys())
            rng.shuffle(nodes_list)

            for node in nodes_list:
                # Count neighbor labels
                label_counts: dict[int, int] = {}
                for edge in self._edges.get(node, []):
                    lbl = labels[edge.to_node]
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
                for edge in self._reverse_edges.get(node, []):
                    lbl = labels[edge.from_node]
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

                if not label_counts:
                    continue

                # Adopt most frequent label
                best_label = max(label_counts, key=label_counts.get)
                if labels[node] != best_label:
                    labels[node] = best_label
                    changed = True

            if not changed:
                logger.info(f"Label propagation converged after {iteration+1} iterations")
                break

        # Map labels to sequential community IDs
        unique_labels = set(labels.values())
        label_to_id = {lbl: i for i, lbl in enumerate(sorted(unique_labels))}
        communities = {name: label_to_id[lbl] for name, lbl in labels.items()}

        num_communities = len(unique_labels)

        # Compute modularity
        modularity = self._compute_modularity(communities)

        result = CommunityResult(
            communities=communities,
            num_communities=num_communities,
            modularity=modularity,
        )
        logger.info(
            f"Community detection: {num_communities} communities, "
            f"Q={modularity:.4f}"
        )
        return result

    def _compute_modularity(self, communities: dict[str, int]) -> float:
        """Compute Newman modularity Q."""
        m = sum(len(edges) for edges in self._edges.values())
        if m == 0:
            return 0.0

        Q = 0.0
        for node in self._nodes:
            ki = len(self._edges.get(node, []))
            for edge in self._edges.get(node, []):
                kj = len(self._edges.get(edge.to_node, []))
                if communities[node] == communities[edge.to_node]:
                    Q += 1.0 - (ki * kj) / (2.0 * m)

        return Q / (2.0 * m) if m > 0 else 0.0

    # ════════════════════════════════════════════════════════════════
    # TOPOLOGICAL SORT
    # ════════════════════════════════════════════════════════════════

    def topological_sort(self) -> list[str]:
        """
        Topological sort (Kahn's algorithm).

        Returns nodes in dependency order (no node appears before its parents).
        """
        in_degree = {name: 0 for name in self._nodes}
        for name, edges in self._edges.items():
            for edge in edges:
                in_degree[edge.to_node] = in_degree.get(edge.to_node, 0) + 1

        queue = deque([n for n, d in in_degree.items() if d == 0])
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for edge in self._edges.get(node, []):
                in_degree[edge.to_node] -= 1
                if in_degree[edge.to_node] == 0:
                    queue.append(edge.to_node)

        self._total_operations += 1
        if len(result) != len(self._nodes):
            logger.warning("Topological sort: cycle detected, partial ordering returned")
        return result

    # ════════════════════════════════════════════════════════════════
    # GRAPH PROPERTIES
    # ════════════════════════════════════════════════════════════════

    def density(self) -> float:
        """Graph density: D = |E| / (|V| * (|V|-1)) for directed graphs."""
        n = len(self._nodes)
        if n < 2:
            return 0.0
        e = sum(len(edges) for edges in self._edges.values())
        return float(e) / (n * (n - 1))

    def in_degree(self, node: str) -> int:
        """In-degree of a node."""
        return len(self._reverse_edges.get(node, []))

    def out_degree(self, node: str) -> int:
        """Out-degree of a node."""
        return len(self._edges.get(node, []))

    def neighbors(self, node: str) -> list[str]:
        """Get all neighbors (both in and out)."""
        out = [e.to_node for e in self._edges.get(node, [])]
        inp = [e.from_node for e in self._reverse_edges.get(node, [])]
        return list(set(out + inp))

    def reachable_from(self, start: str) -> set[str]:
        """BFS from start to find all reachable nodes."""
        visited: set[str] = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for edge in self._edges.get(node, []):
                if edge.to_node not in visited:
                    queue.append(edge.to_node)
        return visited

    def get_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        return {
            "node_count": len(self._nodes),
            "edge_count": sum(len(edges) for edges in self._edges.values()),
            "density": round(self.density(), 4),
            "total_operations": self._total_operations,
        }
