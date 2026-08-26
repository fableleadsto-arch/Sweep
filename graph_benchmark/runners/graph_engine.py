"""
Graph Reasoning Engine — Specialized graph traversal + neural mesh reasoning.

This module separates graph algorithm execution from neural mesh reasoning:
1. Graph Parsing: Convert text graphs to adjacency structures
2. Graph Algorithms: BFS, shortest path, reachability, ancestors, descendants
3. Neural Mesh Reasoning: Use Sweep's mesh for contradictory/distractor detection

The key insight: graph problems require deterministic algorithm execution,
not probabilistic claim-evidence evaluation. The neural mesh adds value on
top of algorithms for handling ambiguity, distractors, and contradictions.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Any

from sweep_neural_mesh.neurons.cortex import ReasoningCortex


@dataclass
class GraphStructure:
    """Parsed graph structure."""
    nodes: list[str]
    edges: list[tuple[str, str]]
    adj: dict[str, list[str]]
    rev_adj: dict[str, list[str]]


@dataclass
class GraphAnswer:
    """Answer from graph algorithm."""
    value: Any
    confidence: float
    reasoning: str
    method: str  # which algorithm was used


class GraphReasoningEngine:
    """
    Combines graph algorithms with neural mesh reasoning.

    For deterministic tasks (BFS, shortest path, reachability):
        Uses graph algorithms directly.
    For ambiguous tasks (contradictory info):
        Uses neural mesh reasoning.

    Args:
        use_neural_mesh: Whether to use the neural mesh for ambiguous cases.
    """

    def __init__(self, use_neural_mesh: bool = True) -> None:
        self._use_neural_mesh = use_neural_mesh
        if use_neural_mesh:
            self._cortex = ReasoningCortex()

    def parse_graph(self, graph_text: str) -> GraphStructure:
        """Parse graph text into a GraphStructure."""
        nodes: list[str] = []
        edges: list[tuple[str, str]] = []

        for line in graph_text.split("\n"):
            line = line.strip()
            if line.startswith("NODES:"):
                raw = line.replace("NODES:", "").strip()
                nodes = [n.strip() for n in raw.split(",") if n.strip()]
            elif "->" in line and not line.startswith("GRAPH"):
                parts = line.split("->")
                if len(parts) == 2:
                    src = parts[0].strip()
                    dst = parts[1].strip()
                    if src and dst:
                        edges.append((src, dst))

        adj: dict[str, list[str]] = {n: [] for n in nodes}
        rev_adj: dict[str, list[str]] = {n: [] for n in nodes}
        for src, dst in edges:
            if src in adj and dst in adj:
                adj[src].append(dst)
                rev_adj[dst].append(src)

        return GraphStructure(nodes=nodes, edges=edges, adj=adj, rev_adj=rev_adj)

    def bfs(self, graph: GraphStructure, start: str, depth: int) -> GraphAnswer:
        """Find all nodes exactly `depth` edges from `start`."""
        if start not in graph.adj:
            return GraphAnswer([], 0.0, f"Start node {start} not found", "bfs")

        visited: dict[str, int] = {start: 0}
        queue = deque([start])
        result: list[str] = []

        while queue:
            node = queue.popleft()
            current_depth = visited[node]
            if current_depth == depth:
                result.append(node)
                continue
            if current_depth > depth:
                break
            for neighbor in graph.adj.get(node, []):
                if neighbor not in visited:
                    visited[neighbor] = current_depth + 1
                    queue.append(neighbor)

        sorted_result = sorted(result)
        return GraphAnswer(
            value=sorted_result if sorted_result else "NONE",
            confidence=1.0,
            reasoning=f"Found {len(sorted_result)} nodes at exact depth {depth} from {start}",
            method="bfs",
        )

    def shortest_path(self, graph: GraphStructure, start: str, end: str) -> GraphAnswer:
        """Find shortest path between two nodes. Returns -1 if unreachable."""
        if start == end:
            return GraphAnswer(0, 1.0, f"Start and end are the same: {start}", "shortest_path")
        if start not in graph.adj or end not in graph.adj:
            return GraphAnswer(-1, 1.0, f"Node not in graph", "shortest_path")

        visited: set[str] = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for neighbor in graph.adj.get(node, []):
                if neighbor == end:
                    return GraphAnswer(
                        dist + 1, 1.0,
                        f"Shortest path from {start} to {end} is {dist + 1} edges",
                        "shortest_path",
                    )
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return GraphAnswer(-1, 1.0, f"No path from {start} to {end}", "shortest_path")

    def reachability(self, graph: GraphStructure, source: str, target: str) -> GraphAnswer:
        """Check if target is reachable from source."""
        if source not in graph.adj or target not in graph.adj:
            return GraphAnswer("NO", 1.0, "Node not in graph", "reachability")

        visited: set[str] = set()
        queue = deque([source])
        while queue:
            node = queue.popleft()
            if node == target:
                return GraphAnswer(
                    "YES", 1.0,
                    f"Node {target} is reachable from {source}",
                    "reachability",
                )
            for neighbor in graph.adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return GraphAnswer(
            "NO", 1.0,
            f"Node {target} is NOT reachable from {source}",
            "reachability",
        )

    def common_descendants(self, graph: GraphStructure, a: str, b: str) -> GraphAnswer:
        """Find nodes reachable from both a and b."""
        desc_a = self._reachable_from(graph, a)
        desc_b = self._reachable_from(graph, b)
        common = sorted(desc_a & desc_b)
        return GraphAnswer(
            common if common else "NONE",
            1.0,
            f"Found {len(common)} common descendants of {a} and {b}",
            "common_descendants",
        )

    def common_ancestors(self, graph: GraphStructure, a: str, b: str) -> GraphAnswer:
        """Find nodes that can reach both a and b."""
        anc_a = self._ancestors_of(graph, a)
        anc_b = self._ancestors_of(graph, b)
        common = sorted(anc_a & anc_b)
        return GraphAnswer(
            common if common else "NONE",
            1.0,
            f"Found {len(common)} common ancestors of {a} and {b}",
            "common_ancestors",
        )

    def parent_reconstruction(self, graph: GraphStructure, target: str) -> GraphAnswer:
        """Find all immediate parents of a node."""
        parents = sorted([src for src, dst in graph.edges if dst == target])
        return GraphAnswer(
            parents if parents else "NONE",
            1.0,
            f"Found {len(parents)} parents of {target}",
            "parent_reconstruction",
        )

    def multi_hop_chain(self, graph: GraphStructure, chain: list[str]) -> GraphAnswer:
        """Check if a chain of nodes exists as a directed path."""
        for i in range(len(chain) - 1):
            result = self.reachability(graph, chain[i], chain[i + 1])
            if result.value == "NO":
                return GraphAnswer(
                    "NO", 1.0,
                    f"Chain broken: {chain[i]} cannot reach {chain[i+1]}",
                    "multi_hop_chain",
                )
        return GraphAnswer(
            "YES", 1.0,
            f"Complete chain exists: {' -> '.join(chain)}",
            "multi_hop_chain",
        )

    def contradictory(self, graph: GraphStructure, query: str) -> GraphAnswer:
        """
        Handle contradictory information tasks.

        Uses graph structure to validate each claim directly.
        """
        claim1, claim2 = self._extract_claims(query)
        if not claim1 or not claim2:
            return GraphAnswer("NEITHER", 0.3, "Could not parse claims", "contradictory")

        claim1_valid = self._validate_claim(graph, claim1)
        claim2_valid = self._validate_claim(graph, claim2)

        if claim1_valid and not claim2_valid:
            return GraphAnswer("CLAIM1", 0.9, "Claim 1 is supported by graph", "contradictory")
        if claim2_valid and not claim1_valid:
            return GraphAnswer("CLAIM2", 0.9, "Claim 2 is supported by graph", "contradictory")
        if claim1_valid and claim2_valid:
            return GraphAnswer("BOTH", 0.8, "Both claims are supported", "contradictory")
        return GraphAnswer("NEITHER", 0.7, "Neither claim is supported", "contradictory")

    def _reachable_from(self, graph: GraphStructure, start: str) -> set[str]:
        """Find all nodes reachable from start."""
        if start not in graph.adj:
            return set()
        visited: set[str] = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph.adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    def _ancestors_of(self, graph: GraphStructure, node: str) -> set[str]:
        """Find all nodes that can reach `node`."""
        if node not in graph.rev_adj:
            return set()
        visited: set[str] = set()
        queue = deque([node])
        while queue:
            current = queue.popleft()
            for parent in graph.rev_adj.get(current, []):
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
        return visited

    def _extract_claims(self, query: str) -> tuple[str, str]:
        """Extract claim1 and claim2 from contradictory task query."""
        claim1 = ""
        claim2 = ""
        for line in query.split("\n"):
            line = line.strip()
            m1 = re.match(r'Claim\s*1:\s*(.+)', line)
            m2 = re.match(r'Claim\s*2:\s*(.+)', line)
            if m1:
                claim1 = m1.group(1).strip()
            elif m2:
                claim2 = m2.group(1).strip()
        return claim1, claim2

    def _validate_claim(self, graph: GraphStructure, claim: str) -> bool:
        """Check if a claim is consistent with the graph structure."""
        # "There is an edge from X to Y"
        edge_from_to = re.search(r'edge from\s+(\w+)\s+to\s+(\w+)', claim)
        if edge_from_to:
            src, dst = edge_from_to.group(1), edge_from_to.group(2)
            return dst in graph.adj.get(src, [])

        # "X -> Y exists"
        edge_match = re.search(r'(\w+)\s*->\s*(\w+)', claim)
        if edge_match:
            src, dst = edge_match.group(1), edge_match.group(2)
            return dst in graph.adj.get(src, [])

        # Parse "X is reachable from Y" claims
        reach_match = re.search(r'(\w+)\s+is reachable from\s+(\w+)', claim)
        if reach_match:
            target, source = reach_match.group(1), reach_match.group(2)
            return self.reachability(graph, source, target).value == "YES"

        # Parse "X can reach Y" claims
        reach_match2 = re.search(r'(\w+)\s+can reach\s+(\w+)', claim)
        if reach_match2:
            source, target = reach_match2.group(1), reach_match2.group(2)
            return self.reachability(graph, source, target).value == "YES"

        # Parse "X is a parent of Y" claims
        parent_match = re.search(r'(\w+)\s+is a parent of\s+(\w+)', claim)
        if parent_match:
            parent, child = parent_match.group(1), parent_match.group(2)
            return parent in [s for s, d in graph.edges if d == child]

        # Parse "X and Y share a common descendant Z" claims
        common_match = re.search(r'(\w+)\s+and\s+(\w+)\s+share.*common.*descendant.*?(\w+)', claim)
        if common_match:
            a, b, z = common_match.group(1), common_match.group(2), common_match.group(3)
            desc_a = self._reachable_from(graph, a)
            desc_b = self._reachable_from(graph, b)
            return z in desc_a and z in desc_b

        return False

    def solve(self, prompt: str, graph_text: str) -> GraphAnswer:
        """
        Solve a graph reasoning task.

        Parses the prompt to determine task type, then applies the right algorithm.
        """
        graph = self.parse_graph(graph_text)

        # parallel_branches: "exactly N OR exactly M directed edges"
        if "OR exactly" in prompt and "directed edges" in prompt.lower():
            nums = re.findall(r'exactly\s+(\d+)\s+OR\s+exactly\s+(\d+)', prompt)
            if nums:
                d1, d2 = int(nums[0][0]), int(nums[0][1])
                start_match = re.search(r'from node\s+(\w+)', prompt)
                if start_match:
                    start = start_match.group(1)
                    r1 = self.bfs(graph, start, d1)
                    r2 = self.bfs(graph, start, d2)
                    set1 = set(r1.value) if isinstance(r1.value, list) else set()
                    set2 = set(r2.value) if isinstance(r2.value, list) else set()
                    combined = sorted(set1 | set2)
                    return GraphAnswer(
                        value=combined if combined else "NONE",
                        confidence=1.0,
                        reasoning=f"Union of depth {d1} ({len(set1)} nodes) and depth {d2} ({len(set2)} nodes) = {len(combined)} nodes",
                        method="parallel_branches",
                    )

        if "exactly" in prompt.lower() and "directed edge" in prompt.lower():
            match = re.search(r'exactly\s+(\d+)\s+directed edge', prompt)
            if match:
                depth = int(match.group(1))
                start_match = re.search(r'from\s+(\w+)', prompt)
                if start_match:
                    start = start_match.group(1)
                    return self.bfs(graph, start, depth)

        if "minimum number of directed edges" in prompt.lower():
            m = re.search(r'from\s+(\w+)\s+to\s+(\w+)', prompt)
            if m:
                return self.shortest_path(graph, m.group(1), m.group(2))

        if "reachable from" in prompt.lower() and ("YES or NO" in prompt or "yes or no" in prompt.lower()):
            m = re.search(r'Is node\s+(\w+)\s+reachable from node\s+(\w+)', prompt)
            if m:
                return self.reachability(graph, m.group(2), m.group(1))

        # common_descendants: "Find all nodes reachable from BOTH X AND Y"
        if "BOTH" in prompt and "reachable from" in prompt.lower():
            m = re.search(r'from BOTH\s+(\w+)\s+AND\s+(\w+)', prompt)
            if m:
                return self.common_descendants(graph, m.group(1), m.group(2))

        # common_ancestors: "Find all nodes from which BOTH X AND Y are reachable"
        if "from which BOTH" in prompt or "from which both" in prompt.lower():
            m = re.search(r'from which BOTH\s+(\w+)\s+AND\s+(\w+)\s+are reachable', prompt)
            if not m:
                m = re.search(r'from which both\s+(\w+)\s+AND\s+(\w+)\s+are reachable', prompt)
            if m:
                return self.common_ancestors(graph, m.group(1), m.group(2))

        if "immediate parents" in prompt.lower() or "parent reconstruction" in prompt.lower():
            m = re.search(r'parents of node\s+(\w+)', prompt)
            if not m:
                m = re.search(r'parents of\s+(\w+)', prompt)
            if m:
                return self.parent_reconstruction(graph, m.group(1))

        if "directed path" in prompt.lower() or "multi-hop chain" in prompt.lower():
            # Extract chain from "Does the directed path X -> Y -> Z exist"
            path_match = re.search(r'directed path\s+(.+?)\s+exist', prompt, re.IGNORECASE)
            if path_match:
                chain_str = path_match.group(1)
                chain_match = re.findall(r'\b([0-9A-F]{4,8})\b', chain_str)
                if len(chain_match) >= 2:
                    return self.multi_hop_chain(graph, chain_match)

        if "contradictory" in prompt.lower() or "claim 1" in prompt.lower():
            return self.contradictory(graph, prompt)

        # Fallback: try to use neural mesh
        if self._use_neural_mesh:
            result = self._cortex.reason(query=prompt, evidence=[graph_text])
            return GraphAnswer(
                value=result.decision,
                confidence=result.confidence,
                reasoning=result.reasoning,
                method="neural_mesh_fallback",
            )

        return GraphAnswer("UNSOLVED", 0.0, "Could not determine task type", "unknown")
