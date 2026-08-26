"""
Graph Generator — Creates directed graphs with controlled properties.

Generates random directed graphs with:
- Configurable node count (10 to 5000)
- Controlled branching factor, depth, and density
- Random hex identifiers (prevents semantic shortcuts)
- Cycles, disconnected components, duplicate-looking identifiers
- Irrelevant edges, converging/diverging paths
- Multiple routes to the same node
- Easy/Medium/Hard/Extreme difficulty levels
"""
from __future__ import annotations

import random
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Graph:
    """A directed graph with metadata."""
    id: str
    nodes: list[str]
    edges: list[tuple[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def adjacency(self) -> dict[str, list[str]]:
        """Build adjacency list."""
        adj: dict[str, list[str]] = {n: [] for n in self.nodes}
        for src, dst in self.edges:
            if dst in adj:
                adj[src].append(dst)
        return adj

    def reverse_adjacency(self) -> dict[str, list[str]]:
        """Build reverse adjacency list (for ancestor queries)."""
        rev: dict[str, list[str]] = {n: [] for n in self.nodes}
        for src, dst in self.edges:
            if src in rev:
                rev[dst].append(src)
        return rev

    def to_text(self) -> str:
        """Render graph as human-readable text for LLM consumption."""
        lines = ["GRAPH:"]
        lines.append("NODES: " + ", ".join(self.nodes))
        lines.append("EDGES:")
        for src, dst in self.edges:
            lines.append(f"  {src} -> {dst}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "nodes": self.nodes,
            "edges": [[s, d] for s, d in self.edges],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Graph:
        return cls(
            id=d["id"],
            nodes=d["nodes"],
            edges=[tuple(e) for e in d["edges"]],
            metadata=d.get("metadata", {}),
        )


def _generate_hex_id(rng: random.Random, length: int = 6, used: set[str] | None = None) -> str:
    """Generate a random hex-like identifier like A73F91."""
    used = used or set()
    chars = "0123456789ABCDEF"
    while True:
        # Mix uppercase letters and digits to look like hex but aren't always
        parts = []
        for _ in range(length):
            if rng.random() < 0.5:
                parts.append(rng.choice("0123456789"))
            else:
                parts.append(rng.choice("ABCDEF"))
        candidate = "".join(parts)
        if candidate not in used:
            return candidate


def _generate_similar_ids(rng: random.Random, base: str, count: int, used: set[str]) -> list[str]:
    """Generate IDs that look similar to base (distractors)."""
    results = []
    for _ in range(count):
        # Flip 1-2 characters
        chars = list(base)
        n_flips = rng.randint(1, min(2, len(chars)))
        positions = rng.sample(range(len(chars)), n_flips)
        for pos in positions:
            chars[pos] = rng.choice("0123456789ABCDEF")
        candidate = "".join(chars)
        attempts = 0
        while (candidate in used or candidate == base) and attempts < 20:
            chars[pos] = rng.choice("0123456789ABCDEF")
            candidate = "".join(chars)
            attempts += 1
        if candidate not in used and candidate != base:
            results.append(candidate)
            used.add(candidate)
    return results


@dataclass
class DifficultyConfig:
    """Configuration for a difficulty level."""
    name: str
    avg_branching: float    # average out-degree
    depth_factor: float     # controls longest path
    cycle_probability: float
    disconnected_prob: float   # probability of disconnected components
    distractor_ratio: float  # extra nodes/edges as distractors
    min_density: float
    max_density: float


DIFFICULTY_PRESETS: dict[str, DifficultyConfig] = {
    "easy": DifficultyConfig(
        name="easy",
        avg_branching=1.5,
        depth_factor=0.5,
        cycle_probability=0.0,
        disconnected_prob=0.0,
        distractor_ratio=0.0,
        min_density=0.05,
        max_density=0.15,
    ),
    "medium": DifficultyConfig(
        name="medium",
        avg_branching=2.5,
        depth_factor=0.7,
        cycle_probability=0.1,
        disconnected_prob=0.05,
        distractor_ratio=0.1,
        min_density=0.1,
        max_density=0.25,
    ),
    "hard": DifficultyConfig(
        name="hard",
        avg_branching=4.0,
        depth_factor=0.9,
        cycle_probability=0.2,
        disconnected_prob=0.1,
        distractor_ratio=0.2,
        min_density=0.15,
        max_density=0.35,
    ),
    "extreme": DifficultyConfig(
        name="extreme",
        avg_branching=6.0,
        depth_factor=1.0,
        cycle_probability=0.3,
        disconnected_prob=0.15,
        distractor_ratio=0.3,
        min_density=0.2,
        max_density=0.5,
    ),
}


class GraphGenerator:
    """
    Generates directed graphs with controlled properties.

    Usage:
        gen = GraphGenerator(seed=42)
        graph = gen.generate(num_nodes=100, difficulty="hard")
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._graph_counter = 0

    def generate(
        self,
        num_nodes: int = 100,
        difficulty: str = "medium",
        id_length: int = 6,
    ) -> Graph:
        """Generate a single directed graph."""
        self._graph_counter += 1
        cfg = DIFFICULTY_PRESETS.get(difficulty, DIFFICULTY_PRESETS["medium"])
        used_ids: set[str] = set()

        # ── Generate node IDs ──
        nodes = []
        for _ in range(num_nodes):
            nid = _generate_hex_id(self._rng, id_length, used_ids)
            nodes.append(nid)
            used_ids.add(nid)

        # ── Generate similar-looking distractor IDs ──
        distractor_count = int(num_nodes * cfg.distractor_ratio)
        distractors = []
        if distractor_count > 0:
            base_nodes = self._rng.sample(nodes, min(distractor_count, len(nodes)))
            for base in base_nodes:
                sims = _generate_similar_ids(self._rng, base, 1, used_ids)
                distractors.extend(sims)

        # ── Build backbone: a depth-limited tree from a root ──
        edges: list[tuple[str, str]] = []
        root = nodes[0]
        assigned_depth: dict[str, int] = {root: 0}
        children_map: dict[str, list[str]] = {n: [] for n in nodes}

        # BFS assignment
        queue = [root]
        node_idx = 1
        max_depth = max(1, int(num_nodes * cfg.depth_factor * 0.3))

        while queue and node_idx < num_nodes:
            parent = queue.pop(0)
            parent_depth = assigned_depth[parent]
            if parent_depth >= max_depth:
                continue

            # How many children?
            remaining_nodes = num_nodes - node_idx
            remaining_depth = max_depth - parent_depth
            if remaining_depth <= 0:
                break
            max_children = min(
                remaining_nodes,
                max(1, int(self._rng.expovariate(1.0 / cfg.avg_branching)))
            )
            max_children = min(max_children, remaining_nodes)
            if max_children <= 0:
                continue

            for _ in range(max_children):
                if node_idx >= num_nodes:
                    break
                child = nodes[node_idx]
                node_idx += 1
                assigned_depth[child] = parent_depth + 1
                children_map[parent].append(child)
                edges.append((parent, child))
                queue.append(child)

        # ── Add cross-edges to increase connectivity ──
        target_edge_count = int(num_nodes * cfg.avg_branching)
        current_edges = len(edges)
        extra_edges_needed = max(0, target_edge_count - current_edges)

        for _ in range(extra_edges_needed):
            src = self._rng.choice(nodes)
            dst = self._rng.choice(nodes)
            if src != dst and (src, dst) not in set(edges):
                edges.append((src, dst))

        # ── Add cycles if configured ──
        if cfg.cycle_probability > 0:
            num_cycles = max(1, int(num_nodes * cfg.cycle_probability * 0.1))
            edge_set: set[tuple[str, str]] = set(edges)
            adj_cache: dict[str, list[str]] = {n: [] for n in nodes}
            for s, d in edges:
                adj_cache[s].append(d)
            for _ in range(num_cycles):
                if len(edges) < 2:
                    break
                src = self._rng.choice(nodes)
                path_len = self._rng.randint(2, min(5, num_nodes - 1))
                path = [src]
                current = src
                for _ in range(path_len):
                    neighbors = adj_cache.get(current, [])
                    if neighbors:
                        next_node = self._rng.choice(neighbors)
                        path.append(next_node)
                        current = next_node
                    else:
                        break
                if len(path) >= 3:
                    cycle_edge = (path[-1], path[0])
                    if cycle_edge not in edge_set and path[-1] != path[0]:
                        edges.append(cycle_edge)
                        edge_set.add(cycle_edge)
                        adj_cache[path[-1]].append(path[0])

        # ── Add disconnected components ──
        components = []
        if cfg.disconnected_prob > 0 and num_nodes >= 20:
            num_extra_components = max(1, int(num_nodes * cfg.disconnected_prob * 0.05))
            used_for_components = set(nodes[:node_idx if node_idx < num_nodes else num_nodes])
            available = [n for n in nodes if n not in assigned_depth]
            if available:
                for _ in range(num_extra_components):
                    comp_size = min(len(available), self._rng.randint(3, max(3, num_nodes // 10)))
                    if comp_size <= 0:
                        break
                    comp_nodes = self._rng.sample(available, comp_size)
                    for n in comp_nodes:
                        available.remove(n)
                    components.append(comp_nodes)
                    # Connect component in a chain
                    for i in range(len(comp_nodes) - 1):
                        edges.append((comp_nodes[i], comp_nodes[i + 1]))

        # ── Ensure graph is valid ──
        edge_set = set(edges)
        edges = [(s, d) for s, d in edges if s != d and s in used_ids and d in used_ids]

        # ── Metadata ──
        metadata = {
            "difficulty": difficulty,
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "num_distractors": len(distractors),
            "num_disconnected_components": len(components),
            "max_depth": max(assigned_depth.values()) if assigned_depth else 0,
            "avg_branching": len(edges) / max(1, num_nodes),
            "density": len(edges) / max(1, num_nodes * (num_nodes - 1)),
        }

        return Graph(
            id=f"G{self._graph_counter:04d}",
            nodes=nodes + distractors,
            edges=edges,
            metadata=metadata,
        )

    def generate_batch(
        self,
        num_graphs: int = 100,
        num_nodes: int = 100,
        difficulty: str = "medium",
        id_length: int = 6,
    ) -> list[Graph]:
        """Generate multiple graphs."""
        return [
            self.generate(num_nodes, difficulty, id_length)
            for _ in range(num_graphs)
        ]
