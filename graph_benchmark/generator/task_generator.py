"""
Task Generator — Creates reasoning tasks from directed graphs.

8 task types:
  A. Exact-depth BFS: Find all nodes exactly N edges from start
  B. Reachability: Is node A reachable from node B?
  C. Shortest path: Minimum edges between two nodes
  D. Common descendants: Nodes reachable from both A and B
  E. Common ancestors: Nodes that can reach both A and B
  F. Parent reconstruction: All immediate parents of a node
  G. Multi-hop chain: Does A -> B -> C -> D -> E exist?
  H. Contradictory information: Test actual graph vs assumptions
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any

from graph_benchmark.generator.graph_generator import Graph


@dataclass
class Task:
    """A single benchmark task."""
    id: str
    task_type: str
    difficulty: str
    graph_id: str
    prompt: str
    ground_truth: Any
    graph_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "graph_id": self.graph_id,
            "prompt": self.prompt,
            "ground_truth": self.ground_truth,
            "graph_text": self.graph_text,
            "metadata": self.metadata,
        }


class TaskGenerator:
    """
    Generates reasoning tasks from directed graphs.

    Usage:
        gen = TaskGenerator(seed=42)
        tasks = gen.generate_all(graph, num_per_type=10)
    """

    TASK_TYPES = ["bfs", "reachability", "shortest_path", "common_descendants",
                  "common_ancestors", "parent_reconstruction", "multi_hop_chain",
                  "contradictory", "parallel_branches"]

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._task_counter = 0

    def _next_id(self) -> str:
        self._task_counter += 1
        return f"T{self._task_counter:06d}"

    def _bfs(self, graph: Graph, start: str, depth: int) -> set[str]:
        """Find all nodes exactly `depth` edges from `start`."""
        adj = graph.adjacency()
        if start not in adj:
            return set()
        visited: dict[str, int] = {start: 0}
        queue = deque([start])
        result: set[str] = set()
        while queue:
            node = queue.popleft()
            current_depth = visited[node]
            if current_depth == depth:
                result.add(node)
                continue
            if current_depth > depth:
                break
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited[neighbor] = current_depth + 1
                    queue.append(neighbor)
        return result

    def _shortest_path(self, graph: Graph, start: str, end: str) -> int:
        """Find shortest path length between two nodes. Returns -1 if unreachable."""
        if start == end:
            return 0
        adj = graph.adjacency()
        if start not in adj or end not in adj:
            return -1
        visited: set[str] = {start}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            for neighbor in adj.get(node, []):
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return -1

    def _reachable_from(self, graph: Graph, start: str) -> set[str]:
        """Find all nodes reachable from start."""
        adj = graph.adjacency()
        if start not in adj:
            return set()
        visited: set[str] = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    def _can_reach(self, graph: Graph, start: str, end: str) -> bool:
        """Check if end is reachable from start."""
        return end in self._reachable_from(graph, start)

    def _parents(self, graph: Graph, node: str) -> list[str]:
        """Find all immediate parents of node."""
        return [src for src, dst in graph.edges if dst == node]

    def _multi_hop_exists(self, graph: Graph, chain: list[str]) -> bool:
        """Check if a chain of nodes exists as a directed path."""
        for i in range(len(chain) - 1):
            if not self._can_reach(graph, chain[i], chain[i + 1]):
                return False
        return True

    def _sample_node(self, graph: Graph, exclude: str | None = None) -> str:
        """Sample a random node from the graph."""
        candidates = [n for n in graph.nodes if n != exclude and len(n) >= 4]
        if not candidates:
            candidates = graph.nodes
        return self._rng.choice(candidates)

    def _sample_pair(self, graph: Graph) -> tuple[str, str]:
        """Sample two distinct nodes."""
        a = self._sample_node(graph)
        b = self._sample_node(graph, exclude=a)
        return a, b

    def generate_bfs(self, graph: Graph, difficulty: str) -> Task:
        """Task A: Exact-depth BFS."""
        start = self._sample_node(graph)
        depth = self._rng.randint(1, max(1, graph.metadata.get("max_depth", 3)))
        answer = sorted(self._bfs(graph, start, depth))

        prompt = (
            f"GRAPH:\n{graph.to_text()}\n\n"
            f"TASK: Find every node reachable from {start} using exactly {depth} directed edge(s).\n"
            f"Return the node IDs as a sorted comma-separated list. If none, return NONE."
        )

        return Task(
            id=self._next_id(),
            task_type="bfs",
            difficulty=difficulty,
            graph_id=graph.id,
            prompt=prompt,
            ground_truth=answer if answer else "NONE",
            graph_text=graph.to_text(),
            metadata={"start": start, "depth": depth, "expected_count": len(answer)},
        )

    def generate_reachability(self, graph: Graph, difficulty: str) -> Task:
        """Task B: Reachability."""
        a, b = self._sample_pair(graph)
        reachable = self._can_reach(graph, a, b)

        prompt = (
            f"GRAPH:\n{graph.to_text()}\n\n"
            f"TASK: Is node {b} reachable from node {a}?\n"
            f"Answer YES or NO."
        )

        return Task(
            id=self._next_id(),
            task_type="reachability",
            difficulty=difficulty,
            graph_id=graph.id,
            prompt=prompt,
            ground_truth="YES" if reachable else "NO",
            graph_text=graph.to_text(),
            metadata={"source": a, "target": b, "reachable": reachable},
        )

    def generate_shortest_path(self, graph: Graph, difficulty: str) -> Task:
        """Task C: Shortest path."""
        a, b = self._sample_pair(graph)
        dist = self._shortest_path(graph, a, b)

        prompt = (
            f"GRAPH:\n{graph.to_text()}\n\n"
            f"TASK: Find the minimum number of directed edges needed to go from {a} to {b}.\n"
            f"Return the distance as an integer. If unreachable, return -1."
        )

        return Task(
            id=self._next_id(),
            task_type="shortest_path",
            difficulty=difficulty,
            graph_id=graph.id,
            prompt=prompt,
            ground_truth=dist,
            graph_text=graph.to_text(),
            metadata={"source": a, "target": b, "distance": dist},
        )

    def generate_common_descendants(self, graph: Graph, difficulty: str) -> Task:
        """Task D: Common descendants."""
        a, b = self._sample_pair(graph)
        desc_a = self._reachable_from(graph, a)
        desc_b = self._reachable_from(graph, b)
        common = sorted(desc_a & desc_b)

        prompt = (
            f"GRAPH:\n{graph.to_text()}\n\n"
            f"TASK: Find all nodes reachable from BOTH {a} AND {b}.\n"
            f"Return the node IDs as a sorted comma-separated list. If none, return NONE."
        )

        return Task(
            id=self._next_id(),
            task_type="common_descendants",
            difficulty=difficulty,
            graph_id=graph.id,
            prompt=prompt,
            ground_truth=common if common else "NONE",
            graph_text=graph.to_text(),
            metadata={
                "node_a": a, "node_b": b,
                "descendants_a": len(desc_a), "descendants_b": len(desc_b),
                "common_count": len(common),
            },
        )

    def generate_common_ancestors(self, graph: Graph, difficulty: str) -> Task:
        """Task E: Common ancestors."""
        a, b = self._sample_pair(graph)
        rev = graph.reverse_adjacency()
        ancestors_a = self._reachable_from_rev(rev, a)
        ancestors_b = self._reachable_from_rev(rev, b)
        common = sorted(ancestors_a & ancestors_b)

        prompt = (
            f"GRAPH:\n{graph.to_text()}\n\n"
            f"TASK: Find all nodes from which BOTH {a} AND {b} are reachable.\n"
            f"Return the node IDs as a sorted comma-separated list. If none, return NONE."
        )

        return Task(
            id=self._next_id(),
            task_type="common_ancestors",
            difficulty=difficulty,
            graph_id=graph.id,
            prompt=prompt,
            ground_truth=common if common else "NONE",
            graph_text=graph.to_text(),
            metadata={
                "node_a": a, "node_b": b,
                "ancestors_a": len(ancestors_a), "ancestors_b": len(ancestors_b),
                "common_count": len(common),
            },
        )

    def _reachable_from_rev(self, rev_adj: dict[str, list[str]], start: str) -> set[str]:
        """Find all nodes that can reach start (using reverse adjacency)."""
        if start not in rev_adj:
            return set()
        visited: set[str] = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in rev_adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited

    def generate_parent_reconstruction(self, graph: Graph, difficulty: str) -> Task:
        """Task F: Parent reconstruction."""
        target = self._sample_node(graph)
        parents = sorted(self._parents(graph, target))

        prompt = (
            f"GRAPH:\n{graph.to_text()}\n\n"
            f"TASK: Find all immediate parents of node {target}.\n"
            f"(A node P is an immediate parent if there is a directed edge from P to {target}.)\n"
            f"Return the node IDs as a sorted comma-separated list. If none, return NONE."
        )

        return Task(
            id=self._next_id(),
            task_type="parent_reconstruction",
            difficulty=difficulty,
            graph_id=graph.id,
            prompt=prompt,
            ground_truth=parents if parents else "NONE",
            graph_text=graph.to_text(),
            metadata={"target": target, "parent_count": len(parents)},
        )

    def generate_multi_hop_chain(self, graph: Graph, difficulty: str) -> Task:
        """Task G: Multi-hop chain existence."""
        chain_length = self._rng.randint(3, 5)
        # Try to find an actual chain first
        adj = graph.adjacency()
        actual_chain = self._find_chain(graph, chain_length)

        if actual_chain and self._rng.random() < 0.5:
            chain = actual_chain
            exists = True
        else:
            # Create a chain that may or may not exist
            chain = [self._sample_node(graph) for _ in range(chain_length)]
            if self._rng.random() < 0.3 and actual_chain:
                chain = actual_chain
                exists = True
            else:
                exists = self._multi_hop_exists(graph, chain)

        chain_str = " -> ".join(chain)

        prompt = (
            f"GRAPH:\n{graph.to_text()}\n\n"
            f"TASK: Does the directed path {chain_str} exist in the graph?\n"
            f"Answer YES or NO."
        )

        return Task(
            id=self._next_id(),
            task_type="multi_hop_chain",
            difficulty=difficulty,
            graph_id=graph.id,
            prompt=prompt,
            ground_truth="YES" if exists else "NO",
            graph_text=graph.to_text(),
            metadata={"chain": chain, "exists": exists},
        )

    def _find_chain(self, graph: Graph, length: int) -> list[str] | None:
        """Try to find an actual directed chain of given length."""
        adj = graph.adjacency()
        for _ in range(50):
            start = self._sample_node(graph)
            chain = [start]
            current = start
            for _ in range(length - 1):
                neighbors = adj.get(current, [])
                if not neighbors:
                    break
                next_node = self._rng.choice(neighbors)
                if next_node in chain:
                    break
                chain.append(next_node)
                current = next_node
            if len(chain) >= length:
                return chain[:length]
        return None

    def generate_contradictory(self, graph: Graph, difficulty: str) -> Task:
        """Task H: Contradictory graph information."""
        # Present a modified version of the graph with some edges flipped/added
        # and ask which version is correct
        adj = graph.adjacency()

        # Pick a real edge
        if not graph.edges:
            # Fallback: just ask about reachability
            return self.generate_reachability(graph, difficulty)

        real_src, real_dst = self._rng.choice(graph.edges)

        # Create a fake edge
        fake_src = self._sample_node(graph)
        fake_dst = self._sample_node(graph)
        while (fake_src, fake_dst) in set(graph.edges) or fake_src == fake_dst:
            fake_src = self._sample_node(graph)
            fake_dst = self._sample_node(graph, exclude=fake_src)

        prompt = (
            f"GRAPH:\n{graph.to_text()}\n\n"
            f"TASK: Consider the following two claims about the graph above:\n"
            f"Claim 1: There is an edge from {real_src} to {real_dst}\n"
            f"Claim 2: There is an edge from {fake_src} to {fake_dst}\n\n"
            f"Which claim is correct according to the graph? Answer CLAIM1, CLAIM2, BOTH, or NEITHER."
        )

        has_real = (real_src, real_dst) in set(graph.edges)
        has_fake = (fake_src, fake_dst) in set(graph.edges)
        if has_real and has_fake:
            answer = "BOTH"
        elif has_real:
            answer = "CLAIM1"
        elif has_fake:
            answer = "CLAIM2"
        else:
            answer = "NEITHER"

        return Task(
            id=self._next_id(),
            task_type="contradictory",
            difficulty=difficulty,
            graph_id=graph.id,
            prompt=prompt,
            ground_truth=answer,
            graph_text=graph.to_text(),
            metadata={
                "real_edge": [real_src, real_dst],
                "fake_edge": [fake_src, fake_dst],
                "answer": answer,
            },
        )

    def generate_parallel_branches(self, graph: Graph, difficulty: str) -> Task:
        """Task I: Parallel branch aggregation.

        Tests whether independent graph branches can be processed independently
        and results combined. Measures aggregation across parallel reasoning paths.
        """
        adj = graph.adjacency()
        start = self._sample_node(graph)
        depth = self._rng.randint(2, max(2, min(4, graph.metadata.get("max_depth", 3))))

        result_a = sorted(self._bfs(graph, start, depth))
        result_b = sorted(self._bfs(graph, start, depth + 1))
        combined = sorted(set(result_a) | set(result_b))

        prompt = (
            f"GRAPH:\n{graph.to_text()}\n\n"
            f"TASK: Starting from node {start}, find ALL nodes reachable using exactly "
            f"{depth} OR exactly {depth + 1} directed edges.\n"
            f"Combine results from both depths. Return the node IDs as a sorted comma-separated list. "
            f"If none, return NONE."
        )

        return Task(
            id=self._next_id(),
            task_type="parallel_branches",
            difficulty=difficulty,
            graph_id=graph.id,
            prompt=prompt,
            ground_truth=combined if combined else "NONE",
            graph_text=graph.to_text(),
            metadata={
                "start": start, "depth_a": depth, "depth_b": depth + 1,
                "count_a": len(result_a), "count_b": len(result_b),
                "combined_count": len(combined),
            },
        )

    def generate_all(
        self,
        graph: Graph,
        tasks_per_type: int = 5,
    ) -> list[Task]:
        """Generate all task types for a given graph."""
        difficulty = graph.metadata.get("difficulty", "medium")
        generators = {
            "bfs": self.generate_bfs,
            "reachability": self.generate_reachability,
            "shortest_path": self.generate_shortest_path,
            "common_descendants": self.generate_common_descendants,
            "common_ancestors": self.generate_common_ancestors,
            "parent_reconstruction": self.generate_parent_reconstruction,
            "multi_hop_chain": self.generate_multi_hop_chain,
            "contradictory": self.generate_contradictory,
            "parallel_branches": self.generate_parallel_branches,
        }
        tasks = []
        for task_type, gen_func in generators.items():
            for _ in range(tasks_per_type):
                tasks.append(gen_func(graph, difficulty))
        return tasks

    def generate_dataset(
        self,
        num_graphs: int = 100,
        num_nodes: int = 100,
        difficulty: str = "medium",
        tasks_per_type: int = 5,
        id_length: int = 6,
    ) -> list[Task]:
        """Generate a full dataset of tasks from multiple graphs."""
        from graph_benchmark.generator.graph_generator import GraphGenerator
        graph_gen = GraphGenerator(seed=self._rng.randint(0, 2**31))

        all_tasks = []
        for _ in range(num_graphs):
            graph = graph_gen.generate(num_nodes, difficulty, id_length)
            tasks = self.generate_all(graph, tasks_per_type)
            all_tasks.extend(tasks)
        return all_tasks
