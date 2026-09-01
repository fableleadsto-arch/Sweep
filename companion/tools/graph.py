"""Graph / network capability — NetworkX.

Analyzes graph structure: nodes, edges, degrees, connected components,
shortest paths, centrality and cycles. Lazy import.
"""

from __future__ import annotations

import re
from typing import Any


def run_graph(payload: dict[str, Any]) -> dict[str, Any]:
    """Analyze a graph described by edges / nodes."""
    data = payload.get("data")
    params = payload.get("params") or {}

    nx = _load_networkx()

    edges = _extract_edges(data)
    if not edges:
        raise ValueError(
            "No edges found. Send `data` as a list of [source, target] pairs, "
            "a dict with an `edges` list, or an adjacency list."
        )

    graph = nx.Graph()
    graph.add_edges_from(edges)
    directed = bool(params.get("directed") or _task_mentions(payload.get("task") or "", ("digraph", "directed", "one-way")))
    if directed:
        graph = nx.DiGraph(graph.edges())

    operation = str(params.get("operation") or "summary").lower()

    if operation == "shortest-path":
        source = str(params.get("from") or params.get("source") or "")
        target = str(params.get("to") or params.get("target") or "")
        if not source or not target:
            raise ValueError("shortest-path needs params.from and params.to (node labels).")
        try:
            path = nx.shortest_path(graph, source=source, target=target)
            length = nx.shortest_path_length(graph, source=source, target=target)
            return {
                "result": {"path": list(path), "length": int(length)},
                "summary": f"Shortest path {source} → {target}: {' → '.join(str(p) for p in path)} ({length} hops).",
                "libraries_used": ["networkx"],
            }
        except nx.NetworkXNoPath as exc:
            raise ValueError(f"No path between {source} and {target}.") from exc

    if operation == "components":
        components = list(nx.connected_components(graph))
        sizes = sorted((len(c) for c in components), reverse=True)
        return {
            "result": {
                "connected_components": len(components),
                "component_sizes": sizes,
                "largest_component_size": sizes[0] if sizes else 0,
            },
            "summary": f"{len(components)} connected component(s); largest has {sizes[0] if sizes else 0} nodes.",
            "libraries_used": ["networkx"],
        }

    if operation == "centrality":
        degree = nx.degree_centrality(graph)
        top = sorted(degree.items(), key=lambda kv: kv[1], reverse=True)[:5]
        return {
            "result": {
                "top_degree_centrality": [(str(k), round(float(v), 4)) for k, v in top],
                "max_degree": int(max(dict(graph.degree()).values(), default=0)),
            },
            "summary": "Most central nodes: " + ", ".join(str(k) for k, _ in top) + ".",
            "libraries_used": ["networkx"],
        }

    if operation == "cycles":
        cycles = list(nx.simple_cycles(graph) if graph.is_directed() else nx.cycle_basis(graph))
        return {
            "result": {
                "cycle_count": len(cycles),
                "cycles": [list(c) for c in cycles[:10]],
                "trivial_cycles": int(nx.number_of_selfloops(graph)),
            },
            "summary": f"Found {len(cycles)} cycle(s) in the graph.",
            "libraries_used": ["networkx"],
        }

    # Default summary.
    degree_sequence = dict(graph.degree())
    return {
        "result": {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
            "directed": graph.is_directed(),
            "average_degree": round(float(sum(degree_sequence.values()) / max(1, graph.number_of_nodes())), 4),
            "max_degree": int(max(degree_sequence.values(), default=0)),
            "isolated_nodes": int(sum(1 for d in degree_sequence.values() if d == 0)),
            "density": round(float(nx.density(graph)), 4),
        },
        "summary": (
            f"Graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} "
            f"edges (density {nx.density(graph):.3f})."
        ),
        "libraries_used": ["networkx"],
    }


def _load_networkx():
    from .common import load

    return load("networkx")


def _extract_edges(data: Any) -> list[tuple]:
    if data is None:
        return []
    if isinstance(data, dict):
        edge_list = data.get("edges") or data.get("adjacency")
        if isinstance(edge_list, list):
            return _normalize_edges(edge_list)
        return []
    if isinstance(data, list):
        return _normalize_edges(data)
    return []


def _normalize_edges(edge_list: list) -> list[tuple]:
    edges: list[tuple] = []
    for item in edge_list:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            edges.append((item[0], item[1]))
        elif isinstance(item, dict) and "source" in item and "target" in item:
            edges.append((item["source"], item["target"]))
    return edges


def _task_mentions(task: str, words: tuple[str, ...]) -> bool:
    text = task.lower()
    return any(w in text for w in words)
