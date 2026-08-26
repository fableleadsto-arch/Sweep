"""
ModelRegistry — dynamic discovery of nodes by capability.

The registry is the Mesh's phone book. Rather than hard-coding model
selection, the router queries the registry:

    candidates = registry.find_capability("visual_feature_extraction")
    selected = router.rank(candidates, context)

This is the primary scalability mechanism.
"""
from __future__ import annotations

import threading
from typing import Any

from ..core.node import Framework, NeuralNode


class ModelRegistry:
    """
    Thread-safe registry of NeuralNodes indexed by capability.

    Models are never looked up by name — only by what they can do.
    This allows hot-swapping implementations without touching callers.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, NeuralNode] = {}
        self._capability_index: dict[str, set[str]] = {}
        self._framework_index: dict[Framework, set[str]] = {}
        self._lock = threading.Lock()

    # -- Registration --

    def register(self, node: NeuralNode) -> None:
        """Register a node and index its capabilities."""
        with self._lock:
            self._nodes[node.node_id] = node
            for cap in node.capabilities:
                self._capability_index.setdefault(cap, set()).add(node.node_id)
            self._framework_index.setdefault(node.framework, set()).add(node.node_id)

    def unregister(self, node_id: str) -> NeuralNode | None:
        """Remove a node from the registry."""
        with self._lock:
            node = self._nodes.pop(node_id, None)
            if node is None:
                return None
            for cap in node.capabilities:
                bucket = self._capability_index.get(cap)
                if bucket:
                    bucket.discard(node_id)
                    if not bucket:
                        del self._capability_index[cap]
            fw_bucket = self._framework_index.get(node.framework)
            if fw_bucket:
                fw_bucket.discard(node_id)
                if not fw_bucket:
                    del self._framework_index[node.framework]
            return node

    # -- Discovery --

    def find_capability(self, capability: str) -> list[NeuralNode]:
        """Find all nodes that declare a given capability."""
        with self._lock:
            node_ids = self._capability_index.get(capability, set())
            return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def find_framework(self, framework: Framework) -> list[NeuralNode]:
        """Find all nodes of a specific framework."""
        with self._lock:
            node_ids = self._framework_index.get(framework, set())
            return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def find_capabilities(
        self, capabilities: list[str], require_all: bool = True
    ) -> list[NeuralNode]:
        """
        Find nodes matching multiple capabilities.

        If require_all=True, node must have ALL listed capabilities.
        If require_all=False, node must have ANY of them.
        """
        with self._lock:
            if not capabilities:
                return []
            if require_all:
                sets = [
                    self._capability_index.get(cap, set()) for cap in capabilities
                ]
                common = sets[0]
                for s in sets[1:]:
                    common = common & s
                return [self._nodes[nid] for nid in common if nid in self._nodes]
            else:
                union: set[str] = set()
                for cap in capabilities:
                    union |= self._capability_index.get(cap, set())
                return [self._nodes[nid] for nid in union if nid in self._nodes]

    def get(self, node_id: str) -> NeuralNode | None:
        return self._nodes.get(node_id)

    @property
    def nodes(self) -> list[NeuralNode]:
        with self._lock:
            return list(self._nodes.values())

    @property
    def capabilities(self) -> list[str]:
        with self._lock:
            return sorted(self._capability_index.keys())

    @property
    def frameworks(self) -> list[Framework]:
        with self._lock:
            return list(self._framework_index.keys())

    @property
    def size(self) -> int:
        return len(self._nodes)

    # -- Stats --

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_nodes": len(self._nodes),
                "capabilities": len(self._capability_index),
                "frameworks": {fw.value: len(ids) for fw, ids in self._framework_index.items()},
                "capability_list": sorted(self._capability_index.keys()),
                "idle_nodes": sum(1 for n in self._nodes.values() if n.status.value == "idle"),
                "ready_nodes": sum(1 for n in self._nodes.values() if n.status.value == "ready"),
                "failed_nodes": sum(1 for n in self._nodes.values() if n.status.value == "failed"),
            }

    def __repr__(self) -> str:
        return f"ModelRegistry(nodes={self.size}, caps={len(self._capability_index)})"
