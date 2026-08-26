"""
CapabilityRegistry — abstract capability taxonomy.

While the ModelRegistry indexes *nodes*, the CapabilityRegistry indexes
*capabilities themselves* — their hierarchy, relationships, and
modality mappings. It answers questions like:

    "What capabilities exist for IMAGE modality?"
    "What capabilities are needed to go from IMAGE → EMBEDDING?"
    "What is the prerequisite graph for speech_to_text?"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.node import Modality


@dataclass
class CapabilityDef:
    """Definition of a single capability."""
    name: str
    modality: Modality
    description: str = ""
    prerequisites: list[str] = field(default_factory=list)
    outputs: list[Modality] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


class CapabilityRegistry:
    """
    Taxonomy of capabilities and their relationships.

    This is the Mesh's "knowledge graph" of what it can do.
    The router uses it to find paths through capabilities.
    """

    def __init__(self) -> None:
        self._capabilities: dict[str, CapabilityDef] = {}

    def register(
        self,
        name: str,
        modality: Modality,
        description: str = "",
        prerequisites: list[str] | None = None,
        outputs: list[Modality] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self._capabilities[name] = CapabilityDef(
            name=name,
            modality=modality,
            description=description,
            prerequisites=prerequisites or [],
            outputs=outputs or [modality],
            tags=tags or [],
        )

    def get(self, name: str) -> CapabilityDef | None:
        return self._capabilities.get(name)

    def find_by_modality(self, modality: Modality) -> list[CapabilityDef]:
        return [c for c in self._capabilities.values() if c.modality == modality]

    def find_by_tag(self, tag: str) -> list[CapabilityDef]:
        return [c for c in self._capabilities.values() if tag in c.tags]

    def find_prerequisites(self, capability: str) -> list[str]:
        """Return the prerequisite chain for a capability."""
        cap = self._capabilities.get(capability)
        if cap is None:
            return []
        return list(cap.prerequisites)

    def find_path(
        self, from_modality: Modality, to_modality: Modality
    ) -> list[str]:
        """
        Find a capability path from one modality to another.

        Uses BFS through the prerequisite/output graph. Returns
        the ordered list of capability names that bridge the gap.
        """
        # Build adjacency from modality → capabilities that produce it
        modality_caps: dict[Modality, list[str]] = {}
        for cap in self._capabilities.values():
            modality_caps.setdefault(cap.modality, []).append(cap.name)

        if from_modality == to_modality:
            # Same modality: find an identity or encoding capability
            candidates = modality_caps.get(from_modality, [])
            for c in candidates:
                cap = self._capabilities[c]
                if from_modality in cap.outputs:
                    return [c]
            return []

        # BFS
        from collections import deque

        visited: set[str] = set()
        queue: deque[tuple[str, list[str]]] = deque()

        for cap_name in modality_caps.get(from_modality, []):
            queue.append((cap_name, [cap_name]))
            visited.add(cap_name)

        while queue:
            current, path = queue.popleft()
            cap = self._capabilities[current]
            for out_mod in cap.outputs:
                if out_mod == to_modality:
                    return path
                for next_name in modality_caps.get(out_mod, []):
                    if next_name not in visited:
                        visited.add(next_name)
                        queue.append((next_name, path + [next_name]))

        return []

    @property
    def all_capabilities(self) -> list[CapabilityDef]:
        return list(self._capabilities.values())

    @property
    def all_names(self) -> list[str]:
        return sorted(self._capabilities.keys())

    def summary(self) -> dict[str, Any]:
        modality_counts: dict[str, int] = {}
        for cap in self._capabilities.values():
            modality_counts[cap.modality.value] = modality_counts.get(cap.modality.value, 0) + 1
        return {
            "total_capabilities": len(self._capabilities),
            "by_modality": modality_counts,
            "names": self.all_names,
        }

    def __repr__(self) -> str:
        return f"CapabilityRegistry(capabilities={len(self._capabilities)})"
