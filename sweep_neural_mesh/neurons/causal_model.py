"""
Causal World Model — a persistent directed graph of cause-effect relationships.

Humans don't just observe correlations — they build mental MODELS of how
things cause each other. "If I drop a glass, it will break because gravity
pulls it down and glass is fragile."

This module implements Pearl's Causal Hierarchy:
1. ASSOCIATION: What correlates with what? ( observational)
2. INTERVENTION: What happens if we DO X? ( do-calculus)
3. COUNTERFACTUAL: What WOULD HAVE happened if X didn't? (possible worlds)

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │              CAUSAL WORLD MODEL                      │
    │                                                     │
    │  ┌──────────────────────────────────────────────┐  │
    │  │           Causal DAG                          │  │
    │  │  (Directed Acyclic Graph)                     │  │
    │  │                                              │  │
    │  │  A ──(0.8)──→ B ──(0.6)──→ C                │  │
    │  │  ↑                           ↑              │  │
    │  │  └──(0.4)─── D ──(0.7)──────┘              │  │
    │  └──────────────────────────────────────────────┘  │
    │         ↑           ↑           ↑                   │
    │  ┌──────────┐ ┌──────────┐ ┌──────────────┐       │
    │  │Observed  │ │Intervene │ │Counterfactual│       │
    │  │Evidence  │ │Simulate  │ │Simulate      │       │
    │  └──────────┘ └──────────┘ └──────────────┘       │
    └─────────────────────────────────────────────────────┘

The causal model persists and grows across reasoning episodes,
building an increasingly rich understanding of how the world works.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CausalNode:
    """A node in the causal graph (a concept or event)."""
    node_id: str
    name: str
    node_type: str              # "event", "state", "process", "entity"
    description: str = ""
    confidence: float = 0.5     # how confident we are this node exists
    created_at: float = field(default_factory=time.time)
    observation_count: int = 0


@dataclass
class CausalEdge:
    """A directed edge representing a causal relationship."""
    source_id: str
    target_id: str
    strength: float             # 0.0-1.0: how strong is the causal link
    edge_type: str              # "direct", "indirect", "mediating", "inhibiting"
    evidence_count: int = 0     # how many observations support this link
    confidence: float = 0.5     # how confident in this causal link
    mechanism: str = ""         # description of the causal mechanism
    created_at: float = field(default_factory=time.time)
    last_reinforced: float = field(default_factory=time.time)


@dataclass
class InterventionResult:
    """Result of a causal intervention (do-calculus)."""
    intervention: str           # what we "did"
    affected_nodes: list[str]   # which nodes were affected
    effects: dict[str, float]   # node_id → magnitude of effect
    confidence: float
    reasoning: str


@dataclass
class CounterfactualResult:
    """Result of a counterfactual simulation."""
    counterfactual_condition: str   # "if X had not happened"
    predicted_outcome: str
    probability: float              # how likely this alternative outcome
    affected_chain: list[str]       # causal chain that was disrupted
    reasoning: str


class CausalModel:
    """
    A persistent directed acyclic graph of cause-effect relationships.

    Like the human brain's causal model of the world, this module:

    1. ACCUMULATES causal relationships from reasoning episodes
    2. PROPAGATES effects through the causal graph
    3. ANSWERS interventional queries ("what if we do X?")
    4. ANSWERS counterfactual queries ("what if X hadn't happened?")
    5. LEARNS new causal links from evidence and feedback

    Pearl's Causal Hierarchy:
    - Level 1 (Association): "What is Y given we observed X?"
    - Level 2 (Intervention): "What is Y if we DO X?"
    - Level 3 (Counterfactual): "What WOULD Y have been if X were different?"

    This is what separates causal reasoning from statistical correlation:
    knowing that ice cream sales correlate with drowning doesn't mean
    ice cream CAUSES drowning. A causal model captures the real
    mechanism (hot weather → more swimming → more drowning).
    """

    def __init__(self) -> None:
        self._nodes: dict[str, CausalNode] = {}
        self._edges: list[CausalEdge] = []
        self._intervention_history: list[InterventionResult] = []
        self._counterfactual_history: list[CounterfactualResult] = []
        self._learning_rate = 0.1
        self._next_node_id = 0

    def add_node(
        self,
        name: str,
        node_type: str = "event",
        description: str = "",
    ) -> CausalNode:
        """Add a concept/event to the causal model."""
        # Check if node already exists
        for node in self._nodes.values():
            if node.name.lower() == name.lower():
                node.observation_count += 1
                return node

        self._next_node_id += 1
        node = CausalNode(
            node_id=f"cn_{self._next_node_id}",
            name=name,
            node_type=node_type,
            description=description,
        )
        self._nodes[node.node_id] = node
        return node

    def add_causal_link(
        self,
        source_name: str,
        target_name: str,
        strength: float = 0.5,
        edge_type: str = "direct",
        mechanism: str = "",
    ) -> CausalEdge | None:
        """
        Add or strengthen a causal link between two concepts.
        """
        # Find or create nodes
        source_node = None
        target_node = None
        for node in self._nodes.values():
            if node.name.lower() == source_name.lower():
                source_node = node
            if node.name.lower() == target_name.lower():
                target_node = node

        if not source_node:
            source_node = self.add_node(source_name)
        if not target_node:
            target_node = self.add_node(target_name)

        # Check if edge already exists
        for edge in self._edges:
            if edge.source_id == source_node.node_id and edge.target_id == target_node.node_id:
                # Strengthen existing edge
                edge.strength = min(1.0, edge.strength + self._learning_rate * strength)
                edge.evidence_count += 1
                edge.confidence = min(0.95, 0.5 + edge.evidence_count * 0.05)
                edge.last_reinforced = time.time()
                if mechanism:
                    edge.mechanism = mechanism
                return edge

        # Create new edge
        edge = CausalEdge(
            source_id=source_node.node_id,
            target_id=target_node.node_id,
            strength=strength,
            edge_type=edge_type,
            evidence_count=1,
            confidence=0.5,
            mechanism=mechanism,
        )
        self._edges.append(edge)
        return edge

    def observe_causation(
        self,
        cause: str,
        effect: str,
        strength: float = 0.5,
        mechanism: str = "",
    ) -> None:
        """
        Record an observed causal relationship.
        Called when evidence reveals that X causes Y.
        """
        self.add_causal_link(cause, effect, strength, "direct", mechanism)

    def query_causes(self, concept: str) -> list[dict[str, Any]]:
        """What causes this concept?"""
        target_node = None
        for node in self._nodes.values():
            if node.name.lower() == concept.lower():
                target_node = node
                break

        if not target_node:
            return []

        causes = []
        for edge in self._edges:
            if edge.target_id == target_node.node_id:
                source_node = self._nodes.get(edge.source_id)
                if source_node:
                    causes.append({
                        "cause": source_node.name,
                        "strength": edge.strength,
                        "mechanism": edge.mechanism,
                        "confidence": edge.confidence,
                    })

        return sorted(causes, key=lambda x: x["strength"], reverse=True)

    def query_effects(self, concept: str) -> list[dict[str, Any]]:
        """What does this concept cause?"""
        source_node = None
        for node in self._nodes.values():
            if node.name.lower() == concept.lower():
                source_node = node
                break

        if not source_node:
            return []

        effects = []
        for edge in self._edges:
            if edge.source_id == source_node.node_id:
                target_node = self._nodes.get(edge.target_id)
                if target_node:
                    effects.append({
                        "effect": target_node.name,
                        "strength": edge.strength,
                        "mechanism": edge.mechanism,
                        "confidence": edge.confidence,
                    })

        return sorted(effects, key=lambda x: x["strength"], reverse=True)

    def propagate_effect(
        self,
        source_concept: str,
        magnitude: float = 1.0,
        max_depth: int = 3,
    ) -> dict[str, float]:
        """
        Propagate an effect through the causal graph.

        Like how knocking over one domino affects subsequent dominos,
        this traces causal chains from a source.
        """
        effects: dict[str, float] = {}
        source_node = None
        for node in self._nodes.values():
            if node.name.lower() == source_concept.lower():
                source_node = node
                break

        if not source_node:
            return effects

        visited = set()
        queue = [(source_node.node_id, magnitude, 0)]

        while queue:
            current_id, current_magnitude, depth = queue.pop(0)
            if current_id in visited or depth > max_depth:
                continue
            visited.add(current_id)

            for edge in self._edges:
                if edge.source_id == current_id:
                    propagated = current_magnitude * edge.strength
                    target_node = self._nodes.get(edge.target_id)
                    if target_node and propagated > 0.05:
                        effects[target_node.name] = propagated
                        queue.append((edge.target_id, propagated, depth + 1))

        return effects

    def do_intervention(
        self,
        intervention: str,
        target_concept: str,
    ) -> InterventionResult:
        """
        Level 2: Interventional reasoning (do-calculus).

        "If we DO X, what happens to Y?"
        Unlike just observing X, doing X cuts all incoming causal links to X
        (because we're setting X directly, not letting nature determine it).
        """
        # Find the intervention node
        intervention_node = None
        for node in self._nodes.values():
            if node.name.lower() == intervention.lower():
                intervention_node = node
                break

        if not intervention_node:
            return InterventionResult(
                intervention=intervention,
                affected_nodes=[],
                effects={},
                confidence=0.0,
                reasoning=f"Intervention node '{intervention}' not found in causal model",
            )

        # Propagate from intervention (simulating do(X))
        effects = self.propagate_effect(intervention, magnitude=1.0, max_depth=4)

        # Find affected nodes
        affected = [name for name, mag in effects.items() if mag > 0.1]

        # Compute confidence based on model completeness
        total_edges = len(self._edges)
        relevant_edges = sum(
            1 for e in self._edges
            if e.source_id == intervention_node.node_id
            or e.target_id == intervention_node.node_id
        )
        confidence = min(0.9, 0.3 + relevant_edges * 0.1) if total_edges > 0 else 0.1

        result = InterventionResult(
            intervention=f"do({intervention})",
            affected_nodes=affected,
            effects=effects,
            confidence=confidence,
            reasoning=f"Intervening on '{intervention}' propagates effects to {len(affected)} nodes",
        )
        self._intervention_history.append(result)
        return result

    def counterfactual(
        self,
        condition: str,
        query_concept: str,
    ) -> CounterfactualResult:
        """
        Level 3: Counterfactual reasoning.

        "What WOULD HAVE happened to Y if X had been different?"
        This is the most advanced form of causal reasoning.
        """
        # Find the condition node
        condition_node = None
        for node in self._nodes.values():
            if node.name.lower() == condition.lower():
                condition_node = node
                break

        if not condition_node:
            return CounterfactualResult(
                counterfactual_condition=f"if {condition} had not happened",
                predicted_outcome=f"Cannot determine - '{condition}' not in causal model",
                probability=0.0,
                affected_chain=[],
                reasoning=f"Condition '{condition}' not found in causal model",
            )

        # Find what would be affected if this node didn't exist
        # Remove incoming edges to condition (simulate it not happening)
        affected_chain = []
        effects_without = {}
        for edge in self._edges:
            if edge.target_id == condition_node.node_id:
                # This edge would be disrupted
                source = self._nodes.get(edge.source_id)
                if source:
                    affected_chain.append(source.name)

        # Propagate what happens downstream without this node
        downstream_effects = {}
        for edge in self._edges:
            if edge.source_id == condition_node.node_id:
                target = self._nodes.get(edge.target_id)
                if target:
                    # Without the condition, this effect is reduced
                    downstream_effects[target.name] = 1.0 - edge.strength

        # Check if query concept is affected
        query_probability = 0.5  # default: uncertain
        if query_concept.lower() in {k.lower() for k in downstream_effects}:
            for name, reduction in downstream_effects.items():
                if name.lower() == query_concept.lower():
                    query_probability = reduction
                    break

        result = CounterfactualResult(
            counterfactual_condition=f"if {condition} had not happened",
            predicted_outcome=f"'{query_concept}' would be affected (probability: {query_probability:.0%})",
            probability=query_probability,
            affected_chain=affected_chain,
            reasoning=f"Removing '{condition}' disrupts {len(affected_chain)} upstream causes and {len(downstream_effects)} downstream effects",
        )
        self._counterfactual_history.append(result)
        return result

    def get_causal_chains(
        self,
        start: str,
        end: str,
        max_depth: int = 5,
    ) -> list[list[str]]:
        """Find all causal chains between two concepts."""
        start_node = None
        end_node = None
        for node in self._nodes.values():
            if node.name.lower() == start.lower():
                start_node = node
            if node.name.lower() == end.lower():
                end_node = node

        if not start_node or not end_node:
            return []

        chains: list[list[str]] = []
        self._find_chains(start_node.node_id, end_node.node_id, [], chains, max_depth)
        return chains

    def _find_chains(
        self,
        current_id: str,
        target_id: str,
        path: list[str],
        chains: list[list[str]],
        max_depth: int,
    ) -> None:
        """DFS to find causal chains."""
        if len(path) > max_depth:
            return

        if current_id == target_id:
            node_names = []
            for nid in path:
                node = self._nodes.get(nid)
                if node:
                    node_names.append(node.name)
            chains.append(node_names)
            return

        for edge in self._edges:
            if edge.source_id == current_id and edge.target_id not in path:
                self._find_chains(
                    edge.target_id, target_id,
                    path + [current_id], chains, max_depth,
                )

    def get_graph_stats(self) -> dict[str, Any]:
        """Get statistics about the causal model."""
        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "interventions": len(self._intervention_history),
            "counterfactuals": len(self._counterfactual_history),
            "avg_edge_strength": (
                sum(e.strength for e in self._edges) / len(self._edges)
                if self._edges else 0.0
            ),
            "avg_edge_confidence": (
                sum(e.confidence for e in self._edges) / len(self._edges)
                if self._edges else 0.0
            ),
        }
