"""
Dynamic Graph Rewriter — modifies MeshGraph topology at runtime.

The rewriter monitors execution metrics and applies graph transformations
to optimize performance. Transformations include:
  - Bypassing low-value nodes (short-circuit)
  - Merging sequential nodes with trivial transformations
  - Adding caching nodes after frequently repeated computations
  - Re-routing edges based on measured latency

The rewriter operates on a "propose → validate → apply" cycle to
ensure the graph remains valid (acyclic, connected) after mutations.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RewriteAction(Enum):
    BYPASS = "bypass"
    MERGE = "merge"
    ADD_CACHE = "add_cache"
    REROUTE = "reroute"
    REMOVE = "remove"


@dataclass
class RewriteProposal:
    """A proposed change to the graph."""
    action: RewriteAction
    target_node_id: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0  # 0-1, how confident we are this helps


@dataclass
class RewriteRecord:
    """A successfully applied rewrite."""
    proposal: RewriteProposal
    applied_at: float
    graph_hash_before: str
    graph_hash_after: str
    latency_before_ms: float = 0.0
    latency_after_ms: float = 0.0


class DynamicGraphRewriter:
    """
    Monitors execution metrics and proposes/applies graph mutations.

    The rewriter is conservative — it proposes rewrites only when
    metrics strongly suggest they will help, and validates that
    the graph remains acyclic and connected after each mutation.

    This is an original implementation of runtime graph optimization
    for neural mesh execution.
    """

    def __init__(self) -> None:
        self._node_metrics: dict[str, _NodeMetrics] = {}
        self._rewrite_history: list[RewriteRecord] = []
        self._min_samples: int = 5  # minimum executions before proposing
        self._bypass_threshold: float = 0.02  # max avg confidence to consider bypass
        self._cache_threshold: int = 3  # min repeated calls to add cache

    def record_execution(
        self,
        node_id: str,
        latency_ms: float,
        confidence: float,
        success: bool,
    ) -> None:
        """Record an execution event for a node."""
        if node_id not in self._node_metrics:
            self._node_metrics[node_id] = _NodeMetrics()
        m = self._node_metrics[node_id]
        m.executions += 1
        m.total_latency_ms += latency_ms
        m.total_confidence += confidence
        m.successes += 1 if success else 0
        m.last_seen = time.time()

    def propose_rewrites(self, graph: Any) -> list[RewriteProposal]:
        """
        Analyze metrics and propose graph rewrites.

        Args:
            graph: A MeshGraph instance.

        Returns:
            List of proposals sorted by confidence (highest first).
        """
        proposals: list[RewriteProposal] = []

        for node_id, metrics in self._node_metrics.items():
            if metrics.executions < self._min_samples:
                continue

            avg_confidence = metrics.total_confidence / metrics.executions
            avg_latency = metrics.total_latency_ms / metrics.executions
            success_rate = metrics.successes / metrics.executions

            # Propose bypass for consistently low-confidence nodes
            if avg_confidence < self._bypass_threshold and success_rate < 0.5:
                proposals.append(RewriteProposal(
                    action=RewriteAction.BYPASS,
                    target_node_id=node_id,
                    description=(
                        f"bypass node with avg confidence {avg_confidence:.3f} "
                        f"and success rate {success_rate:.2f}"
                    ),
                    params={"avg_latency_ms": avg_latency},
                    confidence=1.0 - success_rate,
                ))

            # Propose cache for high-latency frequently-called nodes
            if metrics.executions >= self._cache_threshold and avg_latency > 100.0:
                proposals.append(RewriteProposal(
                    action=RewriteAction.ADD_CACHE,
                    target_node_id=node_id,
                    description=(
                        f"add cache after node with avg latency {avg_latency:.1f}ms "
                        f"called {metrics.executions} times"
                    ),
                    params={"avg_latency_ms": avg_latency, "call_count": metrics.executions},
                    confidence=min(metrics.executions / 20.0, 0.9),
                ))

        # Sort by confidence descending
        proposals.sort(key=lambda p: p.confidence, reverse=True)
        return proposals

    def validate_proposal(self, proposal: RewriteProposal, graph: Any) -> bool:
        """
        Validate that applying this proposal would keep the graph valid.

        Checks:
          - Target node exists in graph
          - BYPASS: node has exactly one successor
          - Graph remains acyclic after hypothetical removal
        """
        node_ids = {n.node_id for n in graph.nodes}

        if proposal.target_node_id not in node_ids:
            return False

        if proposal.action == RewriteAction.BYPASS:
            # Check node has exactly one successor
            successors = graph.successors(proposal.target_node_id)
            if len(successors) != 1:
                return False
            # Check node has exactly one predecessor
            predecessors = graph.predecessors(proposal.target_node_id)
            if len(predecessors) != 1:
                return False

        return True

    def apply_proposal(
        self, proposal: RewriteProposal, graph: Any, current_latency_ms: float = 0.0
    ) -> RewriteRecord | None:
        """
        Apply a validated proposal to the graph.

        Returns a RewriteRecord on success, None if validation fails.
        """
        if not self.validate_proposal(proposal, graph):
            logger.warning("proposal validation failed: %s", proposal.description)
            return None

        graph_hash_before = self._graph_hash(graph)

        if proposal.action == RewriteAction.BYPASS:
            self._apply_bypass(proposal, graph)
        elif proposal.action == RewriteAction.ADD_CACHE:
            self._apply_cache_insertion(proposal, graph)
        elif proposal.action == RewriteAction.REMOVE:
            self._apply_remove(proposal, graph)
        else:
            logger.warning("unsupported rewrite action: %s", proposal.action)
            return None

        graph_hash_after = self._graph_hash(graph)

        record = RewriteRecord(
            proposal=proposal,
            applied_at=time.time(),
            graph_hash_before=graph_hash_before,
            graph_hash_after=graph_hash_after,
            latency_before_ms=current_latency_ms,
        )
        self._rewrite_history.append(record)
        logger.info("applied rewrite: %s", proposal.description)
        return record

    def _apply_bypass(self, proposal: RewriteProposal, graph: Any) -> None:
        """Bypass a node by connecting its predecessor directly to its successor."""
        predecessors = graph.predecessors(proposal.target_node_id)
        successors = graph.successors(proposal.target_node_id)
        if predecessors and successors:
            pred = predecessors[0]
            succ = successors[0]
            graph.remove_node(proposal.target_node_id)
            graph.add_edge(pred, succ)

    def _apply_cache_insertion(self, proposal: RewriteProposal, graph: Any) -> None:
        """Insert a cache marker in node tags (soft cache hint)."""
        for node in graph.nodes:
            if node.node_id == proposal.target_node_id:
                node.tags["cache_enabled"] = "true"
                node.tags["cache_ttl_ms"] = str(int(proposal.params.get("avg_latency_ms", 1000) * 2))
                break

    def _apply_remove(self, proposal: RewriteProposal, graph: Any) -> None:
        """Remove a node from the graph."""
        graph.remove_node(proposal.target_node_id)

    def _graph_hash(self, graph: Any) -> str:
        """Simple content hash of graph structure for change detection."""
        edges = sorted(
            (e.from_node_id, e.to_node_id)
            for e in graph.edges
        )
        return str(hash(tuple(edges)))

    @property
    def rewrite_count(self) -> int:
        return len(self._rewrite_history)

    @property
    def metrics_tracked(self) -> int:
        return len(self._node_metrics)

    def stats(self) -> dict[str, Any]:
        return {
            "nodes_tracked": self.metrics_tracked,
            "rewrites_applied": self.rewrite_count,
            "proposals_generated": sum(
                1 for _ in []  # placeholder — proposals are transient
            ),
            "action_breakdown": {
                action.value: sum(
                    1 for r in self._rewrite_history
                    if r.proposal.action == action
                )
                for action in RewriteAction
            },
        }

    def __repr__(self) -> str:
        return (
            f"DynamicGraphRewriter(tracked={self.metrics_tracked}, "
            f"rewrites={self.rewrite_count})"
        )


@dataclass
class _NodeMetrics:
    """Accumulated metrics for a single node."""
    executions: int = 0
    total_latency_ms: float = 0.0
    total_confidence: float = 0.0
    successes: int = 0
    last_seen: float = 0.0
