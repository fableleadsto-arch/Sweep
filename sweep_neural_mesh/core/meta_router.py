"""
Meta-Router — learns which nodes work best for which conditions.

The Meta-Router sits above the ModelRouter. While the ModelRouter
scores candidates based on static weights, the Meta-Router learns
from historical execution data which routing decisions led to
the best outcomes.

    Input characteristics
            ↓
        Meta-Router
            ↓
        Model selection
            ↓
        Inference
            ↓
        Evaluation
            ↓
        Feedback
            └──→ Meta-Router training data
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoutingExperience:
    """A recorded routing decision and its outcome."""
    timestamp: float
    task: str
    input_modality: str
    input_quality: float
    node_selected: str
    node_capabilities: list[str]
    confidence_achieved: float
    latency_ms: float
    success: bool
    feedback_score: float = 0.0  # 0-1, from external evaluation


@dataclass
class MetaRoutingDecision:
    """Output of the meta-router."""
    recommended_node: str
    confidence: float
    alternatives: list[str] = field(default_factory=list)
    based_on_experiences: int = 0
    reasoning: str = ""


class MetaRouter:
    """
    Learns routing patterns from execution history.

    The MetaRouter maintains a simple statistical model:
    - Per-node success rates broken down by task type
    - Per-node average confidence broken down by input quality
    - Per-node latency distributions

    Initially falls back to the base router. As experiences
    accumulate, it gradually takes over routing decisions.
    """

    def __init__(self, confidence_threshold: float = 0.6) -> None:
        self.confidence_threshold = confidence_threshold
        self._experiences: list[RoutingExperience] = []
        self._node_stats: dict[str, dict[str, list[float]]] = {}

    def record(self, experience: RoutingExperience) -> None:
        """Record a routing experience for learning."""
        self._experiences.append(experience)
        node = experience.node_selected
        if node not in self._node_stats:
            self._node_stats[node] = {
                "successes": [],
                "confidences": [],
                "latencies": [],
                "feedback_scores": [],
            }
        stats = self._node_stats[node]
        stats["successes"].append(1.0 if experience.success else 0.0)
        stats["confidences"].append(experience.confidence_achieved)
        stats["latencies"].append(experience.latency_ms)
        stats["feedback_scores"].append(experience.feedback_score)

    def recommend(
        self,
        task: str,
        input_modality: str,
        input_quality: float,
        candidate_nodes: list[str],
    ) -> MetaRoutingDecision:
        """
        Recommend the best node based on learned patterns.

        If insufficient data exists, returns low confidence
        to signal the base router should decide.
        """
        if not self._experiences or not candidate_nodes:
            return MetaRoutingDecision(
                recommended_node=candidate_nodes[0] if candidate_nodes else "",
                confidence=0.0,
                reasoning="no historical data",
            )

        scores: dict[str, float] = {}
        for node in candidate_nodes:
            if node not in self._node_stats:
                scores[node] = 0.0
                continue
            stats = self._node_stats[node]
            n = len(stats["successes"])
            if n == 0:
                scores[node] = 0.0
                continue

            # Weighted score: success_rate * avg_confidence * (1 / avg_latency)
            success_rate = sum(stats["successes"]) / n
            avg_conf = sum(stats["confidences"]) / n
            avg_lat = sum(stats["latencies"]) / n
            avg_feedback = sum(stats["feedback_scores"]) / n if stats["feedback_scores"] else 0.5

            # Recency bias: more recent experiences matter more
            recency = min(n / 10.0, 1.0)

            score = (
                success_rate * 0.3
                + avg_conf * 0.25
                + (1.0 / (1.0 + avg_lat / 1000.0)) * 0.15
                + avg_feedback * 0.2
                + recency * 0.1
            )
            scores[node] = score

        if not scores:
            return MetaRoutingDecision(
                recommended_node=candidate_nodes[0],
                confidence=0.0,
                reasoning="no scored candidates",
            )

        best = max(scores, key=scores.get)
        sorted_nodes = sorted(scores, key=scores.get, reverse=True)

        # Confidence based on how many experiences we have
        relevant = sum(
            1 for e in self._experiences
            if e.node_selected in candidate_nodes
        )
        data_confidence = min(relevant / 20.0, 1.0)  # saturates at 20 experiences

        return MetaRoutingDecision(
            recommended_node=best,
            confidence=scores[best] * data_confidence,
            alternatives=sorted_nodes[1:],
            based_on_experiences=relevant,
            reasoning=f"scored {len(candidate_nodes)} nodes from {relevant} experiences",
        )

    @property
    def experience_count(self) -> int:
        return len(self._experiences)

    def summary(self) -> dict[str, Any]:
        if not self._experiences:
            return {"experiences": 0, "nodes_seen": 0}
        nodes_seen = set(e.node_selected for e in self._experiences)
        tasks_seen = set(e.task for e in self._experiences)
        return {
            "experiences": len(self._experiences),
            "nodes_seen": len(nodes_seen),
            "tasks_seen": len(tasks_seen),
            "avg_confidence": sum(e.confidence_achieved for e in self._experiences) / len(self._experiences),
            "success_rate": sum(1 for e in self._experiences if e.success) / len(self._experiences),
        }

    def __repr__(self) -> str:
        return f"MetaRouter(experiences={len(self._experiences)})"
