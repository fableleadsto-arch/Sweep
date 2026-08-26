"""
FusionEngine — combines outputs from multiple nodes.

Implements multiple fusion strategies:
  - early fusion (concatenate raw representations)
  - late fusion (combine final outputs)
  - confidence-weighted fusion
  - gated fusion
  - cross-attention fusion (placeholder for transformer-based)
"""
from __future__ import annotations

import math
from typing import Any


class FusionStrategy:
    """Base class for fusion strategies."""
    name: str = "base"

    def fuse(self, packets: list[Any], **kwargs: Any) -> Any:
        raise NotImplementedError


class LateFusion(FusionStrategy):
    """Combine outputs after independent processing."""
    name = "late_fusion"

    def fuse(self, packets: list[Any], **kwargs: Any) -> Any:
        if not packets:
            return None
        if len(packets) == 1:
            return packets[0]
        # Average numerical outputs, or concatenate lists
        first = packets[0]
        if isinstance(first, (int, float)):
            return sum(packets) / len(packets)
        if isinstance(first, list) and all(isinstance(p, list) for p in packets):
            # Element-wise average
            max_len = max(len(p) for p in packets)
            result = []
            for i in range(max_len):
                vals = [p[i] for p in packets if i < len(p) and isinstance(p[i], (int, float))]
                result.append(sum(vals) / len(vals) if vals else 0.0)
            return result
        # Fallback: return all
        return packets


class ConfidenceWeightedFusion(FusionStrategy):
    """Weight each output by its confidence score."""
    name = "confidence_weighted"

    def fuse(self, packets: list[Any], confidences: list[float] | None = None, **kwargs: Any) -> Any:
        if not packets:
            return None
        if len(packets) == 1:
            return packets[0]
        if confidences is None:
            confidences = [1.0 / len(packets)] * len(packets)
        total = sum(confidences)
        if total == 0:
            return LateFusion().fuse(packets)
        weights = [c / total for c in confidences]
        first = packets[0]
        if isinstance(first, (int, float)):
            return sum(p * w for p, w in zip(packets, weights))
        if isinstance(first, list) and all(isinstance(p, list) for p in packets):
            max_len = max(len(p) for p in packets)
            result = []
            for i in range(max_len):
                weighted_sum = 0.0
                weight_sum = 0.0
                for p, w in zip(packets, weights):
                    if i < len(p) and isinstance(p[i], (int, float)):
                        weighted_sum += p[i] * w
                        weight_sum += w
                result.append(weighted_sum / weight_sum if weight_sum > 0 else 0.0)
            return result
        return packets


class EarlyFusion(FusionStrategy):
    """Concatenate raw representations before processing."""
    name = "early_fusion"

    def fuse(self, packets: list[Any], **kwargs: Any) -> Any:
        if not packets:
            return None
        if len(packets) == 1:
            return packets[0]
        # Flatten and concatenate
        result = []
        for p in packets:
            if isinstance(p, list):
                result.extend(p)
            elif isinstance(p, (int, float)):
                result.append(p)
            else:
                result.append(p)
        return result


class GatedFusion(FusionStrategy):
    """
    Use a learned gate (simplified as confidence-based) to
    decide how much each source contributes.
    """
    name = "gated_fusion"

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def fuse(self, packets: list[Any], confidences: list[float] | None = None, **kwargs: Any) -> Any:
        if not packets:
            return None
        if len(packets) == 1:
            return packets[0]
        if confidences is None:
            confidences = [1.0] * len(packets)
        # Softmax gating
        max_c = max(confidences)
        exps = [math.exp((c - max_c) / self.temperature) for c in confidences]
        total = sum(exps)
        gates = [e / total for e in exps]
        first = packets[0]
        if isinstance(first, (int, float)):
            return sum(p * g for p, g in zip(packets, gates))
        if isinstance(first, list) and all(isinstance(p, list) for p in packets):
            max_len = max(len(p) for p in packets)
            result = []
            for i in range(max_len):
                val = 0.0
                for p, g in zip(packets, gates):
                    if i < len(p) and isinstance(p[i], (int, float)):
                        val += p[i] * g
                result.append(val)
            return result
        return packets


STRATEGIES: dict[str, FusionStrategy] = {
    "late": LateFusion(),
    "confidence_weighted": ConfidenceWeightedFusion(),
    "early": EarlyFusion(),
    "gated": GatedFusion(),
}


class FusionEngine:
    """
    Combines outputs from multiple nodes using pluggable strategies.

    The FusionEngine sits between independent model outputs and the
    final result. It does not assume all outputs are the same type —
    it adapts the fusion strategy to the data.
    """

    def __init__(self, default_strategy: str = "confidence_weighted") -> None:
        self.default_strategy = default_strategy

    def fuse(
        self,
        packets: list[Any],
        strategy: str | None = None,
        confidences: list[float] | None = None,
        **kwargs: Any,
    ) -> Any:
        strat_name = strategy or self.default_strategy
        strat = STRATEGIES.get(strat_name)
        if strat is None:
            raise ValueError(f"Unknown fusion strategy: {strat_name}")
        return strat.fuse(packets, confidences=confidences, **kwargs)

    @staticmethod
    def available_strategies() -> list[str]:
        return list(STRATEGIES.keys())
