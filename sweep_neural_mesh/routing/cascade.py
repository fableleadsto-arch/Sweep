"""
Cascade Router — routes through increasingly powerful models.

The cascade approach tries cheap/fast models first and escalates to
more expensive/powerful models only when confidence is insufficient.
This reduces average cost while maintaining quality.

    Query
      ↓
  [Tier 1: cheap/fast]  ──confidence OK──→  Return
      ↓ confidence low
  [Tier 2: moderate]    ──confidence OK──→  Return
      ↓ confidence low
  [Tier 3: expensive]   ──always return──→  Return
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CascadeTier:
    """A tier in the cascade — a model with its confidence threshold."""
    name: str
    cost_weight: float  # relative cost (1.0 = baseline)
    latency_weight: float  # relative latency (1.0 = baseline)
    confidence_threshold: float  # minimum confidence to stop escalating
    handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CascadeResult:
    """Result of a cascade routing decision."""
    tier_used: str
    confidence: float
    output: Any
    tiers_attempted: list[str]
    total_latency_ms: float
    escalated: bool
    cost_saved: float  # fraction of max cost saved by not using most expensive tier

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_used": self.tier_used,
            "confidence": self.confidence,
            "tiers_attempted": self.tiers_attempted,
            "total_latency_ms": self.total_latency_ms,
            "escalated": self.escalated,
            "cost_saved": self.cost_saved,
        }


class CascadeRouter:
    """
    Routes queries through tiers of increasing capability.

    Each tier has a handler that returns {"confidence": float, "output": Any}.
    The cascade stops at the first tier whose confidence meets the threshold.
    If no tier meets it, the last tier's result is used.

    This is an original implementation — the cascade concept is well-known
    in ML cost optimization but this specific API and integration with
    the Neural Mesh is original.
    """

    def __init__(self, tiers: list[CascadeTier] | None = None) -> None:
        self._tiers: list[CascadeTier] = tiers if tiers is not None else []
        self._history: list[CascadeResult] = []

    def add_tier(self, tier: CascadeTier) -> None:
        """Add a tier. Tiers are tried in insertion order."""
        self._tiers.append(tier)

    def route(self, query: dict[str, Any]) -> CascadeResult:
        """
        Route a query through the cascade.

        The query dict is passed to each tier's handler. The handler must
        return {"confidence": float, "output": Any}.
        """
        if not self._tiers:
            raise ValueError("no tiers configured in cascade")

        t0 = time.perf_counter()
        tiers_attempted: list[str] = []
        last_result: dict[str, Any] = {"confidence": 0.0, "output": None}

        for tier in self._tiers:
            tiers_attempted.append(tier.name)

            if tier.handler is not None:
                result = tier.handler(query)
            else:
                result = {"confidence": 0.0, "output": None}

            confidence = float(result.get("confidence", 0.0))
            output = result.get("output")

            if confidence >= tier.confidence_threshold:
                elapsed = (time.perf_counter() - t0) * 1000
                max_cost = self._tiers[-1].cost_weight if self._tiers else 1.0
                cost_saved = 1.0 - (tier.cost_weight / max_cost) if max_cost > 0 else 0.0
                cascade_result = CascadeResult(
                    tier_used=tier.name,
                    confidence=confidence,
                    output=output,
                    tiers_attempted=tiers_attempted,
                    total_latency_ms=elapsed,
                    escalated=len(tiers_attempted) > 1,
                    cost_saved=cost_saved,
                )
                self._history.append(cascade_result)
                return cascade_result

            last_result = {"confidence": confidence, "output": output}

        # No tier met threshold — use last tier's result
        elapsed = (time.perf_counter() - t0) * 1000
        cascade_result = CascadeResult(
            tier_used=self._tiers[-1].name,
            confidence=last_result["confidence"],
            output=last_result["output"],
            tiers_attempted=tiers_attempted,
            total_latency_ms=elapsed,
            escalated=len(tiers_attempted) > 1,
            cost_saved=0.0,
        )
        self._history.append(cascade_result)
        return cascade_result

    @property
    def tier_count(self) -> int:
        return len(self._tiers)

    @property
    def history(self) -> list[CascadeResult]:
        return list(self._history)

    def stats(self) -> dict[str, Any]:
        if not self._history:
            return {"routes": 0, "tiers": self.tier_count}
        total = len(self._history)
        escalated = sum(1 for r in self._history if r.escalated)
        avg_confidence = sum(r.confidence for r in self._history) / total
        avg_cost_saved = sum(r.cost_saved for r in self._history) / total
        tier_usage: dict[str, int] = {}
        for r in self._history:
            tier_usage[r.tier_used] = tier_usage.get(r.tier_used, 0) + 1
        return {
            "routes": total,
            "tiers": self.tier_count,
            "escalation_rate": escalated / total,
            "avg_confidence": avg_confidence,
            "avg_cost_saved": avg_cost_saved,
            "tier_usage": tier_usage,
        }

    def __repr__(self) -> str:
        names = [t.name for t in self._tiers]
        return f"CascadeRouter(tiers={names})"
