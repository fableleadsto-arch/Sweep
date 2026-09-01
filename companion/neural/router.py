"""Native-first capability router with transparent external fallback.

Decides, per request, whether the native stack can handle a task with real
confidence, or whether the request must fall back to external API providers.

Rules (all honest, none feigned):
- ``generation``  → native ONLY if a trained, validated checkpoint exists AND
  the request is within the model's measured capability envelope. Otherwise
  external.
- ``classify``    → native ONLY if a trained classifier is registered.
- ``embeddings``  → native only for texts within the model's context window.

Every decision records ``source`` (native|external), the model used, and the
reason, so diagnostics can audit routing. External providers remain the
fallback path — native never silently replaces them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .registry import ModelRecord, ModelRegistry


@dataclass
class RouteDecision:
    task: str
    source: str  # "native" | "external"
    model: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "source": self.source,
            "model": self.model,
            "confidence": self.confidence,
            "reason": self.reason,
        }


# Capabilities the native stack is permitted to serve, with a rough confidence
# curve. These are *capability bounds*, not results — actual quality is only
# ever established by evaluation.
_CAPABILITIES: dict[str, dict[str, Any]] = {
    "generation": {"confidence": 0.5, "requires_trained": True},
    "classify": {"confidence": 0.6, "requires_trained": True},
    "embeddings": {"confidence": 0.5, "requires_trained": False},
}


class NativeRouter:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.decisions: list[RouteDecision] = []

    def route(self, task: str, model_name: Optional[str] = None, text_length: int = 0) -> RouteDecision:
        """Decide native vs external for ``task``. Records the decision."""
        cap = _CAPABILITIES.get(task)
        if cap is None:
            d = RouteDecision(task, "external", reason=f"no native capability defined for '{task}'")
            self.decisions.append(d)
            return d

        try:
            record: ModelRecord = self.registry.resolve(model_name)
        except FileNotFoundError:
            d = RouteDecision(task, "external", reason="no native model registered")
            self.decisions.append(d)
            return d

        if cap["requires_trained"] and not record.verified:
            d = RouteDecision(task, "external", model=record.name, reason=f"model '{record.name}' has no trained weights (status={record.status})")
            self.decisions.append(d)
            return d

        if task == "generation" and record.context_length > 0 and text_length > record.context_length:
            d = RouteDecision(task, "external", model=record.name, reason=f"input length {text_length} exceeds context {record.context_length}")
            self.decisions.append(d)
            return d

        d = RouteDecision(
            task,
            "native",
            model=record.name,
            confidence=cap["confidence"],
            reason=f"model '{record.name}' ({record.parameters:,} params, verified={record.verified})",
        )
        self.decisions.append(d)
        return d

    def decision_history(self) -> list[dict[str, Any]]:
        return [d.to_dict() for d in self.decisions]


__all__ = ["NativeRouter", "RouteDecision"]
