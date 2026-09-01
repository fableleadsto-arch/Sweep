"""
Core Protocol — shared interfaces and data types for all neural cores.

Every neural core must implement the NeuralCoreProtocol.
This ensures consistent behavior, composability, and clear contracts.

Architecture:
    ┌─────────────────────────────────────────────────┐
    │  NeuralCoreProtocol (interface)                 │
    │  ├── FactualCore     — knowledge lookup         │
    │  ├── ReasoningCore   — logic & common sense     │
    │  ├── EvidenceCore    — evidence analysis         │
    │  ├── TemporalCore    — date/time reasoning       │
    │  └── CausalCore      — cause-effect chains       │
    │                                                 │
    │  CoreResult (data) — output from any core        │
    │  ConsensusResult   — merged output from coordinator │
    └─────────────────────────────────────────────────┘
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ══════════════════════════════════════════════════════════════
# DATA TYPES
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class CoreResult:
    """Result produced by a single neural core.

    Attributes:
        core_id:     Unique identifier for the core (e.g. "factual").
        answer:      The core's answer string (empty string = no answer).
        confidence:  How confident the core is (0.0 – 1.0).
        reasoning:   Human-readable explanation of how the answer was derived.
        latency_ms:  Wall-clock time for this core's processing.
        evidence_used: Number of evidence items consumed.
        metadata:    Arbitrary extra data the core wants to attach.
    """
    core_id: str
    answer: str
    confidence: float
    reasoning: str
    latency_ms: float
    evidence_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    """Final result after multiple cores have been merged.

    Attributes:
        answer:          The consensus answer.
        confidence:      Aggregated confidence.
        reasoning:       Why this answer was chosen.
        core_results:    Individual results from each core.
        agreement_score: How much the cores agree (0.0 – 1.0).
        latency_ms:      Total wall-clock time including coordination.
        method:          How consensus was reached ("voting" | "weighted" | "fallback" | "single" | "none").
    """
    answer: str
    confidence: float
    reasoning: str
    core_results: list[CoreResult]
    agreement_score: float
    latency_ms: float
    method: str


# ══════════════════════════════════════════════════════════════
# PROTOCOL
# ══════════════════════════════════════════════════════════════

@runtime_checkable
class NeuralCoreProtocol(Protocol):
    """Interface that every neural core must satisfy.

    A neural core is a self-contained processing unit that:
      1. Receives a query + optional evidence.
      2. Produces a CoreResult with an answer and confidence.
      3. Reports its own latency.

    Cores are stateless per-call but may hold pre-compiled
    knowledge (patterns, facts, rules) that is loaded once
    at construction time.
    """

    @property
    def core_id(self) -> str:
        """Unique short identifier, e.g. 'factual', 'causal'."""
        ...

    def process(self, query: str, evidence: list[str]) -> CoreResult:
        """Process a query, optionally using provided evidence.

        Args:
            query:    The user's question or statement.
            evidence: List of evidence strings (may be empty).

        Returns:
            CoreResult with answer, confidence, and reasoning.
        """
        ...


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def make_result(
    core_id: str,
    answer: str,
    confidence: float,
    reasoning: str,
    t0: float,
    evidence_used: int = 0,
    **metadata: Any,
) -> CoreResult:
    """Convenience factory that computes latency from a start time.

    Usage inside a core::

        t0 = time.perf_counter()
        # ... do work ...
        return make_result("factual", "Paris", 0.99, "Capital of France", t0)
    """
    return CoreResult(
        core_id=core_id,
        answer=answer,
        confidence=confidence,
        reasoning=reasoning,
        latency_ms=(time.perf_counter() - t0) * 1000,
        evidence_used=evidence_used,
        metadata=metadata,
    )


def empty_result(core_id: str, t0: float, reason: str = "No match") -> CoreResult:
    """Return an empty / no-answer result."""
    return make_result(core_id, "", 0.0, reason, t0)
