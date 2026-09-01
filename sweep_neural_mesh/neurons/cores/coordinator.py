"""
MultiCoreCoordinator — orchestrates parallel core processing and consensus.

Responsibilities:
  - Run all registered cores in parallel (ThreadPoolExecutor).
  - Merge individual CoreResults into a single ConsensusResult.
  - Support voting, weighted, and fallback consensus methods.
  - Integrate with SelfEvolutionCoordinator for learning.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Any

from ..core_protocol import CoreResult, ConsensusResult, NeuralCoreProtocol
from .factual_core import FactualCore
from .reasoning_core import ReasoningCore
from .evidence_core import EvidenceCore
from .temporal_core import TemporalCore
from .causal_core import CausalCore


class MultiCoreCoordinator:
    """Coordinates multiple neural cores for parallel processing.

    Architecture::

        Query → [Core A, B, C, D, E] in parallel → Consensus → Result
                    ↓
           Self-Evolution: Learn → Evolve → Acquire → Optimize

    Consensus methods:
      - voting:    high agreement → boost confidence
      - weighted:  medium agreement → use highest confidence
      - fallback:  low agreement → penalise confidence
      - single:    only one core responded
    """

    def __init__(self, num_cores: int = 5) -> None:
        all_cores: list[NeuralCoreProtocol] = [
            FactualCore(),
            ReasoningCore(),
            EvidenceCore(),
            TemporalCore(),
            CausalCore(),
        ]
        self._cores = all_cores[:num_cores]
        self._history: list[ConsensusResult] = []

        # Self-evolution (lazy import to avoid circular deps)
        try:
            from ..self_evolution import SelfEvolutionCoordinator
            self._evolution = SelfEvolutionCoordinator()
            self._evolution_enabled = True
        except Exception:
            self._evolution = None
            self._evolution_enabled = False

    # ── Public API ──────────────────────────────────────────

    def process(
        self,
        query: str,
        evidence: list[str],
        parallel: bool = True,
    ) -> ConsensusResult:
        """Run all cores and merge into a consensus result."""
        t0 = time.perf_counter()

        if parallel and len(self._cores) > 1:
            results = self._run_parallel(query, evidence)
        else:
            results = self._run_sequential(query, evidence)

        consensus = self._build_consensus(results)
        # Override latency with actual total time
        consensus = ConsensusResult(
            answer=consensus.answer,
            confidence=consensus.confidence,
            reasoning=consensus.reasoning,
            core_results=consensus.core_results,
            agreement_score=consensus.agreement_score,
            latency_ms=(time.perf_counter() - t0) * 1000,
            method=consensus.method,
        )
        self._history.append(consensus)
        return consensus

    def learn_from_feedback(
        self,
        query: str,
        expected: str,
        actual: str,
        confidence: float = 0.8,
    ) -> dict[str, Any]:
        """Learn from feedback on a previous query."""
        if not self._evolution_enabled or self._evolution is None:
            return {"learned": False}

        source = "multi_core"
        if self._history:
            last = self._history[-1]
            if last.core_results:
                best = max(last.core_results, key=lambda r: r.confidence)
                source = best.core_id

        return self._evolution.process_feedback(
            query=query, expected=expected, actual=actual,
            confidence=confidence, source=source,
        )

    def get_evolution_stats(self) -> dict[str, Any]:
        if not self._evolution_enabled or self._evolution is None:
            return {"enabled": False}
        return {"enabled": True, **self._evolution.get_full_stats()}

    def get_optimization_suggestions(self) -> list[dict[str, Any]]:
        if not self._evolution_enabled or self._evolution is None:
            return []
        return self._evolution.get_optimization_suggestions()

    def get_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "total_queries": len(self._history),
            "num_cores": len(self._cores),
            "evolution_enabled": self._evolution_enabled,
        }
        if self._history:
            stats["avg_latency_ms"] = (
                sum(r.latency_ms for r in self._history) / len(self._history)
            )
            stats["avg_confidence"] = (
                sum(r.confidence for r in self._history) / len(self._history)
            )
            stats["avg_agreement"] = (
                sum(r.agreement_score for r in self._history) / len(self._history)
            )
            stats["methods_used"] = {
                m: sum(1 for r in self._history if r.method == m)
                for m in {r.method for r in self._history}
            }
        return stats

    # ── Internal ────────────────────────────────────────────

    def _run_parallel(self, query: str, evidence: list[str]) -> list[CoreResult]:
        results: list[CoreResult] = []
        with ThreadPoolExecutor(max_workers=len(self._cores)) as pool:
            futures = {
                pool.submit(core.process, query, evidence): core
                for core in self._cores
            }
            for future in as_completed(futures):
                core = futures[future]
                try:
                    results.append(future.result(timeout=5.0))
                except Exception as e:
                    results.append(CoreResult(
                        core_id=core.core_id, answer="", confidence=0.0,
                        reasoning=f"Error: {e}", latency_ms=0.0,
                    ))
        return results

    def _run_sequential(self, query: str, evidence: list[str]) -> list[CoreResult]:
        return [core.process(query, evidence) for core in self._cores]

    def _build_consensus(self, results: list[CoreResult]) -> ConsensusResult:
        """Merge individual core results into a single consensus."""
        if not results:
            return self._empty_consensus("No core results")

        valid = [r for r in results if r.answer and r.confidence > 0]
        if not valid:
            return self._empty_consensus("No valid core results", results)

        if len(valid) == 1:
            r = valid[0]
            return ConsensusResult(
                answer=r.answer,
                confidence=r.confidence * 0.8,
                reasoning=f"Single core ({r.core_id}): {r.reasoning}",
                core_results=results,
                agreement_score=0.5,
                latency_ms=max(x.latency_ms for x in results),
                method="single",
            )

        # Compute pairwise agreement
        answers = [
            re.sub(r"\b(the|a|an|is|are|was|were)\b", "", r.answer.lower().strip()).strip()
            for r in valid
        ]
        agree = 0
        pairs = 0
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                if SequenceMatcher(None, answers[i], answers[j]).ratio() > 0.6:
                    agree += 1
                pairs += 1
        agreement = agree / pairs if pairs else 0.0

        best = max(valid, key=lambda r: r.confidence)

        if agreement > 0.8:
            avg_conf = sum(r.confidence for r in valid) / len(valid)
            return ConsensusResult(
                answer=best.answer,
                confidence=min(0.95, avg_conf * 1.1),
                reasoning=f"High agreement ({agreement:.0%}) across {len(valid)} cores",
                core_results=results,
                agreement_score=agreement,
                latency_ms=max(x.latency_ms for x in results),
                method="voting",
            )
        elif agreement > 0.5:
            return ConsensusResult(
                answer=best.answer,
                confidence=best.confidence * 0.9,
                reasoning=f"Medium agreement ({agreement:.0%}), using highest confidence",
                core_results=results,
                agreement_score=agreement,
                latency_ms=max(x.latency_ms for x in results),
                method="weighted",
            )
        else:
            return ConsensusResult(
                answer=best.answer,
                confidence=best.confidence * 0.7,
                reasoning=f"Low agreement ({agreement:.0%}), using highest confidence with penalty",
                core_results=results,
                agreement_score=agreement,
                latency_ms=max(x.latency_ms for x in results),
                method="fallback",
            )

    @staticmethod
    def _empty_consensus(
        reason: str, results: list[CoreResult] | None = None,
    ) -> ConsensusResult:
        return ConsensusResult(
            answer="", confidence=0.0, reasoning=reason,
            core_results=results or [], agreement_score=0.0,
            latency_ms=0.0, method="none",
        )
