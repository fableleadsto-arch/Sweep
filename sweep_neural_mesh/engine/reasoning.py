"""
Reasoning Engine — original orchestration layer for the Sweep Neural Engine.

Pipeline:
    QUESTION
    -> PROBLEM DECOMPOSITION
    -> EVIDENCE COLLECTION
    -> EVIDENCE NORMALIZATION
    -> HYPOTHESIS GENERATION
    -> HYPOTHESIS COMPARISON
    -> VERIFICATION
    -> CONCLUSION

Distinguishes:
    - observation
    - source claim
    - inference
    - hypothesis
    - conclusion

Does not expose private chain-of-thought.
Returns concise reasoning summaries and evidence.

Sweep-original implementation.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from .verification import VerificationCore, VerificationResult

logger = logging.getLogger("sweep.reasoning")


# ════════════════════════════════════════════════════════════════════
# REASONING TYPES
# ════════════════════════════════════════════════════════════════════

class ClaimType:
    OBSERVATION = "observation"
    SOURCE_CLAIM = "source_claim"
    INFERENCE = "inference"
    HYPOTHESIS = "hypothesis"
    CONCLUSION = "conclusion"


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain."""
    step_type: str  # ClaimType
    content: str
    confidence: float = 0.0
    source: str = ""
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class ReasoningChain:
    """Complete reasoning chain from question to conclusion."""
    steps: list[ReasoningStep] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.0
    decision: str = "insufficient"  # supported, refuted, mixed, insufficient
    reasoning_summary: str = ""
    verification: VerificationResult | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """A candidate explanation."""
    statement: str
    confidence: float = 0.0
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════
# REASONING ENGINE
# ════════════════════════════════════════════════════════════════════

class ReasoningEngine:
    """Original reasoning orchestration layer.

    Takes a query and evidence, decomposes the problem, generates
    hypotheses, verifies them, and produces a conclusion.

    This is Sweep's own orchestration logic. It delegates to:
        - LogicalInferenceEngine for formal logic
        - NeuralProofMesh for proof propagation
        - BayesianReasoner for evidence updating
        - VerificationCore for self-checking
    """

    def __init__(self) -> None:
        self._verification = VerificationCore()
        self._logical_engine = None
        self._proof_mesh = None
        self._bayesian = None

    def _ensure_engines(self) -> None:
        """Lazy-load reasoning sub-engines."""
        import sys
        # Ensure neurons package is importable
        _sweep_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _sweep_dir not in sys.path:
            sys.path.insert(0, _sweep_dir)

        if self._logical_engine is None:
            try:
                from neurons.logical_inference import LogicalInferenceEngine
                self._logical_engine = LogicalInferenceEngine()
            except ImportError:
                pass
        if self._proof_mesh is None:
            try:
                from neurons.proof_mesh import NeuralProofMesh
                self._proof_mesh = NeuralProofMesh()
            except ImportError:
                pass
        if self._bayesian is None:
            try:
                from neurons.bayesian import BayesianReasoner
                self._bayesian = BayesianReasoner()
            except ImportError:
                pass

    def reason(
        self,
        query: str,
        evidence: list[str] | None = None,
        context: dict | None = None,
    ) -> ReasoningChain:
        """Run the full reasoning pipeline.

        Returns a ReasoningChain with steps, conclusion, and verification.
        """
        t0 = time.perf_counter()
        self._ensure_engines()
        evidence = evidence or []
        chain = ReasoningChain()

        # Step 1: Problem Decomposition
        sub_questions = self._decompose(query)
        chain.steps.append(ReasoningStep(
            step_type=ClaimType.OBSERVATION,
            content=f"Decomposed into {len(sub_questions)} sub-questions",
            confidence=0.9,
        ))

        # Step 2: Evidence Collection & Normalization
        normalized = self._normalize_evidence(evidence)
        chain.steps.append(ReasoningStep(
            step_type=ClaimType.OBSERVATION,
            content=f"Normalized {len(normalized)} evidence items",
            confidence=0.9,
        ))

        # Step 3: Try formal logic first
        if self._logical_engine:
            try:
                lr = self._logical_engine.analyze(query, evidence)
                if lr.conclusion in ("supported", "refuted", "mixed") and lr.confidence >= 0.5:
                    chain.steps.append(ReasoningStep(
                        step_type=ClaimType.INFERENCE,
                        content=f"Formal logic: {lr.conclusion} ({lr.confidence:.2f})",
                        confidence=lr.confidence,
                    ))
                    chain.conclusion = lr.reasoning
                    chain.confidence = lr.confidence
                    chain.decision = lr.conclusion
            except Exception:
                pass

        # Step 4: Try proof mesh
        if self._proof_mesh and not chain.decision:
            try:
                pr = self._proof_mesh.solve(query, evidence)
                if pr.conclusion in ("supported", "refuted", "mixed") and pr.confidence >= 0.5:
                    chain.steps.append(ReasoningStep(
                        step_type=ClaimType.INFERENCE,
                        content=f"Proof mesh: {pr.conclusion} ({pr.confidence:.2f})",
                        confidence=pr.confidence,
                    ))
                    chain.conclusion = pr.reasoning[0] if pr.reasoning else pr.conclusion
                    chain.confidence = pr.confidence
                    chain.decision = pr.conclusion
            except Exception:
                pass

        # Step 5: Generate hypotheses if no conclusion yet
        if not chain.decision:
            hypotheses = self._generate_hypotheses(query, evidence)
            chain.steps.append(ReasoningStep(
                step_type=ClaimType.HYPOTHESIS,
                content=f"Generated {len(hypotheses)} hypotheses",
                confidence=0.5,
            ))
            if hypotheses:
                best = max(hypotheses, key=lambda h: h.confidence)
                chain.conclusion = best.statement
                chain.confidence = best.confidence
                chain.decision = "supported" if best.confidence > 0.5 else "insufficient"

        # Step 6: Bayesian evidence updating
        if self._bayesian and chain.confidence > 0:
            try:
                # Simple Bayesian update based on evidence count
                sup = sum(1 for e in evidence if not any(
                    w in e.lower() for w in ["not", "no", "never", "false", "refute"]
                ))
                ref = sum(1 for e in evidence if any(
                    w in e.lower() for w in ["not", "no", "never", "false", "refute"]
                ))
                if sup + ref > 0:
                    prior = chain.confidence
                    likelihood_sup = (sup + 0.1) / (sup + ref + 0.2)
                    posterior = (likelihood_sup * prior) / (
                        likelihood_sup * prior + (1 - likelihood_sup) * (1 - prior) + 1e-8
                    )
                    chain.confidence = round(0.7 * chain.confidence + 0.3 * posterior, 4)
            except Exception:
                pass

        # Step 7: Verification
        chain.verification = self._verification.verify(
            query=query,
            answer=chain.conclusion,
            confidence=chain.confidence,
            reasoning=chain.reasoning_summary,
            evidence=evidence,
            decision=chain.decision,
        )
        # Apply verification adjustment
        chain.confidence = max(0.0, min(1.0,
            chain.confidence + chain.verification.confidence_adjustment
        ))

        # Step 8: Generate summary
        chain.reasoning_summary = self._summarize(chain)
        chain.latency_ms = (time.perf_counter() - t0) * 1000

        return chain

    def _decompose(self, query: str) -> list[str]:
        """Decompose a query into sub-questions."""
        sub_questions = [query]
        # Simple decomposition: split compound questions
        if " and " in query.lower():
            parts = query.lower().split(" and ")
            if len(parts) <= 3:
                sub_questions = [p.strip() for p in parts]
        return sub_questions

    def _normalize_evidence(self, evidence: list[str]) -> list[dict]:
        """Normalize evidence into structured format."""
        normalized = []
        for e in evidence:
            normalized.append({
                "text": e,
                "type": ClaimType.SOURCE_CLAIM,
                "confidence": 0.7,
            })
        return normalized

    def _generate_hypotheses(self, query: str, evidence: list[str]) -> list[Hypothesis]:
        """Generate candidate hypotheses from evidence."""
        hypotheses = []
        # Simple: treat each evidence item as supporting a hypothesis
        for e in evidence:
            h = Hypothesis(
                statement=e[:200],
                confidence=0.5,
                supporting_evidence=[e],
            )
            hypotheses.append(h)
        return hypotheses

    def _summarize(self, chain: ReasoningChain) -> str:
        """Generate a concise reasoning summary."""
        parts = []
        if chain.decision:
            parts.append(f"Decision: {chain.decision}")
        if chain.conclusion:
            parts.append(f"Conclusion: {chain.conclusion[:200]}")
        parts.append(f"Confidence: {chain.confidence:.2f}")
        if chain.verification and chain.verification.warnings:
            parts.append(f"Warnings: {'; '.join(chain.verification.warnings[:3])}")
        return " | ".join(parts)
