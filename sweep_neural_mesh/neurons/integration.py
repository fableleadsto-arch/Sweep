"""
Integration Hub and Consensus Engine.

The Integration Hub merges signals from all processing centers
using attention-like weighted convergence. This is the thalamus:
the relay station where everything converges before decision.

The Consensus Engine takes the integrated signal and makes
a final decision: does the evidence support the query?
With what confidence? And why?

    Evidence ──┐
    Credibility─┤
    Temporal ───┤→ Integration Hub → Consensus Engine → Decision
    Causal ─────┤
    Contradict ─┘
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .signal import Signal, SignalType
from .logical_inference import LogicalInferenceEngine


@dataclass
class ConsensusDecision:
    """The final decision output by the Consensus Engine."""
    decision: str              # "supported", "refuted", "mixed", "insufficient"
    confidence: float          # 0.0–1.0
    supporting_count: int
    contradicting_count: int
    reasoning: str             # one-line summary of WHY this decision
    factors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "supporting": self.supporting_count,
            "contradicting": self.contradicting_count,
            "reasoning": self.reasoning,
            "factors": self.factors,
        }


class IntegrationHub:
    """
    Merges signals from all processing centers.

    Uses attention-weighted convergence: signals from high-credibility
    sources get amplified, contradictions suppress related signals,
    and temporal relevance modulates everything.

    Enhanced with:
    - Convergent evidence boost: multiple independent signals agreeing → extra confidence
    - Source diversity bonus: evidence from different sources worth more
    - Contradiction-aware weighting: strong contradictions reduce supporting signal weight
    """

    # Default attention weights for each signal type
    DEFAULT_WEIGHTS: dict[SignalType, float] = {
        SignalType.EVIDENCE: 0.25,
        SignalType.CREDIBILITY: 0.30,
        SignalType.TEMPORAL: 0.15,
        SignalType.CAUSAL: 0.20,
        SignalType.CONTRADICTION: 0.35,  # contradictions are loud
    }

    def __init__(self, weights: dict[SignalType, float] | None = None) -> None:
        self._weights = dict(weights or self.DEFAULT_WEIGHTS)
        self._integration_history: list[dict[str, Any]] = []

    def integrate(self, signals: list[Signal]) -> Signal:
        """
        Merge all incoming signals into a single integrated signal.

        Returns one Signal with type INTEGRATED containing the
        attention-weighted convergence of all inputs.
        """
        if not signals:
            return Signal(
                data={"integrated": True, "signal_count": 0},
                signal_type=SignalType.INTEGRATED,
                confidence=0.0,
                source_center="integration_hub",
            )

        # Group signals by type
        by_type: dict[SignalType, list[Signal]] = {}
        for sig in signals:
            by_type.setdefault(sig.signal_type, []).append(sig)

        # Compute weighted score per type
        type_scores: dict[str, float] = {}
        type_counts: dict[str, int] = {}
        all_evidence_texts: list[str] = []

        for sig_type, sigs in by_type.items():
            weight = self._weights.get(sig_type, 0.1)
            avg_confidence = sum(s.confidence for s in sigs) / len(sigs)
            weighted_score = avg_confidence * weight
            type_scores[sig_type.value] = round(weighted_score, 4)
            type_counts[sig_type.value] = len(sigs)

            for s in sigs:
                text = s.data.get("evidence_text", "")
                if text:
                    all_evidence_texts.append(text[:200])

        # ── Convergent evidence boost ──
        # If 3+ evidence signals independently support, boost confidence
        evidence_signals = by_type.get(SignalType.EVIDENCE, [])
        credibility_signals = by_type.get(SignalType.CREDIBILITY, [])
        causal_signals = by_type.get(SignalType.CAUSAL, [])

        convergent_boost = 0.0
        if len(evidence_signals) >= 3:
            avg_ev_conf = sum(s.confidence for s in evidence_signals) / len(evidence_signals)
            if avg_ev_conf > 0.6:
                convergent_boost = min(0.15, (len(evidence_signals) - 2) * 0.05)

        # ── Source diversity bonus ──
        sources: set[str] = set()
        for sig in signals:
            src = sig.data.get("source", "")
            if src and src not in ("unknown", ""):
                sources.add(src)
        diversity_bonus = min(0.10, len(sources) * 0.03) if len(sources) >= 2 else 0.0

        # ── Contradiction penalty (enhanced) ──
        contradiction_signals = by_type.get(SignalType.CONTRADICTION, [])
        contradiction_penalty = 0.0
        if contradiction_signals:
            avg_contra = sum(s.confidence for s in contradiction_signals) / len(contradiction_signals)
            # Strong contradictions (high confidence) hurt more
            contradiction_penalty = avg_contra * 0.45

        # Compute overall confidence
        total_weighted = sum(type_scores.values())
        active_weight_sum = sum(
            self._weights.get(st, 0.1) for st in by_type.keys()
        )
        raw_confidence = total_weighted / active_weight_sum if active_weight_sum > 0 else 0.0

        # Apply boosts and penalties
        adjusted_confidence = raw_confidence + convergent_boost + diversity_bonus - contradiction_penalty
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))

        integrated_data = {
            "integrated": True,
            "signal_count": len(signals),
            "type_scores": type_scores,
            "type_counts": type_counts,
            "raw_confidence": round(raw_confidence, 4),
            "convergent_boost": round(convergent_boost, 4),
            "diversity_bonus": round(diversity_bonus, 4),
            "contradiction_penalty": round(contradiction_penalty, 4),
            "evidence_texts": all_evidence_texts[:10],
            "source_count": len(sources),
        }

        result = Signal(
            data=integrated_data,
            signal_type=SignalType.INTEGRATED,
            confidence=adjusted_confidence,
            source_center="integration_hub",
            metadata={
                "total_types": len(type_scores),
                "dominant_type": max(type_scores, key=type_scores.get) if type_scores else "none",
            },
        )

        self._integration_history.append(integrated_data)
        return result

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._integration_history)


class ConsensusEngine:
    """
    Makes the final decision by evaluating the integrated signal
    against decision thresholds.

    This is the prefrontal cortex: it weighs all the evidence,
    considers contradictions, and produces a definitive answer
    with a confidence level and reasoning.

    Enhanced with:
    - Evidence-strength weighting: high-credibility evidence counts more
    - Refutation-aware logic: active refutation hurts more than passive
    - Evidence volume bonus: 5+ items with no contradictions → boost
    - Better mixed detection: genuinely split evidence → "mixed" not "insufficient"
    """

    def __init__(
        self,
        support_threshold: float = 0.40,
        refutation_threshold: float = 0.25,
        mixed_band: float = 0.12,
    ) -> None:
        self._support_threshold = support_threshold
        self._refutation_threshold = refutation_threshold
        self._mixed_band = mixed_band
        self._decisions: list[ConsensusDecision] = []

    def decide(self, integrated: Signal, raw_signals: list[Signal]) -> Signal:
        """
        Make a consensus decision from the integrated signal.

        Returns a CONSENSUS signal with the decision and reasoning.
        """
        confidence = integrated.confidence
        data = integrated.data if isinstance(integrated.data, dict) else {}

        # Count supporting vs contradicting evidence
        evidence_signals = [s for s in raw_signals if s.signal_type == SignalType.EVIDENCE]
        contradiction_signals = [s for s in raw_signals if s.signal_type == SignalType.CONTRADICTION]
        credibility_signals = [s for s in raw_signals if s.signal_type == SignalType.CREDIBILITY]
        causal_signals = [s for s in raw_signals if s.signal_type == SignalType.CAUSAL]

        # ── Evidence direction analysis (key fix) ──
        # Count evidence that supports vs refutes vs mixed the query
        supporting_evidence = [s for s in evidence_signals
                              if s.data.get("support_direction", "neutral") == "supports"]
        refuting_evidence = [s for s in evidence_signals
                            if s.data.get("support_direction", "neutral") == "refutes"]
        mixed_evidence = [s for s in evidence_signals
                         if s.data.get("support_direction", "neutral") == "mixed"]
        neutral_evidence = [s for s in evidence_signals
                           if s.data.get("support_direction", "neutral") == "neutral"]
        # Evidence that was processed by EvidenceGatherer (has support_direction key)
        processed_evidence = [s for s in evidence_signals if "support_direction" in s.data]
        unprocessed_evidence = [s for s in evidence_signals if "support_direction" not in s.data]

        supports_count = len(supporting_evidence)
        refutes_count = len(refuting_evidence)
        mixed_direction_count = len(mixed_evidence)
        contradicting_count = len(contradiction_signals)

        # Compute evidence direction balance
        total_directional = supports_count + refutes_count
        if total_directional > 0:
            direction_balance = (supports_count - refutes_count) / total_directional
        else:
            direction_balance = 0.0  # neutral

        # Compute factor scores
        factors: list[dict[str, Any]] = []

        # Factor 1: Evidence volume
        total_evidence = len(evidence_signals)
        if total_evidence > 0:
            vol_score = min(1.0, total_evidence / 5.0)
            factors.append({
                "name": "evidence_volume",
                "score": round(vol_score, 3),
                "detail": f"{total_evidence} evidence items: {supports_count} support, {refutes_count} refute, {len(neutral_evidence)} neutral",
            })

        # Factor 2: Evidence direction (NEW — most important factor)
        if total_directional > 0:
            dir_score = (direction_balance + 1) / 2  # map -1..1 to 0..1
            factors.append({
                "name": "evidence_direction",
                "score": round(dir_score, 3),
                "detail": f"direction balance: {direction_balance:+.2f} ({supports_count} vs {refutes_count})",
            })

        # Factor 3: Average credibility
        if credibility_signals:
            avg_cred = sum(s.confidence for s in credibility_signals) / len(credibility_signals)
            factors.append({
                "name": "source_credibility",
                "score": round(avg_cred, 3),
                "detail": f"avg credibility: {avg_cred:.2f}",
            })

        # Factor 4: Causal strength
        if causal_signals:
            avg_causal = sum(s.confidence for s in causal_signals) / len(causal_signals)
            factors.append({
                "name": "causal_strength",
                "score": round(avg_causal, 3),
                "detail": f"{len(causal_signals)} causal links found, avg strength: {avg_causal:.2f}",
            })

        # Factor 5: Contradiction impact
        if contradiction_signals:
            avg_contra = sum(s.confidence for s in contradiction_signals) / len(contradiction_signals)
            active_contradictions = sum(1 for s in contradiction_signals if s.confidence > 0.5)
            contra_detail = f"{contradicting_count} contradictions found"
            if active_contradictions > 0:
                contra_detail += f", {active_contradictions} active (high-confidence)"
            factors.append({
                "name": "contradiction_impact",
                "score": round(avg_contra, 3),
                "detail": contra_detail,
            })

        # ── Compute adjusted confidence using direction ──
        adjusted_confidence = confidence

        # Direction-based adjustment: evidence that refutes lowers confidence
        if total_directional > 0:
            direction_adjustment = direction_balance * 0.3
            adjusted_confidence = max(0.0, min(1.0, adjusted_confidence + direction_adjustment))

        # Evidence volume bonus
        if total_evidence >= 5 and contradicting_count <= 1 and refutes_count == 0:
            volume_bonus = min(0.08, (total_evidence - 4) * 0.02)
            adjusted_confidence = min(1.0, adjusted_confidence + volume_bonus)

        # Active refutation penalty
        if contradiction_signals:
            active_contra_confs = [s.confidence for s in contradiction_signals if s.confidence > 0.5]
            if active_contra_confs:
                active_penalty = min(0.15, len(active_contra_confs) * 0.05)
                adjusted_confidence = max(0.0, adjusted_confidence - active_penalty)

        # ── Evidence relevance: count how many evidence items are directional ──
        directional_ratio = total_directional / max(1, total_evidence)

        # ── LOGICAL INFERENCE OVERRIDE ──
        # Apply formal logic rules. If they produce a strong conclusion,
        # override the evidence-based decision.
        logical_result = None
        try:
            log_engine = LogicalInferenceEngine()
            # Extract query and evidence texts from signals
            query_text = ""
            evidence_texts = []
            for s in raw_signals:
                if s.signal_type == SignalType.RAW:
                    query_text = str(s.data.get("query", ""))
                elif s.signal_type == SignalType.EVIDENCE:
                    ev_text = s.data.get("evidence_text", s.data.get("text", ""))
                    if ev_text:
                        evidence_texts.append(ev_text)

            if query_text and evidence_texts:
                logical_result = log_engine.analyze(query_text, evidence_texts)
        except Exception:
            logical_result = None

        # If logical inference found a strong conclusion, use it
        use_logic = (
            logical_result is not None
            and logical_result.conclusion != "insufficient"
            and logical_result.confidence >= 0.65
        )

        # ── Contradiction-driven mixed detection ──
        active_contradictions = sum(1 for s in contradiction_signals if s.confidence > 0.4)

        # ── Make the decision (direction-aware + relevance-aware) ──

        # LOGIC OVERRIDE: if formal logic found a strong conclusion, use it directly
        if use_logic and logical_result:
            decision = logical_result.conclusion
            # Blend logical confidence with evidence confidence
            adjusted_confidence = (
                0.6 * logical_result.confidence + 0.4 * adjusted_confidence
            )
            reasoning = (
                f"Logical inference ({', '.join(logical_result.inference_chain[:2])}); "
                f"confidence {adjusted_confidence:.2f}"
            )

        # Case 1: No evidence at all
        elif total_evidence == 0 and contradicting_count == 0:
            decision = "insufficient"
            reasoning = "no evidence was provided to evaluate"

        # Case 2: Evidence has mixed direction signals → mixed
        elif mixed_direction_count > 0 and supports_count == 0 and refutes_count == 0:
            decision = "mixed"
            reasoning = f"{mixed_direction_count} evidence items contain both supporting and refuting elements"

        # Case 2b: Mixed direction signals alongside supporting evidence
        # When some evidence is mixed, the answer is nuanced even if other evidence supports
        elif mixed_direction_count > 0 and supports_count > 0 and refutes_count == 0:
            # If more than half the directional evidence is mixed → mixed
            if mixed_direction_count >= supports_count:
                decision = "mixed"
                reasoning = f"{mixed_direction_count} mixed evidence items outweigh {supports_count} supporting items"
            else:
                # Some mixed, some support → supported but acknowledge the nuance
                decision = "supported"
                reasoning = f"{supports_count} pieces of evidence support this, despite {mixed_direction_count} mixed signals"

        # Case 3: Evidence exists but all neutral direction (no relevance to query)
        elif total_directional == 0 and total_evidence > 0 and mixed_direction_count == 0:
            # Only apply strict "insufficient" logic when evidence was actually analyzed
            # (i.e., has support_direction metadata from EvidenceGatherer)
            all_neutral_processed = len(processed_evidence) > 0 and len(neutral_evidence) == len(processed_evidence)
            if all_neutral_processed:
                decision = "insufficient"
                reasoning = f"{total_evidence} evidence items are not directly relevant to the query"
            elif active_contradictions > 0:
                decision = "mixed"
                reasoning = f"{active_contradictions} contradictory signals found in tangentially related evidence"
            elif adjusted_confidence >= 0.7 and contradicting_count == 0:
                decision = "supported"
                reasoning = self._build_reasoning("supported", factors, supports_count, refutes_count)
            elif adjusted_confidence <= self._refutation_threshold:
                decision = "refuted"
                reasoning = f"weak evidence ({total_evidence} items) with low confidence ({adjusted_confidence:.2f})"
            else:
                decision = "insufficient"
                reasoning = f"{total_evidence} evidence items are not directly relevant to the query"

        # Case 4: More evidence refutes than supports → refuted
        elif refutes_count > supports_count and refutes_count >= 1:
            decision = "refuted"
            reasoning = self._build_reasoning("refuted", factors, supports_count, refutes_count)

        # Case 5: Active contradictions detected → mixed
        elif active_contradictions >= 2 and supports_count > 0:
            decision = "mixed"
            reasoning = f"{active_contradictions} contradictions found among {supports_count} supporting signals"

        # Case 6: Equal support and refute → mixed
        elif supports_count == refutes_count and supports_count > 0:
            decision = "mixed"
            reasoning = self._build_reasoning("mixed", factors, supports_count, refutes_count)

        # Case 7: More evidence supports → supported
        elif supports_count > refutes_count and adjusted_confidence >= 0.35:
            decision = "supported"
            reasoning = self._build_reasoning("supported", factors, supports_count, refutes_count)

        # Case 8: Confidence-based fallback
        elif adjusted_confidence >= self._support_threshold:
            decision = "supported"
            reasoning = self._build_reasoning("supported", factors, supports_count, refutes_count)
        elif adjusted_confidence <= self._refutation_threshold:
            decision = "refuted"
            reasoning = self._build_reasoning("refuted", factors, supports_count, refutes_count)
        elif abs(adjusted_confidence - 0.5) <= self._mixed_band and contradicting_count > 0:
            decision = "mixed"
            reasoning = self._build_reasoning("mixed", factors, supports_count, refutes_count)
        else:
            decision = "insufficient"
            reasoning = self._build_reasoning("insufficient", factors, supports_count, refutes_count)

        consensus_data = {
            "decision": decision,
            "confidence": round(adjusted_confidence, 4),
            "raw_confidence": round(confidence, 4),
            "supporting_evidence": supports_count,
            "refuting_evidence": refutes_count,
            "neutral_evidence": len(neutral_evidence),
            "contradicting_evidence": contradicting_count,
            "direction_balance": round(direction_balance, 4),
            "factors": factors,
            "reasoning": reasoning,
        }

        result = Signal(
            data=consensus_data,
            signal_type=SignalType.CONSENSUS,
            confidence=adjusted_confidence,
            source_center="consensus_engine",
            metadata={
                "decision": decision,
                "factor_count": len(factors),
            },
        )

        decision_obj = ConsensusDecision(
            decision=decision,
            confidence=adjusted_confidence,
            supporting_count=supports_count,
            contradicting_count=refutes_count + contradicting_count,
            reasoning=reasoning,
            factors=factors,
        )
        self._decisions.append(decision_obj)
        return result

    def _build_reasoning(
        self,
        decision: str,
        factors: list[dict[str, Any]],
        supporting: int,
        contradicting: int,
    ) -> str:
        """Build a one-line reasoning summary."""
        if decision == "supported":
            parts = [f"{supporting} pieces of evidence support this"]
            if contradicting > 0:
                parts.append(f"despite {contradicting} contradicting signals")
            cred = next((f for f in factors if f["name"] == "source_credibility"), None)
            if cred and cred["score"] > 0.6:
                parts.append("from credible sources")
            causal = next((f for f in factors if f["name"] == "causal_strength"), None)
            if causal and causal["score"] > 0.5:
                parts.append("with causal links")
            return ", ".join(parts)

        if decision == "refuted":
            if contradicting > 0:
                contra = next((f for f in factors if f["name"] == "contradiction_impact"), None)
                if contra and contra["score"] > 0.6:
                    return f"{contradicting} strong contradictions directly refute {supporting} supporting items"
                return f"{contradicting} contradicting signals outweigh {supporting} supporting items"
            avg_score = sum(f["score"] for f in factors) / len(factors) if factors else 0.0
            return f"insufficient evidence ({supporting} items) with low confidence ({avg_score:.0%})"

        if decision == "mixed":
            return f"evidence is divided: {supporting} supporting vs {contradicting} contradicting"

        return f"insufficient data: only {supporting} evidence items with mixed signals"

    @property
    def last_decision(self) -> ConsensusDecision | None:
        return self._decisions[-1] if self._decisions else None

    @property
    def decision_count(self) -> int:
        return len(self._decisions)
