"""
ExplanationNarrator — translates the reasoning trace into human language.

Like Broca's area converting thoughts into speech, the Narrator
takes the structured explanation data and produces natural,
readable explanations that tell the user:

  1. WHAT Sweep found
  2. WHY it chose this evidence
  3. WHAT the evidence explains
  4. HOW confident it is and why
  5. WHAT contradictions exist (if any)

The narrator produces three output levels:
  - Executive Summary (1-2 sentences)
  - Detailed Breakdown (structured paragraphs)
  - Full Trace (technical debug view)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cortex import ReasoningResult


@dataclass
class Explanation:
    """A multi-level human-readable explanation."""
    executive_summary: str
    detailed_breakdown: str
    full_trace: str
    confidence_badge: str       # "HIGH", "MEDIUM", "LOW", "UNCERTAIN"
    decision_label: str         # "SUPPORTED", "REFUTED", "MIXED", "INSUFFICIENT"

    def to_dict(self) -> dict[str, str]:
        return {
            "executive_summary": self.executive_summary,
            "detailed_breakdown": self.detailed_breakdown,
            "full_trace": self.full_trace,
            "confidence_badge": self.confidence_badge,
            "decision_label": self.decision_label,
        }

    def __str__(self) -> str:
        return self.executive_summary


class ExplanationNarrator:
    """
    Generates human-readable explanations from reasoning results.

    Produces clear, intelligent explanations that tell the user
    exactly why Sweep reached its conclusion — in language a
    human would use to explain their own reasoning.
    """

    def narrate(self, result: ReasoningResult) -> Explanation:
        """Generate a full explanation from a reasoning result."""
        summary = self._executive_summary(result)
        detailed = self._detailed_breakdown(result)
        trace = self._full_trace(result)
        badge = self._confidence_badge(result.confidence)
        label = self._decision_label(result.decision)

        return Explanation(
            executive_summary=summary,
            detailed_breakdown=detailed,
            full_trace=trace,
            confidence_badge=badge,
            decision_label=label,
        )

    def _confidence_badge(self, confidence: float) -> str:
        if confidence >= 0.80:
            return "HIGH"
        if confidence >= 0.55:
            return "MEDIUM"
        if confidence >= 0.35:
            return "LOW"
        return "UNCERTAIN"

    def _decision_label(self, decision: str) -> str:
        return {
            "supported": "SUPPORTED",
            "refuted": "REFUTED",
            "mixed": "MIXED",
            "insufficient": "INSUFFICIENT",
        }.get(decision, decision.upper())

    def _executive_summary(self, result: ReasoningResult) -> str:
        """1-2 sentence summary of the conclusion."""
        q = result.query
        d = result.decision
        c = result.confidence

        if d == "supported":
            return (
                f"Based on analysis of {result.explanation_data.get('evidence_count', '?')} "
                f"evidence items, the answer to \"{q}\" is **yes** "
                f"(confidence: {c:.0%})."
            )
        if d == "refuted":
            return (
                f"Based on analysis of {result.explanation_data.get('evidence_count', '?')} "
                f"evidence items, the answer to \"{q}\" is **no** "
                f"(confidence: {(1 - c):.0%})."
            )
        if d == "mixed":
            return (
                f"The evidence for \"{q}\" is **divided** — "
                f"both supporting and contradicting signals were found "
                f"(confidence: {c:.0%})."
            )
        return (
            f"Insufficient evidence to answer \"{q}\" "
            f"confidently (confidence: {c:.0%})."
        )

    def _detailed_breakdown(self, result: ReasoningResult) -> str:
        """Structured paragraphs explaining the reasoning."""
        parts: list[str] = []
        data = result.explanation_data if isinstance(result.explanation_data, dict) else {}

        # ── Section 0: Brain division overview ──
        trace = result.trace
        parts.append("## Brain Processing")
        parts.append(f"  - **Hindbrain** (fast filter): {trace.hindbrain_ms:.1f}ms — "
                     f"salience score {trace.salience_score:.0%}")
        parts.append(f"  - **Midbrain** (routing): {trace.midbrain_ms:.1f}ms — "
                     f"attention-gated signal routing")
        parts.append(f"  - **Forebrain** (cognition): {trace.forebrain_ms:.1f}ms — "
                     f"processing centers + memory + action selection")
        if trace.bg_decisions > 0:
            parts.append(f"  - **Basal Ganglia**: {trace.bg_decisions} action proposals evaluated via RL")
        if trace.memory_recall_count > 0:
            parts.append(f"  - **Hippocampus**: {trace.memory_recall_count} similar past episodes recalled")

        # ── Section 1: What evidence was found ──
        evidence_items = data.get("evidence_items", [])
        if evidence_items:
            parts.append("## Evidence Found")
            for i, item in enumerate(evidence_items[:5], 1):
                text = item.get("text", "")[:150]
                conf = item.get("confidence", 0)
                source = item.get("source", "unknown")
                parts.append(
                    f"  {i}. \"{text}\" "
                    f"(confidence: {conf:.0%}, source: {source})"
                )

        # ── Section 2: Source credibility ──
        cred = data.get("credibility_summary", {})
        if cred:
            avg = cred.get("avg_score", 0)
            high = cred.get("high_trust_count", 0)
            parts.append(f"\n## Source Credibility")
            parts.append(f"  Average credibility: {avg:.0%}")
            parts.append(f"  High-trust sources: {high}")

        # ── Section 3: Causal connections ──
        causal = data.get("causal_links", [])
        if causal:
            parts.append(f"\n## Connections Found")
            for link in causal[:3]:
                ltype = link.get("type", "unknown")
                strength = link.get("strength", 0)
                shared = link.get("shared", [])
                parts.append(
                    f"  - {ltype.title()} link (strength: {strength:.0%}): "
                    f"shared terms: {', '.join(shared[:3])}"
                )

        # ── Section 4: Contradictions ──
        contra_count = data.get("contradictions_found", 0)
        if contra_count > 0:
            parts.append(f"\n## Contradictions Detected")
            parts.append(f"  {contra_count} conflicting evidence items found.")
            for cd in data.get("contradiction_details", [])[:3]:
                parts.append(
                    f"  - {cd.get('type', 'unknown')} conflict "
                    f"(strength: {cd.get('strength', 0):.0%})"
                )

        # ── Section 5: Why this decision ──
        parts.append(f"\n## Why This Decision")
        parts.append(f"  {result.reasoning}")

        # ── Section 6: Confidence factors ──
        if result.factors:
            parts.append(f"\n## Confidence Factors")
            for f in result.factors:
                name = f.get("name", "unknown")
                score = f.get("score", 0)
                detail = f.get("detail", "")
                parts.append(f"  - {name}: {score:.0%} — {detail}")

        return "\n".join(parts)

    def _full_trace(self, result: ReasoningResult) -> str:
        """Technical trace showing signal flow through all centers and brain divisions."""
        trace = result.trace
        parts = [
            f"=== NEURONAL REASONING TRACE ===",
            f"Query: {trace.query}",
            f"Input evidence: {trace.input_evidence_count} items",
            f"Total latency: {trace.total_latency_ms:.1f}ms",
            f"",
            f"--- BRAIN DIVISIONS ---",
            f"  Hindbrain (fast filter): {trace.hindbrain_ms:.1f}ms | salience: {trace.salience_score:.2f}",
            f"  Midbrain (routing): {trace.midbrain_ms:.1f}ms",
            f"  Forebrain (processing): {trace.forebrain_ms:.1f}ms",
            f"  BG decisions: {trace.bg_decisions} | Memory recalls: {trace.memory_recall_count}",
            f"",
            f"--- CENTER OUTPUTS ---",
        ]
        for center, count in trace.center_outputs.items():
            parts.append(f"  {center}: {count} signals")

        parts.extend([
            f"",
            f"Integration confidence: {trace.integration_confidence:.4f}",
            f"Decision: {trace.decision} (confidence: {trace.decision_confidence:.4f})",
            f"Reasoning: {trace.reasoning}",
            f"",
            f"Synapse State (after learning):",
        ])
        return "\n".join(parts)
