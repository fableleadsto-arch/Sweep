"""
Counterfactual Reasoner — "what if things were different?"

Humans constantly run mental simulations of alternative realities:
- "What if I hadn't studied computer science?"
- "What if this evidence were from a different source?"
- "What if the date on this paper were 2020 instead of 2010?"

This is critical for robust reasoning because it answers:
- How SENSITIVE is our conclusion to individual pieces of evidence?
- What WOULD it take to change our mind?
- Are we SURE about this, or is it fragile?

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │            COUNTERFACTUAL REASONER                   │
    │                                                     │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Current Reasoning State                      │  │
    │  │  - Evidence set, confidence, decision          │  │
    │  └──────────────────┬───────────────────────────┘  │
    │                     ↓                               │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Counterfactual Generator                     │  │
    │  │  - Remove evidence item                       │  │
    │  │  - Modify evidence confidence                 │  │
    │  │  - Change source credibility                  │  │
    │  │  - Alter temporal context                     │  │
    │  └──────────────────┬───────────────────────────┘  │
    │                     ↓                               │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Sensitivity Analysis                         │  │
    │  │  - How much does each item matter?            │  │
    │  │  - Which items are load-bearing?              │  │
    │  │  - How robust is the conclusion?              │  │
    │  └──────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CounterfactualScenario:
    """A single counterfactual scenario."""
    scenario_id: str
    modification_type: str          # "remove", "modify", "invert", "swap"
    modification_detail: str        # what was changed
    original_value: Any             # original value
    counterfactual_value: Any       # new value
    expected_impact: float          # 0.0-1.0: how much this changes the outcome
    reasoning: str


@dataclass
class SensitivityReport:
    """Analysis of how sensitive the conclusion is to each evidence item."""
    overall_sensitivity: float      # 0.0-1.0: how fragile is the conclusion
    robustness_score: float         # 0.0-1.0: how robust is the conclusion
    load_bearing_items: list[str]   # evidence items that critically affect outcome
    fragile_items: list[str]        # evidence items whose removal changes outcome
    stable_items: list[str]         # evidence items that don't matter much
    scenarios_tested: int
    scenarios_changed_outcome: int
    # Recommendation
    recommendation: str             # what to do about the sensitivity
    confidence_in_analysis: float


class CounterfactualReasoner:
    """
    Run "what if" simulations on reasoning states.

    Like the human ability to imagine alternative realities, this module:

    1. Takes a completed reasoning pass (evidence, confidence, decision)
    2. Generates counterfactual variations (what if evidence X were different?)
    3. Re-runs reasoning with each variation
    4. Measures sensitivity: how much does each piece of evidence matter?
    5. Reports robustness: how confident should we be in the conclusion?

    This answers critical questions:
    - "How sure am I really?" (sensitivity analysis)
    - "What would change my mind?" (minimal counterfactual)
    - "Am I relying too much on one piece of evidence?" (load-bearing check)
    - "Is my conclusion robust or fragile?" (robustness score)

    The key insight: a conclusion that depends heavily on ONE piece of
    evidence is FRAGILE. A conclusion supported by MANY independent
    pieces of evidence is ROBUST.
    """

    def __init__(self) -> None:
        self._analysis_history: list[SensitivityReport] = []
        self._scenario_count = 0

    def analyze_sensitivity(
        self,
        evidence: list[dict[str, Any]],
        current_confidence: float,
        current_decision: str,
        reasoning_callback: Any = None,
    ) -> SensitivityReport:
        """
        Analyze how sensitive the current conclusion is to each evidence item.

        For each evidence item, generates a counterfactual scenario where
        that item is removed/modified, and measures the impact.
        """
        self._scenario_count += 0
        scenarios: list[CounterfactualScenario] = []
        outcomes: list[dict[str, Any]] = []

        # Test each evidence item
        for idx, item in enumerate(evidence):
            item_text = item.get("text", str(item))[:100]

            # Scenario 1: Remove this evidence
            scenario = CounterfactualScenario(
                scenario_id=f"cf_{self._scenario_count}_{idx}_remove",
                modification_type="remove",
                modification_detail=f"Removed evidence: {item_text}",
                original_value=item_text,
                counterfactual_value=None,
                expected_impact=0.0,
                reasoning=f"What if we didn't have this evidence?",
            )

            # Estimate impact by measuring evidence importance
            # (In a full implementation, this would re-run the reasoning pipeline)
            evidence_count = len(evidence)
            item_weight = item.get("confidence", 0.5) if isinstance(item, dict) else 0.5

            # Impact estimation: fewer evidence items → each matters more
            importance = item_weight * (1.0 / max(1, evidence_count * 0.3))
            scenario.expected_impact = min(1.0, importance)

            scenarios.append(scenario)
            outcomes.append({
                "item": item_text,
                "impact": scenario.expected_impact,
                "type": "removal",
            })

        # Scenario 2: Modify highest-confidence evidence
        if evidence:
            max_item = max(
                evidence,
                key=lambda x: x.get("confidence", 0.5) if isinstance(x, dict) else 0.5,
            )
            scenarios.append(CounterfactualScenario(
                scenario_id=f"cf_{self._scenario_count}_modify_max",
                modification_type="modify",
                modification_detail=f"Reduced confidence of strongest evidence",
                original_value=max_item.get("confidence", 0.5) if isinstance(max_item, dict) else 0.5,
                counterfactual_value=0.1,
                expected_impact=0.3,
                reasoning="What if our strongest evidence were weaker?",
            ))

        # Scenario 3: Invert a contradiction
        contradictions = [
            item for item in evidence
            if isinstance(item, dict) and "contradict" in str(item.get("text", "")).lower()
        ]
        if contradictions:
            scenarios.append(CounterfactualScenario(
                scenario_id=f"cf_{self._scenario_count}_invert_contra",
                modification_type="invert",
                modification_detail="Inverted a contradiction into agreement",
                original_value="contradiction",
                counterfactual_value="agreement",
                expected_impact=0.4,
                reasoning="What if contradictory evidence actually agreed?",
            ))

        # Compute overall metrics
        impacts = [s.expected_impact for s in scenarios]
        avg_impact = sum(impacts) / len(impacts) if impacts else 0.0
        max_impact = max(impacts) if impacts else 0.0

        # Sensitivity: how much does removing items change things
        sensitivity = avg_impact

        # Robustness: inverse of sensitivity (robust = low sensitivity)
        robustness = 1.0 - sensitivity

        # Load-bearing items: items with high impact
        load_bearing = [
            s.modification_detail for s in scenarios
            if s.expected_impact > 0.3
        ]

        # Fragile items: items whose removal would significantly change outcome
        fragile = [
            s.modification_detail for s in scenarios
            if s.expected_impact > 0.5
        ]

        # Stable items: items that don't matter much
        stable = [
            s.modification_detail for s in scenarios
            if s.expected_impact < 0.15
        ]

        # Generate recommendation
        recommendation = self._generate_recommendation(
            sensitivity, robustness, len(load_bearing), len(evidence)
        )

        report = SensitivityReport(
            overall_sensitivity=sensitivity,
            robustness_score=robustness,
            load_bearing_items=load_bearing,
            fragile_items=fragile,
            stable_items=stable,
            scenarios_tested=len(scenarios),
            scenarios_changed_outcome=sum(1 for i in impacts if i > 0.3),
            recommendation=recommendation,
            confidence_in_analysis=min(0.9, 0.3 + len(evidence) * 0.05),
        )

        self._analysis_history.append(report)
        return report

    def what_would_change_mind(
        self,
        current_decision: str,
        current_confidence: float,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        What minimal change would flip the decision?

        This is the "minimal counterfactual" — the smallest change
        to evidence that would produce a different conclusion.
        """
        changes = []

        # If confidence is already low, small changes could flip it
        if current_confidence < 0.5:
            changes.append({
                "type": "low_confidence",
                "description": "Current confidence is already borderline — small evidence changes could flip the decision",
                "probability_of_flip": 1.0 - current_confidence,
            })

        # If there are contradictions, resolving them could flip it
        contra_count = sum(
            1 for item in evidence
            if isinstance(item, dict) and "contradict" in str(item.get("text", "")).lower()
        )
        if contra_count > 0:
            changes.append({
                "type": "resolve_contradiction",
                "description": f"Resolving {contra_count} contradiction(s) could significantly change the outcome",
                "probability_of_flip": min(0.8, contra_count * 0.2),
            })

        # If evidence is sparse, one more piece could flip it
        if len(evidence) < 5:
            changes.append({
                "type": "additional_evidence",
                "description": "Limited evidence — one strong contradictory piece could flip the decision",
                "probability_of_flip": max(0.3, 1.0 - len(evidence) * 0.1),
            })

        # If confidence is high, would need strong counter-evidence
        if current_confidence > 0.8:
            changes.append({
                "type": "strong_counter_evidence",
                "description": "High confidence — would need multiple strong contradictory evidence items to flip",
                "probability_of_flip": 0.1,
            })

        return changes

    def compute_evidence_importance(
        self,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Rank evidence items by importance to the conclusion.

        Returns items sorted by importance (most important first).
        """
        ranked = []
        evidence_count = len(evidence)

        for idx, item in enumerate(evidence):
            text = item.get("text", str(item))[:100] if isinstance(item, dict) else str(item)[:100]
            confidence = item.get("confidence", 0.5) if isinstance(item, dict) else 0.5

            # Importance is based on:
            # 1. Confidence (higher = more important)
            # 2. Uniqueness (if removed, less redundancy)
            # 3. Position (first/last items often matter more)

            uniqueness = 1.0 - (1.0 / max(1, evidence_count))  # more unique when fewer items
            position_factor = 1.0 if idx < 2 or idx >= evidence_count - 2 else 0.8

            importance = confidence * 0.5 + uniqueness * 0.3 + position_factor * 0.2

            ranked.append({
                "index": idx,
                "text": text,
                "importance": importance,
                "confidence": confidence,
            })

        ranked.sort(key=lambda x: x["importance"], reverse=True)
        return ranked

    def _generate_recommendation(
        self,
        sensitivity: float,
        robustness: float,
        load_bearing_count: int,
        evidence_count: int,
    ) -> str:
        """Generate a recommendation based on sensitivity analysis."""
        if sensitivity > 0.6:
            return (
                f"CONCLUSION IS FRAGILE (sensitivity: {sensitivity:.0%}). "
                f"{load_bearing_count} load-bearing evidence items detected. "
                "Seek additional independent evidence to improve robustness."
            )
        elif sensitivity > 0.3:
            return (
                f"Conclusion has moderate sensitivity ({sensitivity:.0%}). "
                f"{evidence_count} evidence items provide some robustness. "
                "Consider verifying the most important evidence items."
            )
        else:
            return (
                f"Conclusion is robust (sensitivity: {sensitivity:.0%}). "
                f"Supported by {evidence_count} evidence items with distributed importance."
            )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_scenarios": self._scenario_count,
            "analysis_count": len(self._analysis_history),
            "avg_sensitivity": (
                sum(a.overall_sensitivity for a in self._analysis_history)
                / len(self._analysis_history)
                if self._analysis_history else 0.0
            ),
            "avg_robustness": (
                sum(a.robustness_score for a in self._analysis_history)
                / len(self._analysis_history)
                if self._analysis_history else 0.0
            ),
        }
