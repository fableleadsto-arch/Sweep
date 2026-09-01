"""
Human Reasoning — activates and runs human-like reasoning modules.

Modules:
  - Common Sense:      plausibility checking
  - Abductive:         hypothesis generation
  - Theory of Mind:    agent intent inference
  - Narrative:         story coherence assessment
  - Analogical:        cross-domain mapping
  - Causal:            cause-effect graph building
  - Counterfactual:    hypothetical scenario analysis
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HumanReasoningOutput:
    """Collected outputs from all activated human reasoning modules."""
    common_sense_plausibility: float = 0.5
    theory_of_mind_trust: float = 0.5
    abductive_hypotheses: int = 0
    narrative_coherence: float = 0.0
    analogical_mappings: int = 0
    causal_nodes: int = 0
    counterfactual_scenarios: int = 0


def run_human_reasoning(
    modules: list[str],
    query: str,
    evidence_texts: list[str],
    sources: list[str] | None,
    final_confidence: float,
    consensus_decision: str,
    forebrain: Any,
) -> HumanReasoningOutput:
    """Run the selected human reasoning modules and return collected outputs.

    Args:
        modules:           List of module names to activate.
        query:             The user query.
        evidence_texts:    Cleaned evidence text strings.
        sources:           Source identifiers (for Theory of Mind).
        final_confidence:  Current confidence score.
        consensus_decision: Current decision string.
        forebrain:         The Forebrain instance with all sub-modules.
    """
    out = HumanReasoningOutput()

    if "common_sense" in modules:
        cs_check = forebrain.common_sense.check_claim(query)
        out.common_sense_plausibility = cs_check.plausibility_score

    if "theory_of_mind" in modules and sources:
        for src in sources[:3]:
            forebrain.theory_of_mind.register_agent(name=src)
        to_agent_ids = list(forebrain.theory_of_mind._agents.keys())
        toms_result = forebrain.theory_of_mind.infer_intent(
            agent_id=to_agent_ids[0] if to_agent_ids else "unknown",
            text=query,
        )
        out.theory_of_mind_trust = (
            toms_result.intent_confidence if toms_result.should_trust else 0.3
        )

    if "abductive" in modules:
        abd_result = forebrain.abductive.reason(
            observations=evidence_texts[:8] if evidence_texts else [query],
        )
        out.abductive_hypotheses = len(abd_result.hypotheses)

    if "narrative" in modules:
        narr_result = forebrain.narrative.assess_narrative(
            evidence=evidence_texts[:10] if evidence_texts else [],
            query=query,
        )
        out.narrative_coherence = narr_result.overall_coherence

    if "analogical" in modules and re.search(
        r"\b(like|similar|compare|analogous|just as)\b", query.lower()
    ):
        out.analogical_mappings = _run_analogical(
            query, evidence_texts, forebrain,
        )

    if "causal" in modules and len(evidence_texts) >= 2:
        out.causal_nodes = _run_causal(evidence_texts, forebrain)

    if "counterfactual" in modules and len(evidence_texts) >= 2:
        out.counterfactual_scenarios = _run_counterfactual(
            evidence_texts, final_confidence, consensus_decision, forebrain,
        )

    return out


def _run_analogical(query: str, evidence_texts: list[str], forebrain: Any) -> int:
    """Run analogical reasoning and return mapping count."""
    from .analogical import Domain, DomainEntity as AnalogEntity

    src_domain = Domain(
        domain_id="query_ctx",
        name="query_context",
        entities=[
            AnalogEntity(
                name=query[:30], entity_type="concept",
                attributes={"query": query}, relations=[],
            )
        ],
    )
    tgt_domain = Domain(
        domain_id="evidence_ctx",
        name="evidence_context",
        entities=[
            AnalogEntity(
                name=e[:30], entity_type="concept",
                attributes={"text": e}, relations=[],
            )
            for e in evidence_texts[:3]
        ],
    )
    forebrain.analogical.register_domain(src_domain)
    forebrain.analogical.register_domain(tgt_domain)
    analogy = forebrain.analogical.find_analogy("query_ctx", "evidence_ctx")
    return len(analogy.mapping.entity_mappings) if analogy else 0


def _run_causal(evidence_texts: list[str], forebrain: Any) -> int:
    """Build causal graph from evidence and return node count."""
    for text in evidence_texts[:5]:
        forebrain.causal_model.add_node(
            name=text[:50], node_type="evidence",
        )
    for i in range(min(4, len(evidence_texts) - 1)):
        forebrain.causal_model.add_causal_link(
            source_name=evidence_texts[i][:50],
            target_name=evidence_texts[i + 1][:50],
            strength=0.5,
            edge_type="direct",
        )
    return len(forebrain.causal_model._nodes)


def _run_counterfactual(
    evidence_texts: list[str],
    confidence: float,
    decision: str,
    forebrain: Any,
) -> int:
    """Run counterfactual analysis and return scenario count."""
    forebrain.counterfactual.analyze_sensitivity(
        evidence=[
            {"text": e, "confidence": 0.5} for e in evidence_texts[:5]
        ],
        current_confidence=confidence,
        current_decision=decision,
    )
    return len(forebrain.counterfactual._analysis_history)
