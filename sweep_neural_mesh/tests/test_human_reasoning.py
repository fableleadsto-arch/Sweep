"""
Tests for the 7 Human Reasoning Capabilities:
    1. Analogical Reasoning
    2. Causal World Model
    3. Counterfactual Reasoning
    4. Common Sense Knowledge Base
    5. Theory of Mind
    6. Abductive Reasoning
    7. Narrative Coherence
"""
from __future__ import annotations
import time
import pytest

# ── Analogical Reasoning ──
from sweep_neural_mesh.neurons.analogical import (
    AnalogicalReasoner, Domain, DomainEntity, StructuralMapping, Analogy,
)

# ── Causal Model ──
from sweep_neural_mesh.neurons.causal_model import (
    CausalModel, CausalNode, CausalEdge, InterventionResult,
)

# ── Counterfactual ──
from sweep_neural_mesh.neurons.counterfactual import (
    CounterfactualReasoner, CounterfactualScenario, SensitivityReport,
)

# ── Common Sense ──
from sweep_neural_mesh.neurons.common_sense import (
    CommonSense, CommonSenseRule, CommonSenseCheck,
)

# ── Theory of Mind ──
from sweep_neural_mesh.neurons.theory_of_mind import (
    TheoryOfMind, AgentState, IntentAssessment, SocialContext,
)

# ── Abductive ──
from sweep_neural_mesh.neurons.abductive import (
    AbductiveReasoner, Hypothesis, AbductiveResult,
)

# ── Narrative ──
from sweep_neural_mesh.neurons.narrative import (
    NarrativeEngine, NarrativeEntity, StoryArc, NarrativeAssessment,
)


# ═══════════════════════════════════════════════════════════════════
# 1. ANALOGICAL REASONING
# ═══════════════════════════════════════════════════════════════════

class TestAnalogicalReasoner:
    def test_init(self):
        r = AnalogicalReasoner()
        assert r.analogy_count == 0
        assert r.domain_count == 0

    def test_register_domain(self):
        r = AnalogicalReasoner()
        d = Domain(domain_id="d1", name="test", entities=[])
        r.register_domain(d)
        assert r.domain_count == 1

    def test_add_entity_to_domain(self):
        r = AnalogicalReasoner()
        d = Domain(domain_id="physics", name="physics", entities=[])
        r.register_domain(d)
        e = DomainEntity(name="electron", entity_type="particle", attributes={"charge": "-1"}, relations=[])
        r.add_entity_to_domain("physics", e)
        dom = r.get_domain("physics")
        assert dom is not None
        assert len(dom.entities) == 1

    def test_find_analogy_basic(self):
        r = AnalogicalReasoner()
        src = Domain(
            domain_id="src", name="source",
            entities=[
                DomainEntity(name="sun", entity_type="energy_source", attributes={"energy": "fusion"}, relations=[]),
                DomainEntity(name="earth", entity_type="energy_receiver", attributes={"energy": "solar"}, relations=[]),
            ],
        )
        tgt = Domain(
            domain_id="tgt", name="target",
            entities=[
                DomainEntity(name="battery", entity_type="energy_source", attributes={"energy": "chemical"}, relations=[]),
                DomainEntity(name="device", entity_type="energy_receiver", attributes={"energy": "battery"}, relations=[]),
            ],
        )
        r.register_domain(src)
        r.register_domain(tgt)
        analogy = r.find_analogy("src", "tgt")
        assert analogy is not None
        assert isinstance(analogy, Analogy)
        assert r.analogy_count == 1

    def test_analogy_has_mappings(self):
        r = AnalogicalReasoner()
        src = Domain(
            domain_id="brain", name="brain",
            entities=[
                DomainEntity(name="neuron", entity_type="processor", attributes={"signal": "electrical"}, relations=[]),
                DomainEntity(name="synapse", entity_type="connector", attributes={"type": "chemical"}, relations=[]),
            ],
        )
        tgt = Domain(
            domain_id="computer", name="computer",
            entities=[
                DomainEntity(name="cpu", entity_type="processor", attributes={"signal": "electrical"}, relations=[]),
                DomainEntity(name="bus", entity_type="connector", attributes={"type": "electrical"}, relations=[]),
            ],
        )
        r.register_domain(src)
        r.register_domain(tgt)
        analogy = r.find_analogy("brain", "computer")
        assert analogy is not None
        assert len(analogy.mapping.entity_mappings) > 0

    def test_analogy_explanation(self):
        r = AnalogicalReasoner()
        src = Domain(
            domain_id="water", name="water_system",
            entities=[
                DomainEntity(name="pump", entity_type="mover", attributes={"flow": "water"}, relations=[]),
                DomainEntity(name="pipe", entity_type="conduit", attributes={"carries": "water"}, relations=[]),
            ],
        )
        tgt = Domain(
            domain_id="blood", name="blood_system",
            entities=[
                DomainEntity(name="heart", entity_type="mover", attributes={"flow": "blood"}, relations=[]),
                DomainEntity(name="artery", entity_type="conduit", attributes={"carries": "blood"}, relations=[]),
            ],
        )
        r.register_domain(src)
        r.register_domain(tgt)
        analogy = r.find_analogy("water", "blood")
        assert analogy is not None
        assert len(analogy.explanation) > 0


# ═══════════════════════════════════════════════════════════════════
# 2. CAUSAL WORLD MODEL
# ═══════════════════════════════════════════════════════════════════

class TestCausalModel:
    def test_init(self):
        m = CausalModel()
        stats = m.get_graph_stats()
        assert stats["node_count"] == 0

    def test_add_node(self):
        m = CausalModel()
        node = m.add_node("rain", node_type="cause")
        assert node.name == "rain"
        stats = m.get_graph_stats()
        assert stats["node_count"] == 1

    def test_add_causal_link(self):
        m = CausalModel()
        m.add_node("rain")
        m.add_node("wet_ground")
        edge = m.add_causal_link("rain", "wet_ground", strength=0.8)
        assert edge is not None
        stats = m.get_graph_stats()
        assert stats["edge_count"] == 1

    def test_observe_causation(self):
        m = CausalModel()
        m.observe_causation("smoking", "cancer", strength=0.7)
        stats = m.get_graph_stats()
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1

    def test_query_causes(self):
        m = CausalModel()
        m.observe_causation("smoking", "cancer")
        m.observe_causation("asbestos", "cancer")
        causes = m.query_causes("cancer")
        assert len(causes) == 2

    def test_query_effects(self):
        m = CausalModel()
        m.observe_causation("smoking", "cancer")
        m.observe_causation("smoking", "heart_disease")
        effects = m.query_effects("smoking")
        assert len(effects) == 2

    def test_propagate_effect(self):
        m = CausalModel()
        m.observe_causation("a", "b")
        m.observe_causation("b", "c")
        effects = m.propagate_effect("a", max_depth=2)
        assert isinstance(effects, dict)
        assert len(effects) > 0

    def test_do_intervention(self):
        m = CausalModel()
        m.observe_causation("rain", "wet_ground")
        m.observe_causation("sprinkler", "wet_ground")
        result = m.do_intervention("rain", "wet_ground")
        assert isinstance(result, InterventionResult)

    def test_get_causal_chains(self):
        m = CausalModel()
        m.observe_causation("a", "b")
        m.observe_causation("b", "c")
        m.observe_causation("c", "d")
        chains = m.get_causal_chains("a", "d")
        assert len(chains) > 0


# ═══════════════════════════════════════════════════════════════════
# 3. COUNTERFACTUAL REASONING
# ═══════════════════════════════════════════════════════════════════

class TestCounterfactualReasoner:
    def test_init(self):
        r = CounterfactualReasoner()
        stats = r.stats
        assert stats["analysis_count"] == 0

    def test_analyze_sensitivity(self):
        r = CounterfactualReasoner()
        result = r.analyze_sensitivity(
            evidence=[
                {"text": "Python has many libraries", "confidence": 0.8},
                {"text": "Python is easy to learn", "confidence": 0.7},
            ],
            current_confidence=0.75,
            current_decision="python_is_good",
        )
        assert isinstance(result, SensitivityReport)
        assert r.stats["analysis_count"] == 1

    def test_analyze_sensitivity_minimal(self):
        r = CounterfactualReasoner()
        result = r.analyze_sensitivity(
            evidence=[{"text": "evidence A"}, {"text": "evidence B"}],
            current_confidence=0.6,
            current_decision="yes",
        )
        assert result.scenarios_tested >= 2

    def test_what_would_change_mind(self):
        r = CounterfactualReasoner()
        result = r.what_would_change_mind(
            current_decision="ai_will_replace_jobs",
            current_confidence=0.7,
            evidence=[
                {"text": "automation is increasing"},
                {"text": "but creativity is uniquely human"},
            ],
        )
        assert isinstance(result, list)

    def test_compute_evidence_importance(self):
        r = CounterfactualReasoner()
        result = r.compute_evidence_importance(
            evidence=[
                {"text": "temperature records", "confidence": 0.9},
                {"text": "ice core data", "confidence": 0.85},
                {"text": "sea level rise", "confidence": 0.8},
            ],
        )
        assert isinstance(result, list)
        assert len(result) == 3
        # Should be sorted by importance
        importances = [item["importance"] for item in result]
        assert importances == sorted(importances, reverse=True)

    def test_stats(self):
        r = CounterfactualReasoner()
        r.analyze_sensitivity(
            evidence=[{"text": "e1", "confidence": 0.5}],
            current_confidence=0.5,
            current_decision="test",
        )
        r.analyze_sensitivity(
            evidence=[{"text": "e2", "confidence": 0.6}],
            current_confidence=0.6,
            current_decision="test2",
        )
        stats = r.stats
        assert stats["analysis_count"] == 2


# ═══════════════════════════════════════════════════════════════════
# 4. COMMON SENSE KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════

class TestCommonSense:
    def test_init_has_defaults(self):
        cs = CommonSense()
        assert cs.rule_count > 0

    def test_add_rule(self):
        cs = CommonSense()
        initial = cs.rule_count
        cs.add_rule("test", "when X happens", "Y follows", 0.8)
        assert cs.rule_count == initial + 1

    def test_check_claim_supported(self):
        cs = CommonSense()
        check = cs.check_claim("objects fall when dropped due to gravity")
        assert isinstance(check, CommonSenseCheck)
        assert check.plausibility_score > 0.0

    def test_check_claim_violation(self):
        cs = CommonSense()
        check = cs.check_claim("objects float upward when dropped in normal conditions")
        # Should be low plausibility or have violated rules
        assert check.plausibility_score <= 0.5 or len(check.violated_rules) > 0

    def test_check_claim_neutral(self):
        cs = CommonSense()
        check = cs.check_claim("quantum entanglement is interesting to physicists")
        # Neutral — no strong common sense signal
        assert 0.0 <= check.plausibility_score <= 1.0

    def test_learn_from_episode(self):
        cs = CommonSense()
        initial_rules = cs.rule_count
        cs.learn_from_episode(
            claim="fire is hot",
            outcome="people get burned",
            was_plausible=True,
        )
        assert cs.rule_count >= initial_rules

    def test_get_rules_by_category(self):
        cs = CommonSense()
        physical = cs.get_rules_by_category("physical")
        assert len(physical) > 0
        for rule in physical:
            assert rule.category == "physical"

    def test_stats(self):
        cs = CommonSense()
        stats = cs.stats
        assert "total_rules" in stats
        assert "by_category" in stats
        assert stats["total_rules"] > 0


# ═══════════════════════════════════════════════════════════════════
# 5. THEORY OF MIND
# ═══════════════════════════════════════════════════════════════════

class TestTheoryOfMind:
    def test_init(self):
        tom = TheoryOfMind()
        assert tom.agent_count == 0

    def test_register_agent(self):
        tom = TheoryOfMind()
        agent = tom.register_agent("Alice", goal="understand the system")
        assert tom.agent_count == 1
        assert agent.name == "Alice"

    def test_infer_intent_inform(self):
        tom = TheoryOfMind()
        agent = tom.register_agent("Researcher")
        result = tom.infer_intent(agent.agent_id, "According to a 2024 study, X causes Y")
        assert result.primary_intent == "inform"

    def test_infer_intent_persuade(self):
        tom = TheoryOfMind()
        agent = tom.register_agent("Salesperson")
        result = tom.infer_intent(agent.agent_id, "You should buy this product now, it's the best")
        assert result.primary_intent == "persuade"

    def test_infer_intent_deceive(self):
        tom = TheoryOfMind()
        agent = tom.register_agent("Spammer")
        result = tom.infer_intent(agent.agent_id, "You won't believe this secret miracle cure click here")
        assert result.primary_intent == "deceive"

    def test_infer_intent_unknown_agent(self):
        tom = TheoryOfMind()
        result = tom.infer_intent("nonexistent", "test")
        assert result.intent_confidence == 0.0

    def test_update_agent_state(self):
        tom = TheoryOfMind()
        agent = tom.register_agent("Bob")
        updated = tom.update_agent_state(agent.agent_id, goal="new goal", affect="excited")
        assert updated is True
        assert tom.get_agent(agent.agent_id).goal_state == "new goal"

    def test_assess_credibility_adjustment(self):
        tom = TheoryOfMind()
        tom.register_agent("Trusted Source")
        adj, reason = tom.assess_credibility_adjustment("Trusted Source", 0.8)
        assert 0.0 <= adj <= 1.0
        assert len(reason) > 0

    def test_detect_information_gaps(self):
        tom = TheoryOfMind()
        agent = tom.register_agent("Curious")
        gaps = tom.detect_information_gaps("I'm not sure about this, maybe X?", agent.agent_id)
        assert len(gaps) > 0

    def test_stats(self):
        tom = TheoryOfMind()
        tom.register_agent("A")
        tom.register_agent("B")
        stats = tom.stats
        assert stats["agent_count"] == 2


# ═══════════════════════════════════════════════════════════════════
# 6. ABDUCTIVE REASONING
# ═══════════════════════════════════════════════════════════════════

class TestAbductiveReasoner:
    def test_init(self):
        r = AbductiveReasoner()
        assert r.hypothesis_count == 0

    def test_reason_basic(self):
        r = AbductiveReasoner()
        result = r.reason(
            observations=["The ground is wet", "The sky is dark"],
            context={"location": "outdoors"},
        )
        assert isinstance(result, AbductiveResult)
        assert len(result.hypotheses) > 0
        assert result.best_explanation is not None

    def test_reason_generates_multiple_hypotheses(self):
        r = AbductiveReasoner()
        result = r.reason(observations=["Server is slow", "Error rate increased", "Latency spiked"])
        assert len(result.hypotheses) >= 2

    def test_hypotheses_are_ranked(self):
        r = AbductiveReasoner()
        result = r.reason(observations=["obs1", "obs2", "obs3"])
        scores = [h.overall_score for h in result.hypotheses]
        assert scores == sorted(scores, reverse=True)

    def test_reasoning_chain(self):
        r = AbductiveReasoner()
        result = r.reason(observations=["test obs"])
        assert len(result.reasoning_chain) > 0

    def test_learn_from_outcome(self):
        r = AbductiveReasoner()
        r.learn_from_outcome(
            hypothesis="it rained",
            observations=["ground is wet"],
            was_correct=True,
        )
        assert r.stats["learned_patterns"] > 0

    def test_max_hypotheses(self):
        r = AbductiveReasoner()
        result = r.reason(observations=["a", "b", "c"], max_hypotheses=3)
        assert len(result.hypotheses) <= 3

    def test_stats(self):
        r = AbductiveReasoner()
        r.reason(["obs1", "obs2"])
        stats = r.stats
        assert stats["reasoning_sessions"] == 1
        assert stats["total_hypotheses"] > 0


# ═══════════════════════════════════════════════════════════════════
# 7. NARRATIVE COHERENCE
# ═══════════════════════════════════════════════════════════════════

class TestNarrativeEngine:
    def test_init(self):
        e = NarrativeEngine()
        assert e.stats["assessments"] == 0

    def test_assess_narrative_basic(self):
        e = NarrativeEngine()
        result = e.assess_narrative(
            evidence=["Initially the system was slow", "However the database was overloaded", "Therefore we optimized the queries"],
            query="What happened?",
        )
        assert isinstance(result, NarrativeAssessment)
        assert len(result.story_arcs) > 0

    def test_coherence_score(self):
        e = NarrativeEngine()
        result = e.assess_narrative(
            evidence=["First X happened", "Because of X, Y happened", "Therefore Z happened"],
        )
        assert result.overall_coherence > 0.3

    def test_detects_fallacies(self):
        e = NarrativeEngine()
        result = e.assess_narrative(
            evidence=["After the rain, the ground was wet, therefore the rain caused it"],
        )
        assert len(result.narrative_fallacies) > 0

    def test_identifies_missing_context(self):
        e = NarrativeEngine()
        result = e.assess_narrative(
            evidence=["Something happened and it was clearly important"],
        )
        assert len(result.missing_context) > 0

    def test_generates_recommendations(self):
        e = NarrativeEngine()
        result = e.assess_narrative(
            evidence=["The system failed", "It was obvious that the server crashed"],
        )
        assert len(result.recommendations) > 0

    def test_entity_extraction(self):
        e = NarrativeEngine()
        result = e.assess_narrative(
            evidence=["Google released a new product", "Microsoft responded quickly"],
        )
        all_entities = [entity for arc in result.story_arcs for entity in arc.entities]
        assert len(all_entities) > 0

    def test_empty_evidence(self):
        e = NarrativeEngine()
        result = e.assess_narrative(evidence=[])
        assert result.overall_coherence <= 0.5

    def test_stats(self):
        e = NarrativeEngine()
        e.assess_narrative(["test evidence"])
        stats = e.stats
        assert stats["assessments"] == 1


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION: All 7 in Forebrain
# ═══════════════════════════════════════════════════════════════════

class TestHumanReasoningIntegration:
    def test_forebrain_has_all_modules(self):
        from sweep_neural_mesh.neurons.brain import Forebrain
        fb = Forebrain()
        assert fb.analogical is not None
        assert fb.causal_model is not None
        assert fb.counterfactual is not None
        assert fb.common_sense is not None
        assert fb.theory_of_mind is not None
        assert fb.abductive is not None
        assert fb.narrative is not None

    def test_forebrain_common_sense_works(self):
        from sweep_neural_mesh.neurons.brain import Forebrain
        fb = Forebrain()
        check = fb.common_sense.check_claim("objects fall when dropped")
        assert check.plausibility_score > 0.0

    def test_forebrain_abductive_works(self):
        from sweep_neural_mesh.neurons.brain import Forebrain
        fb = Forebrain()
        result = fb.abductive.reason(observations=["obs1", "obs2"])
        assert len(result.hypotheses) > 0

    def test_forebrain_narrative_works(self):
        from sweep_neural_mesh.neurons.brain import Forebrain
        fb = Forebrain()
        result = fb.narrative.assess_narrative(
            evidence=["Initially X", "However Y happened", "Therefore Z"],
        )
        assert len(result.story_arcs) > 0

    def test_forebrain_theory_of_mind_works(self):
        from sweep_neural_mesh.neurons.brain import Forebrain
        fb = Forebrain()
        agent = fb.theory_of_mind.register_agent("TestAgent")
        result = fb.theory_of_mind.infer_intent(agent.agent_id, "According to research, X is true")
        assert result.primary_intent == "inform"

    def test_forebrain_counterfactual_works(self):
        from sweep_neural_mesh.neurons.brain import Forebrain
        fb = Forebrain()
        result = fb.counterfactual.analyze_sensitivity(
            evidence=[{"text": "evidence1", "confidence": 0.5}],
            current_confidence=0.6,
            current_decision="test",
        )
        assert result is not None
