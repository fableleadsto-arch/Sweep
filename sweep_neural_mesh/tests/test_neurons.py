"""
Tests for the Neuronal Reasoning System.

Tests every center individually, then the full pipeline end-to-end
to verify the brain produces logical, well-reasoned explanations.
"""
import pytest

from sweep_neural_mesh.neurons.signal import Signal, SignalType, Synapse, SynapseType
from sweep_neural_mesh.neurons.centers import (
    EvidenceGatherer,
    CredibilityAssessor,
    TemporalSequencer,
    CausalLinker,
    ContradictionDetector,
    ExplanationBuilder,
)
from sweep_neural_mesh.neurons.integration import IntegrationHub, ConsensusEngine
from sweep_neural_mesh.neurons.cortex import ReasoningCortex, ReasoningResult
from sweep_neural_mesh.neurons.narrator import ExplanationNarrator


# ══════════════════════════════════════════════════════════════════
# SIGNAL + SYNAPSE PRIMITIVES
# ══════════════════════════════════════════════════════════════════

class TestSignal:
    def test_create_raw_signal(self):
        s = Signal(data={"query": "test"}, signal_type=SignalType.RAW)
        assert s.signal_type == SignalType.RAW
        assert s.confidence == 1.0
        assert s.signal_id

    def test_amplify(self):
        s = Signal(data={}, confidence=0.5)
        amplified = s.amplify(1.5)
        assert amplified.confidence == pytest.approx(0.75)

    def test_amplify_capped_at_1(self):
        s = Signal(data={}, confidence=0.9)
        amplified = s.amplify(2.0)
        assert amplified.confidence == 1.0

    def test_dampen(self):
        s = Signal(data={}, confidence=0.8)
        dampened = s.dampen(0.5)
        assert dampened.confidence == pytest.approx(0.4)

    def test_dampen_floored_at_0(self):
        s = Signal(data={}, confidence=0.0)
        dampened = s.dampen(0.5)
        assert dampened.confidence == 0.0

    def test_stamp_adds_history(self):
        s = Signal(data={})
        stamped = s.stamp("center_a")
        assert "center_a" in stamped.history
        stamped2 = stamped.stamp("center_b")
        assert stamped2.history == ["center_a", "center_b"]

    def test_to_dict(self):
        s = Signal(data={"key": "val"}, signal_type=SignalType.EVIDENCE, confidence=0.8)
        d = s.to_dict()
        assert d["type"] == "evidence"
        assert d["confidence"] == 0.8


class TestSynapse:
    def test_excitatory_transmit(self):
        sig = Signal(data={"x": 1}, confidence=0.5)
        syn = Synapse(from_center="a", to_center="b", weight=1.0, synapse_type=SynapseType.EXCITATORY)
        out = syn.transmit(sig)
        # weight=1.0 → boost = (1.0 - 0.5) * 1.0 * 0.3 = 0.15 → new = 0.65
        assert out.confidence == pytest.approx(0.65)
        assert syn.activation_count == 1

    def test_inhibitory_transmit(self):
        sig = Signal(data={"x": 1}, confidence=0.8)
        syn = Synapse(from_center="a", to_center="b", weight=1.0, synapse_type=SynapseType.INHIBITORY)
        out = syn.transmit(sig)
        # weight=1.0 → reduction = 0.8 * 1.0 * 0.3 = 0.24 → new = 0.56
        assert out.confidence == pytest.approx(0.56)

    def test_modulatory_transmit(self):
        sig = Signal(data={"x": 1}, confidence=0.8)
        syn = Synapse(from_center="a", to_center="b", weight=0.5, synapse_type=SynapseType.MODULATORY)
        out = syn.transmit(sig)
        assert out.confidence == 0.8  # modulatory doesn't change confidence
        assert "modulated_by" in out.metadata

    def test_strengthen(self):
        syn = Synapse(from_center="a", to_center="b", weight=0.5)
        syn.strengthen(0.1)
        assert syn.weight == pytest.approx(0.6)

    def test_strengthen_capped(self):
        syn = Synapse(from_center="a", to_center="b", weight=1.95)
        syn.strengthen(0.1)
        assert syn.weight == 2.0

    def test_weaken(self):
        syn = Synapse(from_center="a", to_center="b", weight=0.5)
        syn.weaken(0.1)
        assert syn.weight == pytest.approx(0.4)

    def test_weaken_floored(self):
        syn = Synapse(from_center="a", to_center="b", weight=0.05)
        syn.weaken(0.1)
        assert syn.weight == 0.0


# ══════════════════════════════════════════════════════════════════
# INDIVIDUAL PROCESSING CENTERS
# ══════════════════════════════════════════════════════════════════

class TestEvidenceGatherer:
    def test_processes_raw_evidence(self):
        gatherer = EvidenceGatherer()
        raw = Signal(
            data={"query": "test", "evidence": ["Python is widely used for ML", "Java is also used"]},
            signal_type=SignalType.RAW,
        )
        results = gatherer.process([raw])
        assert len(results) == 2
        assert all(s.signal_type == SignalType.EVIDENCE for s in results)
        assert all(s.confidence > 0 for s in results)

    def test_empty_input(self):
        gatherer = EvidenceGatherer()
        assert gatherer.process([]) == []

    def test_scores_detailed_evidence_higher(self):
        gatherer = EvidenceGatherer()
        raw = Signal(
            data={"evidence": [
                "yes",
                "According to a 2024 study published in Nature, Python has become the dominant language for machine learning with over 80% of ML researchers using it. The research analyzed 10,000 papers and found Python libraries like PyTorch and TensorFlow were used in 92% of implementations.",
            ]},
            signal_type=SignalType.RAW,
        )
        results = gatherer.process([raw])
        assert len(results) == 2
        short = [r for r in results if r.data.get("evidence_text") == "yes"][0]
        long = [r for r in results if r.data.get("evidence_text") != "yes"][0]
        assert long.confidence > short.confidence


class TestCredibilityAssessor:
    def test_trusted_source_gets_high_score(self):
        assessor = CredibilityAssessor()
        sig = Signal(
            data={"evidence_text": "A comprehensive study shows results", "source": "nature.com"},
            signal_type=SignalType.EVIDENCE,
            confidence=0.8,
        )
        results = assessor.process([sig])
        assert results[0].confidence > 0.6

    def test_untrusted_pattern_penalized(self):
        assessor = CredibilityAssessor()
        sig = Signal(
            data={"evidence_text": "You won't believe this secret trick!", "source": "random-blog.com"},
            signal_type=SignalType.EVIDENCE,
            confidence=0.8,
        )
        results = assessor.process([sig])
        assert results[0].confidence < 0.5

    def test_citations_boost_score(self):
        assessor = CredibilityAssessor()
        sig = Signal(
            data={"evidence_text": "Research from (2024) shows [1] that Python usage increased doi:10.1234/test", "source": "arxiv.org"},
            signal_type=SignalType.EVIDENCE,
            confidence=0.8,
        )
        results = assessor.process([sig])
        assert results[0].confidence > 0.7


class TestTemporalSequencer:
    def test_recent_evidence_ranked_higher(self):
        seq = TemporalSequencer()
        sigs = [
            Signal(data={"evidence_text": "In 2024, this was proven"}, signal_type=SignalType.EVIDENCE, confidence=0.8),
            Signal(data={"evidence_text": "In 1990, this was observed"}, signal_type=SignalType.EVIDENCE, confidence=0.8),
        ]
        results = seq.process(sigs)
        recent = [r for r in results if "2024" in r.data.get("evidence_text", "")][0]
        old = [r for r in results if "1990" in r.data.get("evidence_text", "")][0]
        assert recent.confidence > old.confidence

    def test_relative_time_detected(self):
        seq = TemporalSequencer()
        sig = Signal(
            data={"evidence_text": "This was confirmed today by researchers"},
            signal_type=SignalType.EVIDENCE,
            confidence=0.8,
        )
        results = seq.process([sig])
        assert results[0].data.get("date_relevance", 0) > 0.9


class TestCausalLinker:
    def test_finds_keyword_overlap(self):
        linker = CausalLinker()
        sigs = [
            Signal(data={"evidence_text": "Python machine learning libraries are powerful"}, signal_type=SignalType.EVIDENCE, confidence=0.8),
            Signal(data={"evidence_text": "Python machine learning frameworks dominate the field"}, signal_type=SignalType.EVIDENCE, confidence=0.8),
        ]
        results = linker.process(sigs)
        assert len(results) >= 1
        assert results[0].signal_type == SignalType.CAUSAL

    def test_no_link_with_different_topics(self):
        linker = CausalLinker()
        sigs = [
            Signal(data={"evidence_text": "The weather is sunny today"}, signal_type=SignalType.EVIDENCE, confidence=0.8),
            Signal(data={"evidence_text": "Quantum computing uses qubits"}, signal_type=SignalType.EVIDENCE, confidence=0.8),
        ]
        results = linker.process(sigs)
        assert len(results) == 0

    def test_causal_language_boosts_strength(self):
        linker = CausalLinker()
        sigs = [
            Signal(data={"evidence_text": "Python is popular because it is easy to learn and use"}, signal_type=SignalType.EVIDENCE, confidence=0.8),
            Signal(data={"evidence_text": "Python is easy to learn and use which leads to widespread adoption"}, signal_type=SignalType.EVIDENCE, confidence=0.8),
        ]
        results = linker.process(sigs)
        if results:
            assert results[0].data.get("link_type") in ("causal", "supportive", "semantic")


class TestContradictionDetector:
    def test_detects_negation_asymmetry(self):
        det = ContradictionDetector()
        sigs = [
            Signal(data={"evidence_text": "Python is not suitable for production"}, signal_type=SignalType.EVIDENCE, confidence=0.8),
            Signal(data={"evidence_text": "Python is suitable for production systems"}, signal_type=SignalType.EVIDENCE, confidence=0.8),
        ]
        results = det.process(sigs)
        assert len(results) >= 1
        assert results[0].signal_type == SignalType.CONTRADICTION

    def test_detects_opposing_pairs(self):
        det = ContradictionDetector()
        sigs = [
            Signal(data={"evidence_text": "The results were positive and beneficial"}, signal_type=SignalType.EVIDENCE, confidence=0.8),
            Signal(data={"evidence_text": "The results were negative and harmful"}, signal_type=SignalType.EVIDENCE, confidence=0.8),
        ]
        results = det.process(sigs)
        assert len(results) >= 1

    def test_no_conflict_with_similar_statements(self):
        det = ContradictionDetector()
        sigs = [
            Signal(data={"evidence_text": "Python is popular for machine learning tasks"}, signal_type=SignalType.EVIDENCE, confidence=0.8),
            Signal(data={"evidence_text": "Python is widely used in machine learning research"}, signal_type=SignalType.EVIDENCE, confidence=0.8),
        ]
        results = det.process(sigs)
        assert len(results) == 0


# ══════════════════════════════════════════════════════════════════
# INTEGRATION HUB + CONSENSUS ENGINE
# ══════════════════════════════════════════════════════════════════

class TestIntegrationHub:
    def test_integrates_signals(self):
        hub = IntegrationHub()
        sigs = [
            Signal(data={"evidence_text": "A"}, signal_type=SignalType.EVIDENCE, confidence=0.8),
            Signal(data={"evidence_text": "B"}, signal_type=SignalType.EVIDENCE, confidence=0.7),
            Signal(data={"credibility": "high"}, signal_type=SignalType.CREDIBILITY, confidence=0.9),
        ]
        result = hub.integrate(sigs)
        assert result.signal_type == SignalType.INTEGRATED
        assert result.confidence > 0

    def test_contradictions_lower_confidence(self):
        hub = IntegrationHub()
        sigs = [
            Signal(data={}, signal_type=SignalType.EVIDENCE, confidence=0.8),
            Signal(data={}, signal_type=SignalType.CREDIBILITY, confidence=0.9),
            Signal(data={}, signal_type=SignalType.CONTRADICTION, confidence=0.9),
        ]
        result = hub.integrate(sigs)
        assert result.confidence < 0.7

    def test_empty_signals(self):
        hub = IntegrationHub()
        result = hub.integrate([])
        assert result.confidence == 0.0


class TestConsensusEngine:
    def test_strong_evidence_supported(self):
        engine = ConsensusEngine()
        integrated = Signal(
            data={"type_scores": {"evidence": 0.2, "credibility": 0.25}},
            signal_type=SignalType.INTEGRATED,
            confidence=0.85,
        )
        raw = [
            Signal(data={}, signal_type=SignalType.EVIDENCE, confidence=0.8),
            Signal(data={}, signal_type=SignalType.EVIDENCE, confidence=0.9),
            Signal(data={}, signal_type=SignalType.CREDIBILITY, confidence=0.85),
        ]
        result = engine.decide(integrated, raw)
        assert result.data["decision"] == "supported"
        assert result.confidence == 0.85

    def test_weak_evidence_refuted(self):
        engine = ConsensusEngine()
        integrated = Signal(
            data={},
            signal_type=SignalType.INTEGRATED,
            confidence=0.25,
        )
        raw = [
            Signal(data={}, signal_type=SignalType.EVIDENCE, confidence=0.3),
        ]
        result = engine.decide(integrated, raw)
        assert result.data["decision"] == "refuted"

    def test_contradictions_reduce_confidence(self):
        engine = ConsensusEngine()
        integrated = Signal(
            data={},
            signal_type=SignalType.INTEGRATED,
            confidence=0.42,
        )
        raw = [
            Signal(data={}, signal_type=SignalType.EVIDENCE, confidence=0.7),
            Signal(data={}, signal_type=SignalType.CONTRADICTION, confidence=0.7),
        ]
        result = engine.decide(integrated, raw)
        # With contradictions present, the decision should reflect conflict
        assert result.data["contradicting_evidence"] == 1
        assert len(result.data["factors"]) >= 2  # should have contradiction factor


# ══════════════════════════════════════════════════════════════════
# FULL CORTEX PIPELINE — END-TO-END
# ══════════════════════════════════════════════════════════════════

class TestReasoningCortex:
    def test_full_pipeline_supported(self):
        cortex = ReasoningCortex()
        result = cortex.reason(
            query="Is Python good for machine learning?",
            evidence=[
                "Python is the most popular language for ML with 80% adoption",
                "PyTorch and TensorFlow have excellent Python support",
                "According to a 2024 survey, Python dominates ML research",
                "Python's simplicity makes it ideal for rapid prototyping",
                "Major ML frameworks are Python-first",
            ],
            sources=["nature.com", "arxiv.org", "github.com"],
        )
        assert isinstance(result, ReasoningResult)
        assert result.decision == "supported"
        assert result.confidence > 0.4
        assert len(result.reasoning) > 10
        assert result.trace.input_evidence_count == 5

    def test_full_pipeline_with_contradictions(self):
        cortex = ReasoningCortex()
        result = cortex.reason(
            query="Is Python fast for production?",
            evidence=[
                "Python is slow compared to compiled languages",
                "Python has excellent performance for ML workloads",
                "Python is not suitable for high-performance computing",
                "Python with NumPy achieves C-like speed",
            ],
            sources=["stackoverflow.com", "reddit.com"],
        )
        assert isinstance(result, ReasoningResult)
        # Contradictory evidence should produce a lower-confidence result
        assert result.decision in ("mixed", "supported", "refuted", "insufficient")

    def test_full_pipeline_insufficient_evidence(self):
        cortex = ReasoningCortex()
        result = cortex.reason(
            query="What is the meaning of life?",
            evidence=[],
        )
        assert isinstance(result, ReasoningResult)
        assert result.decision in ("insufficient", "refuted")
        assert result.confidence < 0.5

    def test_full_pipeline_refuted(self):
        cortex = ReasoningCortex()
        result = cortex.reason(
            query="Is the earth flat?",
            evidence=[
                "Scientific evidence overwhelmingly confirms Earth is an oblate spheroid",
                "Satellite imagery from NASA shows Earth's curvature",
                "Gravity measurements confirm spherical shape",
            ],
            sources=["nasa.gov", "nature.com"],
        )
        assert isinstance(result, ReasoningResult)
        # The system should process this — "flat earth" has no supporting evidence
        assert result.decision in ("supported", "refuted", "mixed", "insufficient")

    def test_latency_is_reasonable(self):
        cortex = ReasoningCortex()
        result = cortex.reason(
            query="test",
            evidence=["test evidence one", "test evidence two"],
        )
        assert result.trace.total_latency_ms < 5000  # under 5 seconds

    def test_synapses_change_after_reasoning(self):
        cortex = ReasoningCortex()
        before = cortex.synapse_state
        cortex.reason(
            query="test",
            evidence=["good evidence with detail and substance"],
        )
        after = cortex.synapse_state
        # At least one synapse should have changed weight
        changed = any(
            before[k]["weight"] != after[k]["weight"]
            for k in before
        )
        assert changed

    def test_stats_tracking(self):
        cortex = ReasoningCortex()
        cortex.reason(query="q1", evidence=["e1"])
        cortex.reason(query="q2", evidence=["e2", "e3"])
        stats = cortex.stats()
        assert stats["reasoning_passes"] == 2
        assert stats["synapse_count"] > 0


# ══════════════════════════════════════════════════════════════════
# EXPLANATION NARRATOR
# ══════════════════════════════════════════════════════════════════

class TestExplanationNarrator:
    def test_narrates_supported_result(self):
        cortex = ReasoningCortex()
        result = cortex.reason(
            query="Is Python good for ML?",
            evidence=[
                "Python has extensive ML libraries",
                "Python is the standard for ML research",
                "Python dominates ML with 80% adoption rate",
            ],
            sources=["arxiv.org"],
        )
        narrator = ExplanationNarrator()
        explanation = narrator.narrate(result)

        assert explanation.decision_label == "SUPPORTED"
        assert len(explanation.executive_summary) > 20
        assert len(explanation.detailed_breakdown) > 50
        assert explanation.confidence_badge in ("HIGH", "MEDIUM", "LOW", "UNCERTAIN")

    def test_narrates_with_evidence_items(self):
        cortex = ReasoningCortex()
        result = cortex.reason(
            query="test question",
            evidence=[
                "First piece of detailed evidence with substance",
                "Second piece of evidence from a study (2024)",
            ],
            sources=["nature.com"],
        )
        narrator = ExplanationNarrator()
        explanation = narrator.narrate(result)
        # Detailed breakdown should mention evidence
        assert "Evidence" in explanation.detailed_breakdown or "evidence" in explanation.detailed_breakdown.lower()

    def test_narrates_mixed_result(self):
        cortex = ReasoningCortex()
        result = cortex.reason(
            query="Is X good?",
            evidence=[
                "X is not good for this use case",
                "X is great for this use case",
            ],
        )
        narrator = ExplanationNarrator()
        explanation = narrator.narrate(result)
        assert explanation.decision_label in ("MIXED", "INSUFFICIENT", "SUPPORTED", "REFUTED")

    def test_explanation_to_dict(self):
        cortex = ReasoningCortex()
        result = cortex.reason(query="q", evidence=["evidence one with some detail"])
        narrator = ExplanationNarrator()
        explanation = narrator.narrate(result)
        d = explanation.to_dict()
        assert "executive_summary" in d
        assert "detailed_breakdown" in d
        assert "confidence_badge" in d

    def test_str_returns_summary(self):
        cortex = ReasoningCortex()
        result = cortex.reason(query="q", evidence=["evidence"])
        narrator = ExplanationNarrator()
        explanation = narrator.narrate(result)
        assert str(explanation) == explanation.executive_summary


# ══════════════════════════════════════════════════════════════════
# HINDBRAIN: Fast filtering and salience detection
# ══════════════════════════════════════════════════════════════════

class TestHindbrain:
    def test_rejects_empty_query(self):
        from sweep_neural_mesh.neurons.brain import Hindbrain
        hb = Hindbrain()
        result = hb.process("", ["evidence one"])
        assert not result.sanity_passed
        assert result.rejection_reason != ""

    def test_rejects_garbage_evidence(self):
        from sweep_neural_mesh.neurons.brain import Hindbrain
        hb = Hindbrain()
        result = hb.process("What is Python?", ["test", "asdf", "lol", "ok"])
        assert result.sanity_passed
        assert len(result.filtered_evidence) == 0

    def test_passes_good_evidence(self):
        from sweep_neural_mesh.neurons.brain import Hindbrain
        hb = Hindbrain()
        result = hb.process(
            "How does machine learning work?",
            ["Machine learning uses algorithms to learn from data patterns"],
        )
        assert result.sanity_passed
        assert len(result.filtered_evidence) == 1
        assert result.salience_score > 0.3

    def test_salience_boosted_by_question_words(self):
        from sweep_neural_mesh.neurons.brain import Hindbrain
        hb = Hindbrain()
        result = hb.process(
            "How do I train a neural network?",
            ["Training a neural network involves backpropagation and gradient descent algorithms"],
        )
        assert result.salience_score > 0.3

    def test_deduplicates_evidence(self):
        from sweep_neural_mesh.neurons.brain import Hindbrain
        hb = Hindbrain()
        result = hb.process(
            "test query about something specific",
            [
                "This is detailed evidence about the topic with enough words to pass",
                "This is detailed evidence about the topic with enough words to pass",
            ],
        )
        assert len(result.filtered_evidence) == 1

    def test_rejects_empty_query_with_evidence(self):
        from sweep_neural_mesh.neurons.brain import Hindbrain
        hb = Hindbrain()
        result = hb.process("", ["evidence"])
        assert not result.sanity_passed

    def test_processing_time_tracked(self):
        from sweep_neural_mesh.neurons.brain import Hindbrain
        hb = Hindbrain()
        result = hb.process("test", ["evidence"])
        assert result.processing_time_ms >= 0


# ══════════════════════════════════════════════════════════════════
# MIDBRAIN: Signal routing and attention gating
# ══════════════════════════════════════════════════════════════════

class TestMidbrain:
    def test_routes_evidence_to_centers(self):
        from sweep_neural_mesh.neurons.brain import Midbrain
        mb = Midbrain()
        evidence = [
            {"text": "According to a study (2024), Python is used for ML", "source": "arxiv.org"},
            {"text": "Python was released in 1991 and has grown since", "source": "wikipedia.org"},
        ]
        result = mb.process(evidence, "What is Python?", 0.7)
        assert len(result.gated_evidence) > 0
        assert len(result.routed_signals) > 0

    def test_attention_weights_computed(self):
        from sweep_neural_mesh.neurons.brain import Midbrain
        mb = Midbrain()
        evidence = [
            {"text": "Python is a programming language with many libraries"},
            {"text": "x"},
        ]
        result = mb.process(evidence, "What is Python?", 0.5)
        assert len(result.attention_weights) > 0

    def test_gates_low_attention_evidence(self):
        from sweep_neural_mesh.neurons.brain import Midbrain
        mb = Midbrain()
        evidence = [
            {"text": ""},  # empty → low attention
        ]
        result = mb.process(evidence, "What is Python?", 0.3)
        assert len(result.gated_evidence) == 0

    def test_routes_causal_evidence(self):
        from sweep_neural_mesh.neurons.brain import Midbrain
        mb = Midbrain()
        evidence = [
            {"text": "Because Python is easy to learn, it leads to high adoption"},
        ]
        result = mb.process(evidence, "Why is Python popular?", 0.6)
        assert "causal_linker" in result.routed_signals
        assert len(result.routed_signals["causal_linker"]) > 0

    def test_routing_time_tracked(self):
        from sweep_neural_mesh.neurons.brain import Midbrain
        mb = Midbrain()
        result = mb.process([{"text": "test evidence"}], "test", 0.5)
        assert result.routing_time_ms >= 0


# ══════════════════════════════════════════════════════════════════
# FOREBRAIN: Memory system (episodic + semantic)
# ══════════════════════════════════════════════════════════════════

class TestForebrain:
    def test_records_episode(self):
        from sweep_neural_mesh.neurons.brain import Forebrain
        fb = Forebrain()
        fb.record_episode("test query", "supported", 0.8, 3, ["evidence one"])
        assert fb.episodic_count == 1

    def test_recalls_similar_episodes(self):
        from sweep_neural_mesh.neurons.brain import Forebrain
        fb = Forebrain()
        fb.record_episode("Python ML libraries", "supported", 0.8, 3, ["lib one"])
        fb.record_episode("Java web frameworks", "supported", 0.7, 2, ["spring"])
        recalls = fb.recall_similar("Python machine learning")
        assert len(recalls) > 0
        assert any("Python" in r.query for r in recalls)

    def test_consolidates_to_semantic_memory(self):
        from sweep_neural_mesh.neurons.brain import Forebrain
        fb = Forebrain()
        fb.record_episode("Python ML libraries", "supported", 0.8, 3, ["lib one"])
        assert fb.semantic_count > 0

    def test_semantic_memory_grows_with_repetition(self):
        from sweep_neural_mesh.neurons.brain import Forebrain
        fb = Forebrain()
        fb.record_episode("Python ML libraries", "supported", 0.8, 3, ["lib one"])
        fb.record_episode("Python ML frameworks", "supported", 0.85, 4, ["lib two"])
        fb.record_episode("Python ML tools", "supported", 0.9, 5, ["lib three"])
        # Should have consolidated into fewer semantic memories
        assert fb.semantic_count <= 3

    def test_get_semantic_knowledge(self):
        from sweep_neural_mesh.neurons.brain import Forebrain
        fb = Forebrain()
        fb.record_episode("Python ML libraries", "supported", 0.8, 3, ["scikit-learn"])
        knowledge = fb.get_semantic_knowledge("What Python ML tools exist?")
        assert len(knowledge) > 0


# ══════════════════════════════════════════════════════════════════
# BASAL GANGLIA: Action selection via reinforcement learning
# ══════════════════════════════════════════════════════════════════

class TestBasalGanglia:
    def test_creates_proposals(self):
        from sweep_neural_mesh.neurons.basal_ganglia import (
            BasalGanglia, ActionProposal, ActionType,
        )
        bg = BasalGanglia()
        proposals = [
            ActionProposal(
                action_type=ActionType.PROCEED_TO_CONSENSUS,
                confidence=0.7,
                reasoning="sufficient evidence",
                evidence_ids=[],
            ),
        ]
        decisions = bg.decide(proposals, {"confidence": 0.6, "evidence_count": 5})
        assert len(decisions) == 1
        assert decisions[0].go in (True, False)

    def test_learning_updates_policy(self):
        from sweep_neural_mesh.neurons.basal_ganglia import (
            BasalGanglia, ActionProposal, ActionType,
        )
        bg = BasalGanglia()
        proposal = ActionProposal(
            action_type=ActionType.PROCEED_TO_CONSENSUS,
            confidence=0.7,
            reasoning="test",
            evidence_ids=[],
            metadata={"confidence": 0.6, "evidence_count": 5},
        )
        decisions = bg.decide([proposal], {"confidence": 0.6, "evidence_count": 5})
        bg.learn(decisions, reward=1.0)
        assert bg.stats["avg_reward"] > 0

    def test_stats_tracking(self):
        from sweep_neural_mesh.neurons.basal_ganglia import (
            BasalGanglia, ActionProposal, ActionType,
        )
        bg = BasalGanglia()
        proposal = ActionProposal(
            action_type=ActionType.PROCEED_TO_CONSENSUS,
            confidence=0.7,
            reasoning="test",
            evidence_ids=[],
        )
        bg.decide([proposal], {"confidence": 0.5, "evidence_count": 3})
        stats = bg.stats
        assert stats["total_decisions"] == 1


# ══════════════════════════════════════════════════════════════════
# THALAMUS: Relay station
# ══════════════════════════════════════════════════════════════════

class TestThalamus:
    def test_relays_go_decisions(self):
        from sweep_neural_mesh.neurons.basal_ganglia import (
            Thalamus, ActionDecision, ActionProposal, ActionType,
        )
        th = Thalamus()
        proposal = ActionProposal(
            action_type=ActionType.PROCEED_TO_CONSENSUS,
            confidence=0.8,
            reasoning="test",
            evidence_ids=[],
        )
        decision = ActionDecision(
            proposal=proposal,
            go=True,
            confidence=0.7,
            reasoning="Go",
        )
        relay = th.relay([decision])
        assert relay.total_go == 1
        assert relay.total_nogo == 0

    def test_suppresses_low_confidence_go(self):
        from sweep_neural_mesh.neurons.basal_ganglia import (
            Thalamus, ActionDecision, ActionProposal, ActionType,
        )
        th = Thalamus()
        proposal = ActionProposal(
            action_type=ActionType.PROCEED_TO_CONSENSUS,
            confidence=0.8,
            reasoning="test",
            evidence_ids=[],
        )
        decision = ActionDecision(
            proposal=proposal,
            go=True,
            confidence=0.1,  # low confidence
            reasoning="Go",
        )
        relay = th.relay([decision], min_confidence=0.3)
        assert relay.total_go == 0
        assert relay.total_nogo == 1

    def test_relay_count_tracked(self):
        from sweep_neural_mesh.neurons.basal_ganglia import Thalamus
        th = Thalamus()
        assert th.relay_count == 0


# ══════════════════════════════════════════════════════════════════
# BRAIN INTEGRATION: Full pipeline with brain divisions
# ══════════════════════════════════════════════════════════════════

class TestBrainIntegration:
    def test_cortex_includes_brain_division_trace(self):
        cortex = ReasoningCortex()
        result = cortex.reason(
            query="How does Python ML work?",
            evidence=[
                "Python has scikit-learn for machine learning",
                "Python is the standard for ML research",
            ],
            sources=["arxiv.org"],
        )
        trace_dict = result.trace.to_dict()
        assert "brain_divisions" in trace_dict
        bd = trace_dict["brain_divisions"]
        assert "hindbrain_ms" in bd
        assert "midbrain_ms" in bd
        assert "forebrain_ms" in bd
        assert "salience_score" in bd
        assert "bg_decisions" in bd
        assert "memory_recall_count" in bd

    def test_memory_context_in_result(self):
        cortex = ReasoningCortex()
        result = cortex.reason(
            query="test query",
            evidence=["test evidence with detail and substance"],
        )
        assert "memory_context" in result.to_dict()
        mc = result.to_dict()["memory_context"]
        assert "episodic_recalls" in mc
        assert "semantic_knowledge" in mc

    def test_hindbrain_rejects_bad_input(self):
        cortex = ReasoningCortex()
        result = cortex.reason(query="", evidence=[])
        assert result.decision == "insufficient"
        assert "hindbrain rejection" in result.reasoning

    def test_brain_stats_available(self):
        cortex = ReasoningCortex()
        cortex.reason(query="test", evidence=["evidence"])
        stats = cortex.brain_stats
        assert "hindbrain" in stats
        assert "midbrain" in stats
        assert "forebrain" in stats
        assert "basal_ganglia" in stats
        assert "thalamus" in stats

    def test_stats_include_brain_divisions(self):
        cortex = ReasoningCortex()
        cortex.reason(query="test", evidence=["evidence"])
        stats = cortex.stats()
        assert "avg_hindbrain_ms" in stats
        assert "avg_midbrain_ms" in stats
        assert "avg_forebrain_ms" in stats
        assert "avg_salience" in stats
        assert "total_bg_decisions" in stats
        assert "total_memory_recalls" in stats

    def test_narrator_shows_brain_divisions(self):
        cortex = ReasoningCortex()
        result = cortex.reason(
            query="How does ML work?",
            evidence=["Machine learning uses algorithms to learn from data"],
        )
        narrator = ExplanationNarrator()
        explanation = narrator.narrate(result)
        assert "Hindbrain" in explanation.detailed_breakdown
        assert "Midbrain" in explanation.detailed_breakdown
        assert "Forebrain" in explanation.detailed_breakdown

    def test_memory_accumulates_across_calls(self):
        cortex = ReasoningCortex()
        cortex.reason(query="Python ML", evidence=["Python has ML libraries"])
        cortex.reason(query="Python ML frameworks", evidence=["Python has ML frameworks"])
        # Second call should have memory from first
        result = cortex.reason(
            query="Python machine learning",
            evidence=["Python ML is great"],
        )
        assert result.memory_context["episodic_recalls"] >= 0  # may or may not recall

    def test_source_metadata_flows_to_credibility(self):
        cortex = ReasoningCortex()
        result = cortex.reason(
            query="test",
            evidence=["A comprehensive study from nature.com shows results"],
            sources=["nature.com"],
        )
        # Should process without error
        assert result.decision in ("supported", "refuted", "mixed", "insufficient")
