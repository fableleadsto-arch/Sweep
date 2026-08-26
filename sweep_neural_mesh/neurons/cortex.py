"""
ReasoningCortex — the orchestrator that runs the full neuronal pipeline.

Now implements the complete three-division brain architecture with
all 9 biological mechanisms:

    ┌─────────────────────────────────────────────────────┐
    │  FOREBRAIN (Cortex + Basal Ganglia + Hippocampus)   │
    │  + Global Workspace + Working Memory + Metacognition │
    │  Higher cognition, decision-making, memory          │
    ├─────────────────────────────────────────────────────┤
    │  MIDBRAIN (Thalamus + Superior Colliculus + VTA)    │
    │  + Dopaminergic Reward + Salience Modulation        │
    │  + Inhibitory Gating                                │
    │  Sensory relay, signal routing, attention gating    │
    ├─────────────────────────────────────────────────────┤
    │  HINDBRAIN (Brainstem + Cerebellum)                 │
    │  + Predictive Coding + Reflexive Shortcuts          │
    │  + Energy Gating                                    │
    │  Fast reflexes, basic filters, sanity checks        │
    └─────────────────────────────────────────────────────┘

Signal flow:

    Raw Input
        ↓
    Hindbrain: Predict → Reflex Check → Energy Gate → Filter → Salience
        ↓
    Midbrain: Value Predict → Salience Modulate → Inhibit → Route
        ↓
    Forebrain: Workspace Broadcast → Working Memory → Process Centers
        ↓
    Cortex-BG-Thalamus Loop → Metacognition Check
        ↓
    Output (decision + grade + explanation)
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from .signal import Signal, SignalType, Synapse, SynapseType
from .centers import (
    EvidenceGatherer,
    CredibilityAssessor,
    TemporalSequencer,
    CausalLinker,
    ContradictionDetector,
    ExplanationBuilder,
    ProcessingCenter,
)
from .integration import IntegrationHub, ConsensusEngine, ConsensusDecision
from .brain import Hindbrain, Midbrain, Forebrain
from .basal_ganglia import (
    BasalGanglia,
    Thalamus,
    ActionProposal,
    ActionType,
)
from .plasticity import SynapticPlasticity, MasteryPhase
from .grading import EvidenceGrader, EvidenceGrade
from .semantic_embeddings import SemanticEmbedder, EmbeddingResult, SimilarityResult
from .ner_engine import NEREngine, Entity, NERResult
from .sentiment_engine import SentimentEngine, SentimentResult, SentimentLabel
from .text_summarizer import TextSummarizer, SummaryResult

logger = logging.getLogger(__name__)


@dataclass
class ReasoningTrace:
    """Complete trace of a single reasoning pass through the cortex."""
    query: str
    input_evidence_count: int
    center_outputs: dict[str, int]         # center_name → signal count
    integration_confidence: float
    decision: str
    decision_confidence: float
    reasoning: str
    total_latency_ms: float
    factors: list[dict[str, Any]] = field(default_factory=list)
    # Brain division timings
    hindbrain_ms: float = 0.0
    midbrain_ms: float = 0.0
    forebrain_ms: float = 0.0
    bg_decisions: int = 0
    salience_score: float = 0.0
    memory_recall_count: int = 0
    # Learning phase
    mastery_phase: str = "novice"
    # Multi-dimensional grade
    grade: dict[str, Any] = field(default_factory=dict)
    # ── New biological mechanisms ──
    # Hindbrain: Predictive coding
    prediction_accuracy: float = 0.0
    reflexive_shortcut: bool = False
    # Hindbrain: Energy gating
    energy_state: str = "fresh"
    # Midbrain: Dopaminergic reward
    avg_value_prediction: float = 0.0
    # Midbrain: Salience modulation
    avg_salience_modulation: float = 0.0
    # Midbrain: Inhibitory gating
    avg_inhibition: float = 0.0
    # Forebrain: Global Workspace
    workspace_ignitions: int = 0
    workspace_entries: int = 0
    # Forebrain: Working Memory
    working_memory_size: int = 0
    # Forebrain: Metacognition
    metacognition_awareness: float = 0.0
    uncertainty_signals: int = 0
    escalation_recommended: bool = False
    # ── Human Reasoning Capabilities ──
    analogical_mappings: int = 0
    causal_nodes: int = 0
    counterfactual_scenarios: int = 0
    common_sense_plausibility: float = 0.5
    theory_of_mind_trust: float = 0.5
    abductive_hypotheses: int = 0
    narrative_coherence: float = 0.0
    # ── Adaptive pipeline depth ──
    query_complexity: str = "moderate"
    active_modules: list[str] = field(default_factory=list)
    # ── Advanced Math Modules ──
    evidence_entropy_bits: float = 0.0
    evidence_pagerank: dict[str, float] = field(default_factory=dict)
    # ── ML Engine Outputs ──
    query_sentiment: str = "neutral"
    query_sentiment_valence: float = 0.0
    evidence_sentiments: list[str] = field(default_factory=list)
    extracted_entities: list[dict[str, str]] = field(default_factory=list)
    query_embedding_backend: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "input_evidence_count": self.input_evidence_count,
            "center_outputs": self.center_outputs,
            "integration_confidence": round(self.integration_confidence, 4),
            "decision": self.decision,
            "decision_confidence": round(self.decision_confidence, 4),
            "reasoning": self.reasoning,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "factors": self.factors,
            "brain_divisions": {
                "hindbrain_ms": round(self.hindbrain_ms, 2),
                "midbrain_ms": round(self.midbrain_ms, 2),
                "forebrain_ms": round(self.forebrain_ms, 2),
                "salience_score": round(self.salience_score, 4),
                "bg_decisions": self.bg_decisions,
                "memory_recall_count": self.memory_recall_count,
            },
            "biological_mechanisms": {
                "prediction_accuracy": round(self.prediction_accuracy, 4),
                "reflexive_shortcut": self.reflexive_shortcut,
                "energy_state": self.energy_state,
                "avg_value_prediction": round(self.avg_value_prediction, 4),
                "avg_salience_modulation": round(self.avg_salience_modulation, 4),
                "avg_inhibition": round(self.avg_inhibition, 4),
                "workspace_ignitions": self.workspace_ignitions,
                "workspace_entries": self.workspace_entries,
                "working_memory_size": self.working_memory_size,
                "metacognition_awareness": round(self.metacognition_awareness, 4),
                "uncertainty_signals": self.uncertainty_signals,
                "escalation_recommended": self.escalation_recommended,
            },
            "human_reasoning": {
                "analogical_mappings": self.analogical_mappings,
                "causal_nodes": self.causal_nodes,
                "counterfactual_scenarios": self.counterfactual_scenarios,
                "common_sense_plausibility": round(self.common_sense_plausibility, 4),
                "theory_of_mind_trust": round(self.theory_of_mind_trust, 4),
                "abductive_hypotheses": self.abductive_hypotheses,
                "narrative_coherence": round(self.narrative_coherence, 4),
            },
            "adaptive_pipeline": {
                "query_complexity": self.query_complexity,
                "active_modules": self.active_modules,
                "skipped_modules": [m for m in [
                    "common_sense", "abductive", "theory_of_mind",
                    "causal", "narrative", "analogical", "counterfactual"
                ] if m not in self.active_modules],
            },
            "math_modules": {
                "evidence_entropy_bits": round(self.evidence_entropy_bits, 4),
                "evidence_pagerank": {k: round(v, 4) for k, v in self.evidence_pagerank.items()},
            },
            "ml_engines": {
                "query_sentiment": self.query_sentiment,
                "query_sentiment_valence": round(self.query_sentiment_valence, 4),
                "evidence_sentiments": self.evidence_sentiments,
                "extracted_entities": self.extracted_entities,
                "query_embedding_backend": self.query_embedding_backend,
            },
            "mastery_phase": self.mastery_phase,
            "grade": self.grade,
        }


class ReasoningCortex:
    """
    The master orchestrator of Sweep's neuronal reasoning system.

    Implements the three-division brain architecture:
    1. Hindbrain: fast filtering and salience detection
    2. Midbrain: signal routing and attention gating
    3. Forebrain: processing centers + memory + action selection

    Also implements the cortex-basal ganglia-thalamus loop:
    the cortex proposes actions, the basal ganglia decides
    Go/NoGo via reinforcement learning, and the thalamus
    relays selected actions back for execution.

    Usage:
        cortex = ReasoningCortex()
        result = cortex.reason(
            query="Is Python a good language for ML?",
            evidence=["Python has extensive ML libraries", ...],
            sources=["wikipedia", "github"],
        )
        print(result.explanation)
    """

    def __init__(self, enable_ml: bool = False) -> None:
        self._enable_ml = enable_ml
        # ── Hindbrain: fast filtering ──
        self._hindbrain = Hindbrain()

        # ── Midbrain: routing and attention ──
        self._midbrain = Midbrain()

        # ── Forebrain: processing centers + memory ──
        self._forebrain = Forebrain()
        self._centers: dict[str, ProcessingCenter] = {
            "evidence_gatherer": EvidenceGatherer(),
            "credibility_assessor": CredibilityAssessor(),
            "temporal_sequencer": TemporalSequencer(),
            "causal_linker": CausalLinker(),
            "contradiction_detector": ContradictionDetector(),
            "explanation_builder": ExplanationBuilder(),
        }

        # ── Integration and consensus (forebrain) ──
        self._integration_hub = IntegrationHub()
        self._consensus_engine = ConsensusEngine()

        # ── Basal Ganglia: action selection via RL ──
        self._basal_ganglia = BasalGanglia()

        # ── Thalamus: relay station ──
        self._thalamus = Thalamus()

        # ── Synaptic Plasticity: learning mechanism ──
        self._plasticity = SynapticPlasticity()

        # ── Evidence Grader: multi-dimensional grading ──
        self._grader = EvidenceGrader()

        # ── Real ML Engines (lazy-loaded on first use) ──
        self._embedder = None
        self._ner_engine = None
        self._sentiment_engine = None
        self._summarizer = None
        self._ml_loaded = False

        # Synapses between centers (plastic connections)
        self._synapses: dict[str, Synapse] = {}
        self._build_default_synapses()

        # Reasoning history
        self._traces: list[ReasoningTrace] = []

        logger.info("ReasoningCortex initialized with 6 centers, BG-Thalamus loop, plasticity, grading")

    def _ensure_ml_engines(self):
        """Lazy-load ML engines only when first needed."""
        if self._ml_loaded:
            return
        self._ml_loaded = True
        try:
            self._embedder = SemanticEmbedder()
            self._ner_engine = NEREngine()
            self._sentiment_engine = SentimentEngine()
            self._summarizer = TextSummarizer()
            logger.info("ML engines loaded (semantic, NER, sentiment, summarizer)")
        except Exception as e:
            logger.warning(f"Failed to load ML engines: {e}")

    def _build_default_synapses(self) -> None:
        """Wire up the default synapse network."""
        connections = [
            ("evidence_gatherer", "credibility_assessor", 0.9, SynapseType.EXCITATORY),
            ("evidence_gatherer", "temporal_sequencer", 0.8, SynapseType.EXCITATORY),
            ("evidence_gatherer", "causal_linker", 0.85, SynapseType.EXCITATORY),
            ("evidence_gatherer", "contradiction_detector", 0.8, SynapseType.EXCITATORY),
            ("credibility_assessor", "causal_linker", 0.7, SynapseType.MODULATORY),
            ("contradiction_detector", "explanation_builder", 0.6, SynapseType.INHIBITORY),
        ]
        for from_c, to_c, weight, stype in connections:
            key = f"{from_c}->{to_c}"
            self._synapses[key] = Synapse(
                from_center=from_c,
                to_center=to_c,
                weight=weight,
                synapse_type=stype,
            )

    def reason(
        self,
        query: str,
        evidence: list[str | dict[str, Any]],
        sources: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> ReasoningResult:
        """
        Run the full neuronal reasoning pipeline with all 9 biological mechanisms.

        Signal flow:
            Raw Input
                ↓
            Hindbrain: Predict → Reflex → Energy → Filter → Salience
                ↓
            Midbrain: Value Predict → Salience Modulate → Inhibit → Route
                ↓
            Forebrain: Workspace Broadcast → Working Memory → Process
                ↓
            Cortex-BG-Thalamus Loop → Metacognition Check → Output
        """
        t0 = time.perf_counter()
        logger.info(f"Reasoning: '{query[:80]}...' ({len(evidence)} evidence items, "
                     f"{len(sources or [])} sources)")

        # ══════════════════════════════════════════════════════════
        # HINDBRAIN: Predict → Reflex → Energy → Filter → Salience
        # ══════════════════════════════════════════════════════════
        hindbrain_result = self._hindbrain.process(query, evidence, sources)
        hindbrain_ms = (time.perf_counter() - t0) * 1000

        # If hindbrain rejected input, short-circuit
        if not hindbrain_result.sanity_passed:
            return self._short_circuit(
                query, evidence, hindbrain_result.rejection_reason, t0
            )

        # If reflexive shortcut fired, return immediately
        if hindbrain_result.reflexive_match:
            return self._short_circuit(
                query, evidence,
                f"reflexive shortcut: {hindbrain_result.reflexive_response}",
                t0,
            )

        filtered_evidence = hindbrain_result.filtered_evidence
        salience = hindbrain_result.salience_score

        # ── ML Preprocessing: Sentiment, NER, Embeddings (only if enabled) ──
        query_sent_result = SentimentResult(text=query, label=SentimentLabel.NEUTRAL, score=0.5, valence=0.0, confidence=0.0, backend="none")
        evidence_sentiments: list[str] = []
        all_entities: list = []
        query_emb = EmbeddingResult(text=query, vector=None, dim=0, backend="none")

        if self._enable_ml:
            self._ensure_ml_engines()
            if self._sentiment_engine:
                query_sent_result = self._sentiment_engine.analyze(query)
                for ev in filtered_evidence[:5]:
                    ev_text = ev.get("text", "")
                    if ev_text:
                        ev_sent = self._sentiment_engine.analyze(ev_text)
                        evidence_sentiments.append(ev_sent.label.value)

            if self._ner_engine:
                query_ner = self._ner_engine.extract(query)
                all_entities.extend(query_ner.entities)
                for ev in filtered_evidence[:2]:
                    ev_text = ev.get("text", "")
                    if ev_text:
                        ev_ner = self._ner_engine.extract(ev_text)
                        all_entities.extend(ev_ner.entities)

            if self._embedder:
                query_emb = self._embedder.embed(query)

        # Check memory for similar past reasoning
        memory_recalls = self._forebrain.recall_similar(query)
        semantic_knowledge = self._forebrain.get_semantic_knowledge(query)

        # ══════════════════════════════════════════════════════════
        # MIDBRAIN: Value Predict → Salience Modulate → Inhibit → Route
        # ══════════════════════════════════════════════════════════
        midbrain_result = self._midbrain.process(
            filtered_evidence, query, salience,
            prediction=hindbrain_result.prediction,
        )
        midbrain_ms = (time.perf_counter() - t0) * 1000 - hindbrain_ms

        # ══════════════════════════════════════════════════════════
        # FOREBRAIN: Workspace → Working Memory → Processing Centers
        # ══════════════════════════════════════════════════════════
        forebrain_start = time.perf_counter()

        # ── Working Memory: set up current reasoning context ──
        self._forebrain.working_memory.insert(
            slot_type=self._forebrain._MemorySlot.QUERY,
            content={"query": query, "sources": sources or []},
            priority=0.9,
        )
        if semantic_knowledge:
            self._forebrain.working_memory.insert(
                slot_type=self._forebrain._MemorySlot.CONTEXT,
                content={"semantic_knowledge": [sm.topic for sm in semantic_knowledge[:3]]},
                priority=0.6,
            )
        if memory_recalls:
            self._forebrain.working_memory.insert(
                slot_type=self._forebrain._MemorySlot.CONTEXT,
                content={"episodic_recalls": [ep.query for ep in memory_recalls[:3]]},
                priority=0.5,
            )

        # ── Global Workspace: publish initial findings ──
        self._forebrain.workspace.publish(
            source_center="hindbrain",
            content={"salience": salience, "evidence_count": len(filtered_evidence)},
            salience=salience,
        )
        if midbrain_result.value_predictions:
            avg_value = sum(v["value"] for v in midbrain_result.value_predictions) / len(midbrain_result.value_predictions)
            self._forebrain.workspace.publish(
                source_center="midbrain",
                content={"avg_value_prediction": avg_value},
                salience=avg_value,
            )

        # Convert filtered evidence back to raw signal for evidence gatherer
        raw_data = {
            "query": query,
            "evidence": [
                {"text": e.get("text", ""), "source": e.get("source", e.get("_hindbrain_source", ""))}
                for e in filtered_evidence
            ],
            "sources": sources or [],
            "context": context or {},
            "_hindbrain_salience": salience,
            "_midbrain_attention": midbrain_result.attention_weights,
            "_semantic_knowledge": [
                {"topic": sm.topic, "pattern": sm.pattern}
                for sm in semantic_knowledge
            ],
            "_episodic_recalls": [
                {"query": ep.query, "decision": ep.decision, "confidence": ep.confidence}
                for ep in memory_recalls
            ],
        }
        input_signal = Signal(
            data=raw_data,
            signal_type=SignalType.RAW,
            confidence=1.0,
            source_center="sensory_input",
        )

        # ── Feed through processing centers ──
        center_outputs: dict[str, list[Signal]] = {}
        all_signals: list[Signal] = [input_signal]

        # Evidence gatherer first
        evidence_signals = self._centers["evidence_gatherer"].process([input_signal])
        center_outputs["evidence_gatherer"] = evidence_signals
        all_signals.extend(evidence_signals)

        # ── Global Workspace: publish evidence findings ──
        if evidence_signals:
            top_evidence = max(evidence_signals, key=lambda s: s.confidence)
            ws_result = self._forebrain.workspace.publish(
                source_center="evidence_gatherer",
                content={
                    "top_evidence": top_evidence.data.get("evidence_text", "")[:200],
                    "evidence_count": len(evidence_signals),
                },
                salience=top_evidence.confidence,
            )

        # Route through midbrain-attended centers (parallel execution)
        processing_centers = [
            "credibility_assessor",
            "temporal_sequencer",
            "causal_linker",
            "contradiction_detector",
        ]
        center_processing_times: dict[str, float] = {}

        def _process_center(name: str) -> tuple[str, list[Signal], float]:
            center = self._centers[name]
            start = time.perf_counter()
            modulated = self._apply_synaptic_input(name, all_signals)
            output = center.process(modulated)
            elapsed = (time.perf_counter() - start) * 1000
            return name, output, elapsed

        with ThreadPoolExecutor(max_workers=min(4, len(processing_centers))) as pool:
            futures = {pool.submit(_process_center, n): n for n in processing_centers}
            for future in as_completed(futures):
                name, output, elapsed = future.result()
                center_processing_times[name] = elapsed
                center_outputs[name] = output
                all_signals.extend(output)

        # ── Post-parallel: Working Memory, Workspace, Plasticity ──
        for name in processing_centers:
            output = center_outputs.get(name, [])
            if output:
                avg_conf = sum(s.confidence for s in output) / len(output)
                self._forebrain.working_memory.insert(
                    slot_type=self._forebrain._MemorySlot.FINDING,
                    content={
                        "center": name,
                        "signal_count": len(output),
                        "avg_confidence": avg_conf,
                    },
                    priority=avg_conf * 0.8,
                )
                self._forebrain.workspace.publish(
                    source_center=name,
                    content={
                        "signal_count": len(output),
                        "avg_confidence": avg_conf,
                    },
                    salience=avg_conf,
                )
                self._plasticity.record_activation(
                    from_center="evidence_gatherer",
                    to_center=name,
                    output_quality=avg_conf,
                    processing_time_ms=center_processing_times.get(name, 0.0),
                )
                self._hebbian_update(name, output)

        # ── Evidence Cross-Referencing ──
        # After all centers process, cross-reference evidence to boost
        # corroborated items and suppress isolated claims
        evidence_signals_for_xref = center_outputs.get("evidence_gatherer", [])
        credibility_for_xref = center_outputs.get("credibility_assessor", [])
        causal_for_xref = center_outputs.get("causal_linker", [])
        contradiction_for_xref = center_outputs.get("contradiction_detector", [])

        xref_boosted, xref_suppressed = self._cross_reference_evidence(
            evidence_signals_for_xref,
            credibility_for_xref,
            causal_for_xref,
            contradiction_for_xref,
        )

        # Apply cross-reference adjustments to evidence signals
        if xref_boosted or xref_suppressed:
            boosted_ids = {id(s) for s in xref_boosted}
            suppressed_ids = {id(s) for s in xref_suppressed}
            adjusted_evidence = []
            for es in evidence_signals_for_xref:
                if id(es) in boosted_ids:
                    # Boost corroborated evidence
                    boosted_sig = Signal(
                        data={**es.data, "_xref_boosted": True},
                        signal_type=es.signal_type,
                        confidence=min(1.0, es.confidence * 1.15),
                        source_center=es.source_center,
                        metadata={**es.metadata, "xref_action": "boosted"},
                        history=list(es.history),
                    )
                    adjusted_evidence.append(boosted_sig)
                elif id(es) in suppressed_ids:
                    # Suppress isolated claims
                    suppressed_sig = Signal(
                        data={**es.data, "_xref_suppressed": True},
                        signal_type=es.signal_type,
                        confidence=max(0.0, es.confidence * 0.80),
                        source_center=es.source_center,
                        metadata={**es.metadata, "xref_action": "suppressed"},
                        history=list(es.history),
                    )
                    adjusted_evidence.append(suppressed_sig)
                else:
                    adjusted_evidence.append(es)
            center_outputs["evidence_gatherer"] = adjusted_evidence
            # Update all_signals with adjusted evidence
            all_signals = [s for s in all_signals if s.signal_type != SignalType.EVIDENCE]
            all_signals.extend(adjusted_evidence)

        # ── Integration Hub: weighted merge ──
        processed_signals = [s for s in all_signals if s.signal_type != SignalType.RAW]
        integrated = self._integration_hub.integrate(processed_signals)

        # ══════════════════════════════════════════════════════════
        # CORTEX-BASAL GANGLIA-THALAMUS LOOP
        # ══════════════════════════════════════════════════════════
        context_for_bg = {
            "confidence": integrated.confidence,
            "evidence_count": len(filtered_evidence),
            "salience": salience,
        }
        proposals = self._cortex_propose(
            integrated, all_signals, context_for_bg, memory_recalls
        )
        bg_decisions = self._basal_ganglia.decide(proposals, context_for_bg)
        thalamus_relay = self._thalamus.relay(bg_decisions)

        # Execute selected actions
        integrated = self._execute_actions(
            thalamus_relay.selected_actions, integrated, all_signals
        )

        # ── Consensus Engine: final decision ──
        consensus = self._consensus_engine.decide(integrated, all_signals)

        # ══════════════════════════════════════════════════════════
        # FOREBRAIN MEMORY + METACOGNITION: Record and evaluate
        # ══════════════════════════════════════════════════════════
        consensus_data = consensus.data if isinstance(consensus.data, dict) else {}
        explanation_signals = self._centers["explanation_builder"].process(all_signals)

        # ── Metacognition: assess reasoning quality ──
        contradictions = len(center_outputs.get("contradiction_detector", []))
        meta_assessment = self._forebrain.metacognition.monitor_reasoning(
            confidence=consensus_data.get("confidence", 0.0),
            evidence_count=len(filtered_evidence),
            center_outputs={name: len(sigs) for name, sigs in center_outputs.items()},
            contradictions=contradictions,
            processing_phase=self._plasticity.get_metrics().overall_phase.value,
        )

        # Apply metacognitive confidence adjustment
        final_confidence = consensus_data.get("confidence", 0.0)
        if meta_assessment.should_adjust_confidence:
            final_confidence = max(0.0, min(1.0,
                final_confidence + meta_assessment.confidence_adjustment
            ))

        # ── Working Memory: record decision ──
        self._forebrain.working_memory.insert(
            slot_type=self._forebrain._MemorySlot.FINDING,
            content={
                "center": "consensus",
                "decision": consensus_data.get("decision", "unknown"),
                "confidence": final_confidence,
            },
            priority=0.9,
        )

        # ── Record episode to memory ──
        key_evidence = [
            s.data.get("evidence_text", "")
            for s in evidence_signals[:5]
            if s.data.get("evidence_text")
        ]
        self._forebrain.record_episode(
            query=query,
            decision=consensus_data.get("decision", "unknown"),
            confidence=final_confidence,
            evidence_count=len(filtered_evidence),
            key_evidence=key_evidence,
        )

        # ══════════════════════════════════════════════════════════
        # HUMAN REASONING CAPABILITIES (adaptive depth)
        # ══════════════════════════════════════════════════════════
        evidence_texts = [e.get("text", "") for e in filtered_evidence if e.get("text")]

        complexity = self._classify_query_complexity(query, len(filtered_evidence))
        active_modules = self._select_reasoning_modules(complexity, len(filtered_evidence))
        logger.debug(f"Query complexity: {complexity}, active modules: {active_modules}")

        # ── Information Theory: measure evidence entropy ──
        evidence_entropy_bits = 0.0
        if evidence_signals:
            # Create a distribution of evidence confidences
            confidence_buckets: dict[str, float] = {}
            for i, es in enumerate(evidence_signals):
                bucket = f"ev_{i}"
                confidence_buckets[bucket] = es.confidence
            entropy_result = self._forebrain.information_theory.shannon_entropy(confidence_buckets)
            evidence_entropy_bits = entropy_result.entropy

        # ── Graph Algorithms: build evidence graph and compute PageRank ──
        evidence_pagerank: dict[str, float] = {}
        if len(evidence_signals) >= 2:
            graph = self._forebrain.reasoning_graph
            for i, es in enumerate(evidence_signals):
                graph.add_node(f"ev_{i}", weight=es.confidence)
            for i in range(len(evidence_signals)):
                for j in range(i + 1, min(i + 4, len(evidence_signals))):
                    # Connect nearby evidence items
                    graph.add_edge(f"ev_{i}", f"ev_{j}", weight=0.5)
            try:
                pr_result = graph.pagerank(max_iter=20)
                evidence_pagerank = pr_result.rankings
            except Exception:
                pass

        common_sense_plausibility = 0.5
        theory_of_mind_trust = 0.5
        abductive_hypotheses = 0
        narrative_coherence = 0.0
        analogical_mappings = 0
        causal_nodes = 0
        counterfactual_scenarios = 0

        # Always run common_sense if selected
        if "common_sense" in active_modules:
            cs_check = self._forebrain.common_sense.check_claim(query)
            common_sense_plausibility = cs_check.plausibility_score

        # Theory of Mind
        if "theory_of_mind" in active_modules and sources:
            for src in sources[:3]:
                self._forebrain.theory_of_mind.register_agent(name=src)
            to_agent_ids = list(self._forebrain.theory_of_mind._agents.keys())
            toms_result = self._forebrain.theory_of_mind.infer_intent(
                agent_id=to_agent_ids[0] if to_agent_ids else "unknown",
                text=query,
            )
            theory_of_mind_trust = toms_result.intent_confidence if toms_result.should_trust else 0.3

        # Abductive Reasoning
        if "abductive" in active_modules:
            abd_result = self._forebrain.abductive.reason(
                observations=evidence_texts[:8] if evidence_texts else [query],
                context=context,
            )
            abductive_hypotheses = len(abd_result.hypotheses)

        # Narrative Coherence
        if "narrative" in active_modules:
            narr_result = self._forebrain.narrative.assess_narrative(
                evidence=evidence_texts[:10] if evidence_texts else [],
                query=query,
            )
            narrative_coherence = narr_result.overall_coherence

        # Analogical Reasoning
        if "analogical" in active_modules and re.search(
            r'\b(like|similar|compare|analogous|just as)\b', query.lower()
        ):
            from .analogical import Domain, DomainEntity as AnalogEntity
            src_domain = Domain(
                domain_id="query_ctx", name="query_context",
                entities=[AnalogEntity(name=query[:30], entity_type="concept", attributes={"query": query}, relations=[])],
            )
            tgt_domain = Domain(
                domain_id="evidence_ctx", name="evidence_context",
                entities=[AnalogEntity(name=e[:30], entity_type="concept", attributes={"text": e}, relations=[]) for e in evidence_texts[:3]],
            )
            self._forebrain.analogical.register_domain(src_domain)
            self._forebrain.analogical.register_domain(tgt_domain)
            analogy = self._forebrain.analogical.find_analogy("query_ctx", "evidence_ctx")
            analogical_mappings = len(analogy.mapping.entity_mappings) if analogy else 0

        # Causal Model
        if "causal" in active_modules and len(evidence_texts) >= 2:
            for i, text in enumerate(evidence_texts[:5]):
                self._forebrain.causal_model.add_node(
                    name=text[:50], node_type="evidence"
                )
            causal_nodes = len(self._forebrain.causal_model._nodes)
            for i in range(min(4, len(evidence_texts) - 1)):
                self._forebrain.causal_model.add_causal_link(
                    source_name=evidence_texts[i][:50],
                    target_name=evidence_texts[i + 1][:50],
                    strength=0.5,
                    edge_type="direct",
                )

        # Counterfactual
        if "counterfactual" in active_modules and len(evidence_texts) >= 2:
            self._forebrain.counterfactual.analyze_sensitivity(
                evidence=[{"text": e, "confidence": 0.5} for e in evidence_texts[:5]],
                current_confidence=final_confidence,
                current_decision=consensus_data.get("decision", "unknown"),
            )
            counterfactual_scenarios = len(self._forebrain.counterfactual._analysis_history)

        # ══════════════════════════════════════════════════════════

        # ── Compute timing ──
        forebrain_ms = (time.perf_counter() - forebrain_start) * 1000
        total_latency = (time.perf_counter() - t0) * 1000

        # ── Apply myelination speedup to future calls ──
        for center_name in processing_centers:
            adjusted = self._plasticity.apply_myelination_speedup(
                center_name, center_processing_times.get(center_name, 0.0)
            )

        # ── Get learning metrics ──
        learning_metrics = self._plasticity.get_metrics()

        # ── Multi-dimensional grading ──
        grade = self._grader.grade(
            evidence_signals=evidence_signals,
            credibility_signals=center_outputs.get("credibility_assessor", []),
            temporal_signals=center_outputs.get("temporal_sequencer", []),
            causal_signals=center_outputs.get("causal_linker", []),
            contradiction_signals=center_outputs.get("contradiction_detector", []),
            integrated_confidence=integrated.confidence,
            processing_phase=learning_metrics.overall_phase.value,
        )

        # ── Collect biological mechanism metrics ──
        # Predictive coding accuracy
        prediction_accuracy = 0.0
        if hindbrain_result.prediction:
            prediction_accuracy = hindbrain_result.prediction.confidence

        # Midbrain metrics
        avg_value = 0.0
        if midbrain_result.value_predictions:
            avg_value = sum(v["value"] for v in midbrain_result.value_predictions) / len(midbrain_result.value_predictions)
        avg_salience_mod = 0.0
        if midbrain_result.salience_modulations:
            avg_salience_mod = sum(
                m["modulated"] for m in midbrain_result.salience_modulations
            ) / len(midbrain_result.salience_modulations)
        avg_inhibition = 0.0
        if midbrain_result.inhibition_decisions:
            avg_inhibition = sum(
                d["gated"] for d in midbrain_result.inhibition_decisions
            ) / len(midbrain_result.inhibition_decisions)

        # Workspace metrics
        ws_stats = self._forebrain.workspace.stats

        # Working memory metrics
        wm_stats = self._forebrain.working_memory.stats

        # ── Build the trace ──
        center_counts = {name: len(sigs) for name, sigs in center_outputs.items()}
        explanation_data = (
            explanation_signals[0].data
            if explanation_signals and isinstance(explanation_signals[0].data, dict)
            else {}
        )

        trace = ReasoningTrace(
            query=query,
            input_evidence_count=len(evidence),
            center_outputs=center_counts,
            integration_confidence=integrated.confidence,
            decision=consensus_data.get("decision", "unknown"),
            decision_confidence=final_confidence,
            reasoning=consensus_data.get("reasoning", ""),
            total_latency_ms=total_latency,
            factors=consensus_data.get("factors", []),
            hindbrain_ms=hindbrain_ms,
            midbrain_ms=midbrain_ms,
            forebrain_ms=forebrain_ms,
            bg_decisions=len(bg_decisions),
            salience_score=salience,
            memory_recall_count=len(memory_recalls),
            mastery_phase=learning_metrics.overall_phase.value,
            grade=grade.to_dict(),
            # Biological mechanisms
            prediction_accuracy=prediction_accuracy,
            reflexive_shortcut=hindbrain_result.reflexive_match,
            energy_state=hindbrain_result.energy_state,
            avg_value_prediction=avg_value,
            avg_salience_modulation=avg_salience_mod,
            avg_inhibition=avg_inhibition,
            workspace_ignitions=ws_stats.get("ignition_count", 0),
            workspace_entries=ws_stats.get("active_entries", 0),
            working_memory_size=wm_stats.get("size", 0),
            metacognition_awareness=meta_assessment.awareness_score,
            uncertainty_signals=len(meta_assessment.uncertainty_signals),
            escalation_recommended=meta_assessment.escalation_recommended,
            # Human reasoning capabilities
            analogical_mappings=analogical_mappings,
            causal_nodes=causal_nodes,
            counterfactual_scenarios=counterfactual_scenarios,
            common_sense_plausibility=common_sense_plausibility,
            theory_of_mind_trust=theory_of_mind_trust,
            abductive_hypotheses=abductive_hypotheses,
            narrative_coherence=narrative_coherence,
            query_complexity=complexity,
            active_modules=active_modules,
            evidence_entropy_bits=evidence_entropy_bits,
            evidence_pagerank=evidence_pagerank,
            query_sentiment=query_sent_result.label.value,
            query_sentiment_valence=query_sent_result.valence,
            evidence_sentiments=evidence_sentiments,
            extracted_entities=[{"text": e.text, "label": e.label} for e in all_entities[:10]],
            query_embedding_backend=query_emb.backend,
        )
        self._traces.append(trace)

        logger.info(f"Reasoning complete: {consensus_data.get('decision', 'unknown')} "
                     f"(conf={final_confidence:.3f}, grade={grade.overall_grade}, "
                     f"latency={total_latency*1000:.1f}ms)")

        return ReasoningResult(
            query=query,
            decision=consensus_data.get("decision", "unknown"),
            confidence=final_confidence,
            reasoning=consensus_data.get("reasoning", ""),
            explanation_data=explanation_data,
            trace=trace,
            factors=consensus_data.get("factors", []),
            memory_context={
                "episodic_recalls": len(memory_recalls),
                "semantic_knowledge": len(semantic_knowledge),
            },
            grade=grade.to_dict(),
        )

    def _cortex_propose(
        self,
        integrated: Signal,
        all_signals: list[Signal],
        context: dict[str, Any],
        memory_recalls: list,
    ) -> list[ActionProposal]:
        """
        Cortex proposes actions based on current state.

        Like the cerebral cortex predicting which actions to take,
        this generates proposals for the basal ganglia to evaluate.
        """
        proposals: list[ActionProposal] = []
        confidence = integrated.confidence
        evidence_count = context.get("evidence_count", 0)

        # Proposal 1: If confidence is moderate, escalate credibility check
        if 0.3 < confidence < 0.7:
            proposals.append(ActionProposal(
                action_type=ActionType.ESCALATE_CREDIBILITY,
                confidence=0.6,
                reasoning="moderate confidence suggests credibility could help",
                evidence_ids=[],
                metadata=context,
            ))

        # Proposal 2: If many evidence items, check for contradictions
        if evidence_count > 5:
            proposals.append(ActionProposal(
                action_type=ActionType.ESCALATE_CONTRADICTION,
                confidence=0.5,
                reasoning=f"{evidence_count} items warrant contradiction check",
                evidence_ids=[],
                metadata=context,
            ))

        # Proposal 3: If we have memory recalls, use them
        if memory_recalls:
            avg_memory_conf = sum(r.confidence for r in memory_recalls) / len(memory_recalls)
            if avg_memory_conf > 0.6:
                proposals.append(ActionProposal(
                    action_type=ActionType.INCREASE_CONFIDENCE,
                    confidence=avg_memory_conf,
                    reasoning=f"{len(memory_recalls)} similar past episodes support this",
                    evidence_ids=[],
                    metadata=context,
                ))

        # Proposal 4: If confidence is high enough, proceed to consensus
        if confidence > 0.5:
            proposals.append(ActionProposal(
                action_type=ActionType.PROCEED_TO_CONSENSUS,
                confidence=confidence,
                reasoning="sufficient integration for decision",
                evidence_ids=[],
                metadata=context,
            ))

        # Proposal 5: If evidence is sparse, request more
        if evidence_count < 3:
            proposals.append(ActionProposal(
                action_type=ActionType.REQUEST_MORE_EVIDENCE,
                confidence=0.4,
                reasoning=f"only {evidence_count} evidence items available",
                evidence_ids=[],
                metadata=context,
            ))

        return proposals

    def _execute_actions(
        self,
        selected_actions: list,
        integrated: Signal,
        all_signals: list[Signal],
    ) -> Signal:
        """
        Execute actions selected by the basal ganglia via thalamus.

        Like the thalamus relaying Go signals to cortex for execution.
        """
        current_confidence = integrated.confidence
        for action in selected_actions:
            proposal = action.proposal

            if proposal.action_type == ActionType.INCREASE_CONFIDENCE:
                # Boost confidence based on memory support
                boost = proposal.confidence * 0.1
                current_confidence = min(1.0, current_confidence + boost)

            elif proposal.action_type == ActionType.DECREASE_CONFIDENCE:
                # Reduce confidence
                reduction = proposal.confidence * 0.1
                current_confidence = max(0.0, current_confidence - reduction)

        # Return modified integrated signal
        return Signal(
            data=integrated.data,
            signal_type=integrated.signal_type,
            confidence=current_confidence,
            source_center=integrated.source_center,
            metadata={**integrated.metadata, "bg_adjusted": True},
            history=list(integrated.history),
        )

    def _apply_synaptic_input(self, center_name: str, signals: list[Signal]) -> list[Signal]:
        """Apply synaptic modulation to signals before they reach a center."""
        modulated: list[Signal] = []
        for sig in signals:
            for key, synapse in self._synapses.items():
                if synapse.to_center == center_name:
                    sig = synapse.transmit(sig)
            modulated.append(sig)
        return modulated

    def _cross_reference_evidence(
        self,
        evidence_signals: list[Signal],
        credibility_signals: list[Signal],
        causal_signals: list[Signal],
        contradiction_signals: list[Signal],
    ) -> tuple[list[Signal], list[Signal]]:
        """
        Cross-reference evidence across processing centers to boost
        corroborated items and suppress isolated claims.

        Logic:
        - Evidence supported by multiple centers → boost
        - Evidence contradicted by high-confidence contradictions → suppress
        - Evidence with causal links to other evidence → slight boost
        - Evidence with no corroboration from any center → suppress

        Returns (boosted_signals, suppressed_signals).
        """
        if not evidence_signals:
            return [], []

        # Build corroboration map: for each evidence item, count how many
        # centers support it
        evidence_texts = [s.data.get("evidence_text", "") for s in evidence_signals]
        corroboration: dict[int, int] = {i: 0 for i in range(len(evidence_signals))}
        contradiction_map: dict[int, float] = {i: 0.0 for i in range(len(evidence_signals))}

        # Check credibility corroboration
        for cs in credibility_signals:
            cred_text = cs.data.get("evidence_text", "")
            for i, ev_text in enumerate(evidence_texts):
                if cred_text and ev_text and self._texts_overlap(cred_text, ev_text):
                    corroboration[i] += 1

        # Check causal corroboration
        for cl in causal_signals:
            ev_a = cl.data.get("evidence_a", "")
            ev_b = cl.data.get("evidence_b", "")
            link_strength = cl.confidence
            for i, ev_text in enumerate(evidence_texts):
                if ev_text and (self._texts_overlap(ev_text, ev_a) or self._texts_overlap(ev_text, ev_b)):
                    corroboration[i] += 1 if link_strength > 0.3 else 0

        # Check contradiction suppression
        for ct in contradiction_signals:
            ev_a = ct.data.get("evidence_a", "")
            ev_b = ct.data.get("evidence_b", "")
            contra_strength = ct.confidence
            for i, ev_text in enumerate(evidence_texts):
                if ev_text and self._texts_overlap(ev_text, ev_a):
                    contradiction_map[i] = max(contradiction_map[i], contra_strength)
                if ev_text and self._texts_overlap(ev_text, ev_b):
                    contradiction_map[i] = max(contradiction_map[i], contra_strength)

        boosted = []
        suppressed = []
        for i, sig in enumerate(evidence_signals):
            corroboration_count = corroboration[i]
            contra_strength = contradiction_map[i]

            # Boost if corroborated by 2+ centers
            if corroboration_count >= 2:
                boosted.append(sig)
            # Suppress if contradicted with high confidence
            elif contra_strength > 0.5:
                suppressed.append(sig)
            # Suppress if isolated (no corroboration at all and low base confidence)
            elif corroboration_count == 0 and sig.confidence < 0.5:
                suppressed.append(sig)

        return boosted, suppressed

    def _texts_overlap(self, text_a: str, text_b: str, threshold: float = 0.3) -> bool:
        """Check if two texts share enough content to be considered related."""
        if not text_a or not text_b:
            return False
        stop = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "was", "this", "that", "with"}
        words_a = set(re.findall(r'\b[a-z]{4,}\b', text_a.lower())) - stop
        words_b = set(re.findall(r'\b[a-z]{4,}\b', text_b.lower())) - stop
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b)
        union = len(words_a | words_b)
        jaccard = overlap / union if union > 0 else 0.0
        return jaccard >= threshold

    def _hebbian_update(self, center_name: str, output: list[Signal]) -> None:
        """Strengthen synapses that produced useful output (Hebbian learning)."""
        if not output:
            return
        avg_confidence = sum(s.confidence for s in output) / len(output)
        if avg_confidence > 0.4:
            for key, synapse in self._synapses.items():
                if synapse.to_center == center_name:
                    synapse.strengthen(0.02)
        elif avg_confidence < 0.2:
            for key, synapse in self._synapses.items():
                if synapse.to_center == center_name:
                    synapse.weaken(0.01)

    def _short_circuit(
        self,
        query: str,
        evidence: list,
        reason: str,
        t0: float,
    ) -> ReasoningResult:
        """Return early when hindbrain rejects input."""
        total_latency = (time.perf_counter() - t0) * 1000
        trace = ReasoningTrace(
            query=query,
            input_evidence_count=len(evidence),
            center_outputs={},
            integration_confidence=0.0,
            decision="insufficient",
            decision_confidence=0.0,
            reasoning=f"hindbrain rejection: {reason}",
            total_latency_ms=total_latency,
        )
        self._traces.append(trace)
        return ReasoningResult(
            query=query,
            decision="insufficient",
            confidence=0.0,
            reasoning=f"hindbrain rejection: {reason}",
            explanation_data={},
            trace=trace,
            factors=[],
            memory_context={"episodic_recalls": 0, "semantic_knowledge": 0},
        )

    @property
    def traces(self) -> list[ReasoningTrace]:
        return list(self._traces)

    @property
    def synapse_state(self) -> dict[str, dict[str, Any]]:
        return {key: syn.to_dict() for key, syn in self._synapses.items()}

    @property
    def brain_stats(self) -> dict[str, Any]:
        """Statistics about the brain divisions and learning."""
        learning = self._plasticity.get_metrics()
        return {
            "hindbrain": {"salience_history": len(self._hindbrain._process_history) if hasattr(self._hindbrain, '_process_history') else 0},
            "midbrain": {"attention_history": len(self._midbrain._attention_history)},
            "forebrain": {
                "episodic_memories": self._forebrain.episodic_count,
                "semantic_memories": self._forebrain.semantic_count,
            },
            "basal_ganglia": self._basal_ganglia.stats,
            "thalamus": {"relay_count": self._thalamus.relay_count},
            "plasticity": {
                "mastery_phase": learning.overall_phase.value,
                "total_activations": learning.total_activations,
                "total_ltp": learning.total_ltp_events,
                "total_ltd": learning.total_ltd_events,
                "avg_myelination": round(learning.avg_myelination, 4),
                "shortcut_count": learning.shortcut_count,
                "efficiency_gain": round(learning.efficiency_gain, 4),
            },
        }

    def stats(self) -> dict[str, Any]:
        if not self._traces:
            return {"reasoning_passes": 0}
        learning = self._plasticity.get_metrics()
        return {
            "reasoning_passes": len(self._traces),
            "avg_latency_ms": sum(t.total_latency_ms for t in self._traces) / len(self._traces),
            "decision_breakdown": {
                d: sum(1 for t in self._traces if t.decision == d)
                for d in set(t.decision for t in self._traces)
            },
            "synapse_count": len(self._synapses),
            "avg_hindbrain_ms": sum(t.hindbrain_ms for t in self._traces) / len(self._traces),
            "avg_midbrain_ms": sum(t.midbrain_ms for t in self._traces) / len(self._traces),
            "avg_forebrain_ms": sum(t.forebrain_ms for t in self._traces) / len(self._traces),
            "avg_salience": sum(t.salience_score for t in self._traces) / len(self._traces),
            "total_bg_decisions": sum(t.bg_decisions for t in self._traces),
            "total_memory_recalls": sum(t.memory_recall_count for t in self._traces),
            "mastery_phase": learning.overall_phase.value,
            "total_ltp_events": learning.total_ltp_events,
            "total_ltd_events": learning.total_ltd_events,
            "avg_myelination": round(learning.avg_myelination, 4),
            "efficiency_gain": round(learning.efficiency_gain, 4),
        }

    # ════════════════════════════════════════════════════════════════
    # ADAPTIVE PIPELINE DEPTH
    # ════════════════════════════════════════════════════════════════

    def _classify_query_complexity(self, query: str, evidence_count: int) -> str:
        """
        Classify query complexity to determine which reasoning modules to activate.

        Returns: 'trivial', 'simple', 'moderate', 'complex', or 'deep'
        """
        q = query.lower().strip()
        word_count = len(q.split())

        # Trivial: greetings, single-word queries
        if word_count <= 2 and evidence_count == 0:
            return "trivial"

        # Simple: factual lookups
        simple_patterns = [
            r'\b(what is|who is|when did|where is|how many)\b',
            r'\b(define|meaning of|abbreviation)\b',
        ]
        if word_count <= 8 and any(re.search(p, q) for p in simple_patterns) and evidence_count < 3:
            return "simple"

        # Moderate: analysis questions
        moderate_patterns = [
            r'\b(compare|difference|why|how does|explain)\b',
            r'\b(best|worse|should|recommend)\b',
        ]
        if word_count <= 15 and evidence_count < 8:
            if any(re.search(p, q) for p in moderate_patterns):
                return "moderate"

        # Deep: complex multi-step reasoning
        deep_patterns = [
            r'\b(counterfactual|what if|suppose|imagine)\b',
            r'\banalog(y|ous|ize)\b',
            r'\b(narrative|story|sequence of events)\b',
            r'\b(caus(al|e)|root cause|consequence)\b',
        ]
        has_deep = any(re.search(p, q) for p in deep_patterns)

        # Complex: evidence-heavy or multi-source
        if evidence_count >= 8 or (evidence_count >= 4 and word_count >= 15) or has_deep:
            return "deep"

        return "moderate"

    def _select_reasoning_modules(
        self, complexity: str, evidence_count: int
    ) -> list[str]:
        """
        Select which human reasoning modules to activate based on query complexity.

        This implements adaptive pipeline depth:
        - trivial: skip all reasoning
        - simple: common_sense only
        - moderate: common_sense + abductive
        - complex: add theory_of_mind + causal
        - deep: all 7 modules
        """
        if complexity == "trivial":
            return []

        modules = ["common_sense"]

        if complexity in ("moderate", "complex", "deep"):
            modules.append("abductive")

        if complexity in ("complex", "deep"):
            modules.append("theory_of_mind")
            modules.append("causal")

        if complexity == "deep":
            modules.append("narrative")
            modules.append("analogical")
            modules.append("counterfactual")

        return modules


@dataclass
class ReasoningResult:
    """The complete output of a reasoning pass."""
    query: str
    decision: str
    confidence: float
    reasoning: str
    explanation_data: dict[str, Any]
    trace: ReasoningTrace
    factors: list[dict[str, Any]]
    memory_context: dict[str, Any] = field(default_factory=dict)
    grade: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "factors": self.factors,
            "explanation": self.explanation_data,
            "trace": self.trace.to_dict(),
            "memory_context": self.memory_context,
            "grade": self.grade,
        }
