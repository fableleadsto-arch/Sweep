"""
Sweep Neuronal Reasoning System — the brain.

Models Sweep's reasoning after biological neural architecture
with three brain divisions (hindbrain/midbrain/forebrain),
cortex-basal ganglia-thalamus action selection loop,
hippocampal memory system, synaptic plasticity (LTP/LTD),
myelination, and multi-dimensional evidence grading.

Now includes all 9 biological mechanisms:

HINDBRAIN (Survival Layer):
    1. Predictive Coding — generate hypotheses before processing
    2. Reflexive Shortcuts — bypass for known patterns
    3. Energy Gating — monitor system load

MIDBRAIN (Relay Layer):
    4. Dopaminergic Reward — predict signal value before processing
    5. Salience Modulation — amplify important, suppress noise
    6. Inhibitory Gating — block irrelevant channels (TRN)

FOREBRAIN (Processing Layer):
    7. Global Workspace — shared broadcasting between centers
    8. Working Memory — active context buffer (4-7 items)
    9. Metacognition — self-monitoring of reasoning quality

Plus 7 Human Reasoning Capabilities:

    1. Analogical Reasoning — find structural similarities across domains
    2. Causal World Model — build and update causal graphs
    3. Counterfactual Reasoning — explore what-if scenarios
    4. Common Sense KB — default assumptions about the world
    5. Theory of Mind — understand other agents' beliefs and intentions
    6. Abductive Reasoning — inference to the best explanation
    7. Narrative Coherence — structure evidence into coherent stories

Architecture:
    Raw Input
        ↓
    Hindbrain (predict → reflex → energy → filter → salience)
        ↓
    Midbrain (value predict → salience modulate → inhibit → route)
        ↓
    Forebrain (workspace broadcast → working memory → process → metacognition)
        ↓
    Reasoning Capabilities (analogical, causal, counterfactual, common sense,
                           theory of mind, abductive, narrative)
        ↓
    Synaptic Plasticity (LTP/LTD, myelination, circuit reorganization)
        ↓
    Multi-Dimensional Grading (depth, breadth, novelty, reliability, coherence, actionability)
        ↓
    Output (decision + grade + explanation)
"""
from .signal import Signal, SignalType, Synapse, SynapseType
from .centers import (
    ProcessingCenter,
    EvidenceGatherer,
    CredibilityAssessor,
    TemporalSequencer,
    CausalLinker,
    ContradictionDetector,
    ExplanationBuilder,
)
from .integration import IntegrationHub, ConsensusEngine, ConsensusDecision
from .cortex import ReasoningCortex, ReasoningResult, ReasoningTrace
from .narrator import ExplanationNarrator, Explanation
from .brain import Hindbrain, Midbrain, Forebrain
from .basal_ganglia import (
    BasalGanglia,
    Thalamus,
    ActionProposal,
    ActionDecision,
    ThalamusRelay,
    ActionType,
)
from .plasticity import SynapticPlasticity, MasteryPhase, LearningMetrics
from .grading import EvidenceGrader, EvidenceGrade, DimensionGrade, score_to_grade
# ── New biological mechanisms ──
from .predictive import PredictiveCoder, ReflexiveSystem, EnergyGating, Prediction, EnergyState
from .reward import DopaminergicSystem, SalienceModulator, InhibitoryGate
from .workspace import GlobalWorkspace, WorkspaceEntry, BroadcastResult
from .working_memory import WorkingMemory, WorkingMemoryItem, MemorySlot
from .metacognition import MetacognitiveSystem, MetacognitiveAssessment, UncertaintySignal
# ── Human reasoning capabilities ──
from .analogical import AnalogicalReasoner, Analogy, StructuralMapping, Domain, DomainEntity
from .causal_model import CausalModel, CausalNode, CausalEdge, InterventionResult, CounterfactualResult as CausalCounterfactual
from .counterfactual import CounterfactualReasoner, CounterfactualScenario, SensitivityReport
from .common_sense import CommonSense, CommonSenseRule, CommonSenseCheck
from .theory_of_mind import TheoryOfMind, AgentState, IntentAssessment, SocialContext
from .abductive import AbductiveReasoner, Hypothesis, AbductiveResult
from .narrative import NarrativeEngine, NarrativeEntity, StoryArc, NarrativeAssessment
# ── Improvement modules ──
from .embeddings import EmbeddingEngine
from .amygdala import Amygdala, ValenceCategory
from .forgetting import ForgettingCurve, MemoryTrace
# ── Advanced Math/Logic Modules ──
from .information import InformationTheory, EntropyResult, MutualInfoResult, InformationGainResult
from .fuzzy_logic import (
    FuzzyReasoner, FuzzyEvidenceGrader, FuzzySet, FuzzyRule, FuzzyResult,
    triangular_mf, trapezoidal_mf, gaussian_mf, sigmoid_mf,
    fuzzy_and, fuzzy_or, fuzzy_not, fuzzy_implies,
)
from .graph_algorithms import (
    ReasoningGraph, GraphNode, GraphEdge,
    PageRankResult, ShortestPathResult, CentralityResult, CommunityResult,
)
# ── Real ML Engines ──
from .semantic_embeddings import SemanticEmbedder, EmbeddingResult, SimilarityResult, get_embedder
from .ner_engine import NEREngine, Entity, NERResult, get_ner_engine
from .sentiment_engine import SentimentEngine, SentimentResult, SentimentLabel, get_sentiment_engine
from .speech_recognition import SpeechRecognizer, TranscriptResult, TranscriptSegment, get_recognizer
from .text_summarizer import TextSummarizer, SummaryResult, get_summarizer

__all__ = [
    "Signal", "SignalType", "Synapse", "SynapseType",
    "ProcessingCenter",
    "EvidenceGatherer", "CredibilityAssessor", "TemporalSequencer",
    "CausalLinker", "ContradictionDetector", "ExplanationBuilder",
    "IntegrationHub", "ConsensusEngine", "ConsensusDecision",
    "ReasoningCortex", "ReasoningResult", "ReasoningTrace",
    "ExplanationNarrator", "Explanation",
    "Hindbrain", "Midbrain", "Forebrain",
    "BasalGanglia", "Thalamus", "ActionProposal", "ActionDecision",
    "ThalamusRelay", "ActionType",
    "SynapticPlasticity", "MasteryPhase", "LearningMetrics",
    "EvidenceGrader", "EvidenceGrade", "DimensionGrade", "score_to_grade",
    # New biological mechanisms
    "PredictiveCoder", "ReflexiveSystem", "EnergyGating", "Prediction", "EnergyState",
    "DopaminergicSystem", "SalienceModulator", "InhibitoryGate",
    "GlobalWorkspace", "WorkspaceEntry", "BroadcastResult",
    "WorkingMemory", "WorkingMemoryItem", "MemorySlot",
    "MetacognitiveSystem", "MetacognitiveAssessment", "UncertaintySignal",
    # Human reasoning capabilities
    "AnalogicalReasoner", "Analogy", "StructuralMapping", "Domain", "DomainEntity",
    "CausalModel", "CausalNode", "CausalEdge", "InterventionResult", "CausalCounterfactual",
    "CounterfactualReasoner", "CounterfactualScenario", "SensitivityReport",
    "CommonSense", "CommonSenseRule", "CommonSenseCheck",
    "TheoryOfMind", "AgentState", "IntentAssessment", "SocialContext",
    "AbductiveReasoner", "Hypothesis", "AbductiveResult",
    "NarrativeEngine", "NarrativeEntity", "StoryArc", "NarrativeAssessment",
    # Improvement modules
    "EmbeddingEngine",
    "Amygdala", "ValenceCategory",
    "ForgettingCurve", "MemoryTrace",
    # Advanced Math/Logic Modules
    "InformationTheory", "EntropyResult", "MutualInfoResult", "InformationGainResult",
    "FuzzyReasoner", "FuzzyEvidenceGrader", "FuzzySet", "FuzzyRule", "FuzzyResult",
    "triangular_mf", "trapezoidal_mf", "gaussian_mf", "sigmoid_mf",
    "fuzzy_and", "fuzzy_or", "fuzzy_not", "fuzzy_implies",
    "ReasoningGraph", "GraphNode", "GraphEdge",
    "PageRankResult", "ShortestPathResult", "CentralityResult", "CommunityResult",
    # Real ML Engines
    "SemanticEmbedder", "EmbeddingResult", "SimilarityResult", "get_embedder",
    "NEREngine", "Entity", "NERResult", "get_ner_engine",
    "SentimentEngine", "SentimentResult", "SentimentLabel", "get_sentiment_engine",
    "SpeechRecognizer", "TranscriptResult", "TranscriptSegment", "get_recognizer",
    "TextSummarizer", "SummaryResult", "get_summarizer",
]
