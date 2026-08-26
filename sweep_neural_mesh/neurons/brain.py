"""
Brain Divisions — the three-part biological brain architecture.

Modeled after the vertebrate brain's three core divisions:

    ┌─────────────────────────────────────────────────────┐
    │  FOREBRAIN (Cortex + Basal Ganglia + Hippocampus)   │
    │  Higher cognition, decision-making, memory          │
    │  + Global Workspace + Working Memory + Metacognition │
    │  ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ │
    │  │ Processing │ │Basal     │ │Hippocampus│ │Meta- │ │
    │  │ Centers    │ │Ganglia   │ │(Memory)   │ │cogn. │ │
    │  └───────────┘ └──────────┘ └──────────┘ └──────┘ │
    │  ┌──────────────────┐  ┌────────────────────────┐  │
    │  │ Global Workspace  │  │ Working Memory Buffer   │  │
    │  │ (shared broadcast)│  │ (4-7 active items)      │  │
    │  └──────────────────┘  └────────────────────────┘  │
    ├─────────────────────────────────────────────────────┤
    │  MIDBRAIN (Thalamus + Superior Colliculus + VTA)    │
    │  Sensory relay, signal routing, attention gating    │
    │  + Dopaminergic Reward + Salience Modulation        │
    │  + Inhibitory Gating (TRN)                          │
    │  ┌──────────────────────────────────────────────┐   │
    │  │Reward Pred→Salience Mod→Inhibitory→Router   │   │
    │  └──────────────────────────────────────────────┘   │
    ├─────────────────────────────────────────────────────┤
    │  HINDBRAIN (Brainstem + Cerebellum)                 │
    │  Fast reflexes, basic filters, sanity checks        │
    │  + Predictive Coding + Reflexive Shortcuts          │
    │  + Energy Gating                                    │
    │  ┌──────────────────────────────────────────────┐   │
    │  │Predict→Reflex→Energy→Filter→Salience Det.   │   │
    │  └──────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────┘

Signal flow follows biological processing:

    Raw Input
        ↓
    Hindbrain (predict, reflex check, energy gate, filter)
        ↓
    Midbrain (reward predict, salience modulate, inhibit, route)
        ↓
    Forebrain (workspace broadcast, working memory, processing, metacognition)
        ↓
    Output
"""
from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .signal import Signal, SignalType
from .predictive import PredictiveCoder, ReflexiveSystem, EnergyGating, Prediction, EnergyState
from .reward import DopaminergicSystem, SalienceModulator, InhibitoryGate

logger = logging.getLogger("sweep.neurons.brain")


# ──────────────────────────────────────────────────────────────────
# HINDBRAIN: Fast reflexes and basic filters
# Like the brainstem and reticular formation, this division
# performs rapid, low-level processing on raw input before
# it reaches higher brain areas.
# ──────────────────────────────────────────────────────────────────


@dataclass
class HindbrainResult:
    """Output of the hindbrain's fast processing."""
    salience_score: float           # 0.0–1.0: how relevant is this input
    sanity_passed: bool             # did it pass basic sanity checks
    filtered_evidence: list[dict]   # evidence items that survived filtering
    rejection_reason: str           # why it was rejected (empty if passed)
    processing_time_ms: float
    # Predictive coding output
    prediction: Prediction | None = None
    reflexive_match: bool = False   # did a reflexive shortcut fire?
    reflexive_response: str = ""    # shortcut response if fired
    # Energy gating output
    energy_state: str = "fresh"     # current energy state
    energy_reason: str = ""         # energy gating decision reason


class Hindbrain:
    """
    Fast reflex processing — the brainstem of reasoning.

    Like the reticular activating system that filters sensory input
    before it reaches the cortex, this division now includes:

    1. PREDICTIVE CODING: Generate hypotheses before full processing
    2. REFLEXIVE SHORTCUTS: Bypass processing for known patterns
    3. ENERGY GATING: Defer non-urgent processing under load
    4. Input filtering and sanity checks
    5. Salience detection

    Signal flow:
        Raw Input → Predict → Reflex Check → Energy Gate → Filter → Salience
    """

    # Patterns that indicate obviously bad input
    GARBAGE_PATTERNS: list[str] = [
        r"^(test|asdf|lol|haha|ok)$",
        r"^[\W_]+$",                # only symbols/underscores (NOT numbers)
        r".{500,}",                 # suspiciously long (likely dump)
    ]

    # Patterns that indicate high-relevance input
    RELEVANCE_BOOSTERS: list[str] = [
        r"how\s+(do|does|can|to)",
        r"what\s+(is|are|was|were)",
        r"why\s+(do|does|is|are)",
        r"when\s+(did|was|will|is)",
        r"where\s+(is|are|was|can)",
    ]

    def __init__(self) -> None:
        # ── Predictive Coding: generate hypotheses before processing ──
        self._predictive_coder = PredictiveCoder()
        # ── Reflexive Shortcuts: bypass for known patterns ──
        self._reflexive_system = ReflexiveSystem()
        # ── Energy Gating: monitor system load ──
        self._energy_gating = EnergyGating()
        # Processing history
        self._process_history: list[dict[str, Any]] = []

    def process(
        self,
        query: str,
        evidence: list[str | dict[str, Any]],
        sources: list[str] | None = None,
    ) -> HindbrainResult:
        """
        Fast-filter raw input before it reaches higher processing.

        Now includes predictive coding, reflexive shortcuts, and energy gating.
        Returns filtered evidence, salience score, and prediction context.
        """
        t0 = time.perf_counter()

        # ── Step 0: Energy Gate — check if we should process at all ──
        energy_state, energy_reason = self._energy_gating.check_energy(query_priority=0.5)
        if energy_state == EnergyState.EXHAUSTED:
            return self._reject("system exhausted, query deferred", t0, energy_state.value, energy_reason)

        # ── Step 1: Sanity check the query ──
        if not query or not isinstance(query, str) or len(query.strip()) < 2:
            return self._reject("empty or invalid query", t0, energy_state.value, energy_reason)

        query_clean = query.strip()

        # ── Step 2: Predictive Coding — generate hypothesis BEFORE filtering ──
        evidence_dicts = []
        for item in evidence:
            if isinstance(item, str):
                evidence_dicts.append({"text": item})
            elif isinstance(item, dict):
                evidence_dicts.append(item)

        prediction = self._predictive_coder.predict(query_clean, evidence_dicts)

        # ── Step 3: Reflexive Check — can we shortcut? ──
        reflexive_match = self._reflexive_system.check_reflex(query_clean, prediction)
        if reflexive_match and reflexive_match.confidence > 0.7:
            elapsed = (time.perf_counter() - t0) * 1000
            self._energy_gating.record_processing_time(elapsed)
            self._process_history.append({
                "query": query_clean[:100],
                "reflexive": True,
                "salience": 0.0,
            })
            logger.debug(f"Hindbrain: reflexive shortcut fired for '{query_clean[:50]}...'")
            return HindbrainResult(
                salience_score=0.0,
                sanity_passed=True,
                filtered_evidence=[],  # no filtering needed for reflexive
                rejection_reason="",
                processing_time_ms=elapsed,
                prediction=prediction,
                reflexive_match=True,
                reflexive_response=reflexive_match.response,
                energy_state=energy_state.value,
                energy_reason=energy_reason,
            )

        # ── Step 4: Compute query salience ──
        salience = self._compute_salience(query_clean)

        # ── Step 5: Filter evidence ──
        filtered: list[dict] = []
        seen_texts: set[str] = set()

        for item in evidence:
            # Normalize to dict
            if isinstance(item, str):
                item_dict = {"text": item}
            elif isinstance(item, dict):
                item_dict = item
            else:
                continue

            text = item_dict.get("text", "")
            if not text or not isinstance(text, str):
                continue

            # Skip garbage
            if self._is_garbage(text):
                continue

            # Skip duplicates (by first 100 chars)
            text_key = text[:100].lower().strip()
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            # Compute per-item salience
            item_salience = self._compute_item_salience(text, query_clean)
            item_dict["_hindbrain_salience"] = item_salience
            item_dict["_hindbrain_source"] = (
                item_dict.get("source", "unknown")
                if isinstance(item_dict.get("source"), str)
                else "unknown"
            )

            # Only pass evidence with minimum salience
            if item_salience > 0.15:
                filtered.append(item_dict)

        # ── Step 6: Compute overall salience ──
        if filtered:
            avg_item_salience = sum(
                i.get("_hindbrain_salience", 0) for i in filtered
            ) / len(filtered)
            overall_salience = (salience * 0.6 + avg_item_salience * 0.4)
        else:
            overall_salience = salience * 0.3  # no evidence = low salience

        # ── Step 7: Gate check — reject if salience is too low ──
        if overall_salience < 0.1 and len(filtered) == 0:
            return self._reject("input too irrelevant or empty", t0, energy_state.value, energy_reason)

        elapsed = (time.perf_counter() - t0) * 1000
        self._energy_gating.record_processing_time(elapsed)

        self._process_history.append({
            "query": query_clean[:100],
            "reflexive": False,
            "salience": overall_salience,
            "energy_state": energy_state.value,
        })

        logger.debug(f"Hindbrain: processed '{query_clean[:50]}...' → salience={overall_salience:.3f}, "
                     f"{len(filtered)} evidence items, energy={energy_state.value}")
        return HindbrainResult(
            salience_score=max(0.0, min(1.0, overall_salience)),
            sanity_passed=True,
            filtered_evidence=filtered,
            rejection_reason="",
            processing_time_ms=elapsed,
            prediction=prediction,
            reflexive_match=False,
            reflexive_response="",
            energy_state=energy_state.value,
            energy_reason=energy_reason,
        )

    def _compute_salience(self, query: str) -> float:
        """How relevant is this query to potential evidence?"""
        score = 0.5  # base

        # Question words boost relevance
        for pattern in self.RELEVANCE_BOOSTERS:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.15
                break

        # Query length: too short is vague, too long is noisy
        word_count = len(query.split())
        if word_count < 3:
            score -= 0.15
        elif word_count > 30:
            score -= 0.10
        elif 5 <= word_count <= 15:
            score += 0.10

        # Contains domain-specific terms
        domain_terms = [
            "python", "javascript", "machine", "learning", "data",
            "api", "security", "performance", "algorithm", "database",
            "network", "system", "software", "hardware", "code",
        ]
        query_lower = query.lower()
        domain_hits = sum(1 for t in domain_terms if t in query_lower)
        score += min(0.2, domain_hits * 0.05)

        return max(0.0, min(1.0, score))

    def _compute_item_salience(self, text: str, query: str) -> float:
        """How relevant is this evidence item to the query?"""
        text_lower = text.lower()
        query_lower = query.lower()

        # Extract query keywords (skip common words)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "do", "does",
            "did", "will", "can", "could", "should", "would", "may",
            "how", "what", "why", "when", "where", "which", "who",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "this", "that", "these", "those", "it", "its",
        }
        query_words = {
            w.lower() for w in re.findall(r'\b\w+\b', query_lower)
            if w.lower() not in stop_words and len(w) > 2
        }

        if not query_words:
            return 0.4  # can't compute, return neutral

        # How many query words appear in the evidence?
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        overlap = query_words & text_words
        coverage = len(overlap) / len(query_words)

        # Base score from coverage
        score = 0.3 + coverage * 0.5

        # Length bonus: substantial evidence scores higher
        word_count = len(text.split())
        if word_count > 30:
            score += 0.10
        elif word_count < 5:
            # Short evidence: reduce penalty if it contains a number or named entity
            has_number = bool(re.search(r'\b\d+\b', text))
            has_named = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
            if has_number or has_named:
                score -= 0.05  # lighter penalty for factual short evidence
            else:
                score -= 0.15

        # Question-answer alignment: if evidence answers "how", it should explain
        if re.search(r'\bhow\s+many\b', query_lower) and re.search(r'\b\d+\b', text_lower):
            score += 0.20  # number in evidence for "how many" question is highly relevant

        if re.search(r'\bwho\b', query_lower) and re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
            score += 0.20  # named person in evidence for "who" question is highly relevant

        if re.search(r'\bwhat\b', query_lower) and re.search(
            r'\b(step|process|method|way|approach|technique)\b', text_lower
        ):
            score += 0.15

        if re.search(r'\bwhy\b', query_lower) and re.search(
            r'\b(because|reason|cause|due|since|therefore)\b', text_lower
        ):
            score += 0.15

        return max(0.0, min(1.0, score))

    def _is_garbage(self, text: str) -> bool:
        """Quick garbage detection."""
        text = text.strip()
        if len(text) < 2:
            return True
        for pattern in self.GARBAGE_PATTERNS:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return False

    def _reject(self, reason: str, t0: float, energy_state: str = "fresh", energy_reason: str = "") -> HindbrainResult:
        """Return a rejection result."""
        elapsed = (time.perf_counter() - t0) * 1000
        return HindbrainResult(
            salience_score=0.0,
            sanity_passed=False,
            filtered_evidence=[],
            rejection_reason=reason,
            processing_time_ms=elapsed,
            prediction=None,
            reflexive_match=False,
            energy_state=energy_state,
            energy_reason=energy_reason,
        )


# ──────────────────────────────────────────────────────────────────
# MIDBRAIN: Sensory relay and attention gating
# Like the thalamus and superior colliculus, this division
# routes signals to the appropriate cortical areas and
# controls which signals get attention.
# ──────────────────────────────────────────────────────────────────


@dataclass
class MidbrainResult:
    """Output of the midbrain's routing and gating."""
    routed_signals: dict[str, list[dict]]  # center_name → evidence items
    attention_weights: dict[str, float]     # center_name → attention weight
    gated_evidence: list[dict]              # all evidence with attention scores
    routing_time_ms: float
    # Dopaminergic reward predictions
    value_predictions: list[dict] = field(default_factory=list)
    # Salience modulation results
    salience_modulations: list[dict] = field(default_factory=list)
    # Inhibitory gating decisions
    inhibition_decisions: list[dict] = field(default_factory=list)


class Midbrain:
    """
    Signal routing and attention gating — the thalamus of reasoning.

    Now enhanced with three new mechanisms:

    1. DOPAMINERGIC REWARD PREDICTION: Predict value before processing
    2. SALIENCE-BASED ATTENTION MODULATION: Amplify important, suppress noise
    3. INHIBITORY GATING (TRN): Block irrelevant channels

    Like the thalamus relaying sensory information to the correct
    cortical areas, this division:

    1. Predicts value of each signal (dopaminergic)
    2. Modulates attention based on salience (superior colliculus)
    3. Applies inhibitory gating to suppress irrelevant channels (TRN)
    4. Routes evidence to processing centers based on features
    5. Assigns attention weights to control compute allocation

    Signal flow:
        Filtered Evidence → Value Predict → Salience Modulate
        → Inhibitory Gate → Route to Centers
    """

    # Default routing rules: which evidence characteristics
    # should be sent to which processing centers
    ROUTING_RULES: dict[str, list[str]] = {
        "credibility_assessor": [
            "has_url", "has_source", "has_citation", "has_authority",
        ],
        "temporal_sequencer": [
            "has_date", "has_recency", "has_time_reference",
        ],
        "causal_linker": [
            "has_causal_language", "has_multiple_items", "has_entities",
        ],
        "contradiction_detector": [
            "has_multiple_items", "has_negation", "has_opposing_claims",
        ],
    }

    def __init__(self) -> None:
        self._attention_history: list[dict[str, Any]] = []
        # ── Dopaminergic: predict value before processing ──
        self._dopaminergic = DopaminergicSystem()
        # ── Salience: modulate attention dynamically ──
        self._salience_modulator = SalienceModulator()
        # ── Inhibitory Gating: suppress irrelevant channels ──
        self._inhibitory_gate = InhibitoryGate()

    def process(
        self,
        evidence: list[dict],
        query: str,
        hindbrain_salience: float,
        prediction: Any = None,
    ) -> MidbrainResult:
        """
        Route evidence to processing centers with reward prediction,
        salience modulation, and inhibitory gating.

        Enhanced signal flow:
            Evidence → Value Predict → Salience Modulate →
            Inhibitory Gate → Route to Centers
        """
        t0 = time.perf_counter()

        routed: dict[str, list[dict]] = {
            "credibility_assessor": [],
            "temporal_sequencer": [],
            "causal_linker": [],
            "contradiction_detector": [],
        }

        gated: list[dict] = []
        value_predictions: list[dict] = []
        salience_modulations: list[dict] = []
        inhibition_decisions: list[dict] = []

        for item in evidence:
            # Extract features for routing decisions
            features = self._extract_features(item, query)

            # ── Dopaminergic: predict value before processing ──
            value_pred = self._dopaminergic.predict_value(
                signal_features={**item, **{f: True for f in features}},
                signal_id=item.get("text", "")[:20],
            )
            value_predictions.append({
                "value": value_pred.predicted_value,
                "confidence": value_pred.prediction_confidence,
                "reasoning": value_pred.reasoning,
            })

            # ── Salience: modulate attention dynamically ──
            base_attention = self._compute_attention(item, features, hindbrain_salience)
            salience_result = self._salience_modulator.modulate(
                signal_id=item.get("text", "")[:20],
                base_attention=base_attention,
                signal_features={**item, "text": item.get("text", "")},
                context={"evidence_count": len(evidence)},
            )
            salience_modulations.append({
                "original": salience_result.original_attention,
                "modulated": salience_result.modulated_attention,
                "reason": salience_result.amplification_reason,
            })

            # Use modulated attention
            attention = salience_result.modulated_attention
            item["_midbrain_attention"] = attention

            # ── Inhibitory Gating: suppress irrelevant channels ──
            gated_attention, inhibition_reason = self._inhibitory_gate.apply_gate(
                signal_id=item.get("text", "")[:20],
                signal_features={**item, "text": item.get("text", "")},
                base_attention=attention,
                context={"evidence_count": len(evidence)},
            )
            inhibition_decisions.append({
                "channel": item.get("source", "unknown"),
                "inhibition_reason": inhibition_reason,
                "original": attention,
                "gated": gated_attention,
            })

            # Apply inhibitory gating
            attention = gated_attention
            item["_midbrain_attention"] = attention

            # Gate: only process items above attention threshold
            if attention < 0.15:
                continue

            # Route to appropriate centers based on features
            routed_to_any = False
            for center_name, required_features in self.ROUTING_RULES.items():
                if any(f in features for f in required_features):
                    routed[center_name].append(item)
                    routed_to_any = True

            # If no specific routing match, send to all centers (general evidence)
            if not routed_to_any:
                for center_name in routed:
                    routed[center_name].append(item)

            gated.append(item)

        # Compute per-center attention weights
        attention_weights: dict[str, float] = {}
        for center_name, items in routed.items():
            if items:
                avg_attention = sum(
                    i.get("_midbrain_attention", 0.5) for i in items
                ) / len(items)
                attention_weights[center_name] = avg_attention
            else:
                attention_weights[center_name] = 0.0

        elapsed = (time.perf_counter() - t0) * 1000

        result = MidbrainResult(
            routed_signals=routed,
            attention_weights=attention_weights,
            gated_evidence=gated,
            routing_time_ms=elapsed,
            value_predictions=value_predictions,
            salience_modulations=salience_modulations,
            inhibition_decisions=inhibition_decisions,
        )

        self._attention_history.append({
            "query": query[:100],
            "evidence_count": len(evidence),
            "routed_count": len(gated),
            "attention_weights": attention_weights,
            "avg_value_prediction": (
                sum(v["value"] for v in value_predictions) / len(value_predictions)
                if value_predictions else 0.5
            ),
        })

        logger.debug(f"Midbrain: routed {len(gated)}/{len(evidence)} evidence → "
                     f"centers={attention_weights}")
        return result

    def _extract_features(self, item: dict, query: str) -> list[str]:
        """Extract routing-relevant features from an evidence item."""
        features: list[str] = []
        text = item.get("text", "")
        source = item.get("source", "")
        text_lower = text.lower()

        # URL presence
        if re.search(r'https?://', text) or source:
            features.append("has_url")
            features.append("has_source")

        # Citation patterns
        if re.search(r'\[\d+\]|\(20\d{2}\)|doi:|arxiv:', text):
            features.append("has_citation")

        # Authority indicators
        if re.search(
            r'(university|institute|journal|publish|official|government|\.edu|\.gov)',
            text, re.IGNORECASE,
        ):
            features.append("has_authority")

        # Date/time references
        if re.search(r'\b(19|20)\d{2}\b', text):
            features.append("has_date")
        if re.search(
            r'(today|yesterday|last\s+(week|month|year)|recently|currently)',
            text, re.IGNORECASE,
        ):
            features.append("has_recency")
        if re.search(
            r'(before|after|during|while|since|until|timeline|history)',
            text, re.IGNORECASE,
        ):
            features.append("has_time_reference")

        # Causal language
        if re.search(
            r'(because|therefore|caused|leads?\s+to|result|consequence|due\s+to|since|thus)',
            text_lower,
        ):
            features.append("has_causal_language")

        # Multiple items (lists, comparisons)
        if re.search(r'(also|additionally|moreover|furthermore|however|on\s+the\s+other)', text_lower):
            features.append("has_multiple_items")

        # Entity density (proper nouns, technical terms)
        words = re.findall(r'\b\w{4,}\b', text_lower)
        if len(words) > 10:
            features.append("has_entities")

        # Negation
        if re.search(r'\b(not|no|never|neither|doesn.t|didn.t|won.t|isn.t)\b', text_lower):
            features.append("has_negation")

        # Opposing claims
        if re.search(
            r'(however|but|although|despite|contrary|opposed|disagree|critics?)',
            text_lower,
        ):
            features.append("has_opposing_claims")

        return features

    def _compute_attention(
        self,
        item: dict,
        features: list[str],
        hindbrain_salience: float,
    ) -> float:
        """
        Compute attention weight for an evidence item.

        Higher attention = more processing time allocated.
        Like the superior colliculus directing eye movements,
        this determines which evidence gets deep processing.
        """
        text = item.get("text", "")
        word_count = len(text.split())

        # Empty or very short text gets minimal attention
        if word_count < 2:
            return 0.05

        score = 0.4  # base attention

        # Hindbrain salience feeds into attention
        score += hindbrain_salience * 0.2

        # Feature-based attention boosts
        if "has_source" in features:
            score += 0.15
        if "has_citation" in features:
            score += 0.10
        if "has_authority" in features:
            score += 0.10
        if "has_causal_language" in features:
            score += 0.08
        if "has_date" in features:
            score += 0.05
        if "has_negation" in features:
            score += 0.10  # contradictions need attention
        if "has_opposing_claims" in features:
            score += 0.12

        # Length consideration: very short evidence gets less attention
        if word_count < 5:
            score -= 0.20
        elif word_count > 50:
            score += 0.05

        return max(0.0, min(1.0, score))


# ──────────────────────────────────────────────────────────────────
# FOREBRAIN: Higher cognition (placeholder — real processing
# centers are in centers.py, integration.py, cortex.py)
# The forebrain division coordinates the existing processing
# centers and adds the hippocampal memory system.
# ──────────────────────────────────────────────────────────────────


@dataclass
class EpisodicMemory:
    """A single episodic memory: a past reasoning result."""
    query: str
    decision: str
    confidence: float
    evidence_count: int
    timestamp: float
    key_evidence: list[str]      # top evidence texts
    outcome: str = ""            # "correct", "incorrect", "unknown"


@dataclass
class SemanticMemory:
    """A semantic memory: a learned pattern about a topic."""
    topic: str                   # e.g., "python ml libraries"
    pattern: str                 # "python supports ml via libraries like X, Y, Z"
    confidence: float            # how sure we are about this pattern
    source_count: int            # how many episodes contributed
    last_updated: float


class Forebrain:
    """
    Higher cognition coordinator — the cerebral cortex of reasoning.

    Now enhanced with three new mechanisms:

    1. GLOBAL WORKSPACE: Shared broadcasting between all centers
    2. WORKING MEMORY: Active context buffer (4-7 items)
    3. METACOGNITION: Self-monitoring of reasoning quality

    The forebrain manages:
    1. The processing centers (evidence, credibility, temporal, etc.)
    2. The hippocampal memory system (episodic + semantic memory)
    3. Memory consolidation: turning episodic memories into semantic knowledge
    4. Global workspace: shared blackboard for inter-center communication
    5. Working memory: active context for ongoing reasoning
    6. Metacognition: monitoring and regulating our own reasoning

    Like the human forebrain, it uses past experience (episodic memory)
    to build general knowledge (semantic memory) that improves future
    reasoning without re-processing raw evidence.
    """

    def __init__(self) -> None:
        self._episodic_memory: list[EpisodicMemory] = []
        self._semantic_memory: list[SemanticMemory] = []
        self._max_episodic = 1000
        self._max_semantic = 500
        # Import here to avoid circular imports
        from .workspace import GlobalWorkspace
        from .working_memory import WorkingMemory, MemorySlot
        from .metacognition import MetacognitiveSystem
        # ── Global Workspace: shared broadcasting between centers ──
        self._workspace = GlobalWorkspace(capacity=12)
        # ── Working Memory: active context buffer ──
        self._working_memory = WorkingMemory(capacity=7)
        self._MemorySlot = MemorySlot
        # ── Metacognition: self-monitoring ──
        self._metacognition = MetacognitiveSystem()
        # ── Human Reasoning Capabilities ──
        from .analogical import AnalogicalReasoner
        from .causal_model import CausalModel
        from .counterfactual import CounterfactualReasoner
        from .common_sense import CommonSense
        from .theory_of_mind import TheoryOfMind
        from .abductive import AbductiveReasoner
        from .narrative import NarrativeEngine
        self._analogical = AnalogicalReasoner()
        self._causal_model = CausalModel()
        self._counterfactual = CounterfactualReasoner()
        self._common_sense = CommonSense()
        self._theory_of_mind = TheoryOfMind()
        self._abductive = AbductiveReasoner()
        self._narrative = NarrativeEngine()
        # ── Advanced Math/Logic Modules ──
        from .information import InformationTheory
        from .fuzzy_logic import FuzzyReasoner, FuzzyEvidenceGrader
        from .graph_algorithms import ReasoningGraph
        self._information_theory = InformationTheory()
        self._fuzzy_reasoner = FuzzyReasoner()
        self._fuzzy_grader = FuzzyEvidenceGrader()
        self._reasoning_graph = ReasoningGraph()
        logger.info("Forebrain initialized with all reasoning capabilities and math modules")

    @property
    def workspace(self):
        """Access the global workspace."""
        return self._workspace

    @property
    def working_memory(self):
        """Access working memory."""
        return self._working_memory

    @property
    def metacognition(self):
        """Access the metacognitive system."""
        return self._metacognition

    @property
    def analogical(self):
        """Access the analogical reasoner."""
        return self._analogical

    @property
    def causal_model(self):
        """Access the causal world model."""
        return self._causal_model

    @property
    def counterfactual(self):
        """Access the counterfactual reasoner."""
        return self._counterfactual

    @property
    def common_sense(self):
        """Access the common sense knowledge base."""
        return self._common_sense

    @property
    def theory_of_mind(self):
        """Access the Theory of Mind module."""
        return self._theory_of_mind

    @property
    def abductive(self):
        """Access the abductive reasoner."""
        return self._abductive

    @property
    def narrative(self):
        """Access the narrative coherence engine."""
        return self._narrative

    @property
    def information_theory(self):
        """Access the information theory engine."""
        return self._information_theory

    @property
    def fuzzy_reasoner(self):
        """Access the fuzzy logic reasoner."""
        return self._fuzzy_reasoner

    @property
    def fuzzy_grader(self):
        """Access the fuzzy evidence grader."""
        return self._fuzzy_grader

    @property
    def reasoning_graph(self):
        """Access the reasoning graph."""
        return self._reasoning_graph

    def record_episode(
        self,
        query: str,
        decision: str,
        confidence: float,
        evidence_count: int,
        key_evidence: list[str],
    ) -> None:
        """Record a reasoning result as an episodic memory."""
        episode = EpisodicMemory(
            query=query,
            decision=decision,
            confidence=confidence,
            evidence_count=evidence_count,
            timestamp=time.time(),
            key_evidence=key_evidence[:5],
        )
        self._episodic_memory.append(episode)

        # Trim if over capacity
        if len(self._episodic_memory) > self._max_episodic:
            self._episodic_memory = self._episodic_memory[-self._max_episodic:]

        # Try to consolidate into semantic memory
        self._consolidate(episode)
        logger.debug(f"Forebrain: recorded episode '{query[:50]}...' → "
                     f"episodic={len(self._episodic_memory)}, semantic={len(self._semantic_memory)}")

    def recall_similar(self, query: str, top_k: int = 3) -> list[EpisodicMemory]:
        """Recall episodic memories similar to the current query using embedding similarity."""
        if not self._episodic_memory:
            return []

        # Use SimHash embedding for semantic similarity
        try:
            from .embeddings import EmbeddingEngine
            engine = EmbeddingEngine()
            query_fp = engine.fingerprint(query)
        except ImportError:
            # Fallback to word-overlap if embeddings unavailable
            query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
            if not query_words:
                return []
            scored = []
            for ep in self._episodic_memory:
                ep_words = set(re.findall(r'\b\w{3,}\b', ep.query.lower()))
                overlap = len(query_words & ep_words)
                similarity = overlap / len(query_words) if query_words else 0.0
                age_hours = (time.time() - ep.timestamp) / 3600
                recency = max(0.0, 1.0 - age_hours / 168)
                score = similarity * 0.7 + recency * 0.3
                if score > 0.1:
                    scored.append((score, ep))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [ep for _, ep in scored[:top_k]]

        scored: list[tuple[float, EpisodicMemory]] = []
        for ep in self._episodic_memory:
            ep_fp = getattr(ep, '_embedding_fp', None)
            if ep_fp is None:
                ep_fp = engine.fingerprint(ep.query)
                ep._embedding_fp = ep_fp

            similarity = engine.similarity(query_fp, ep_fp)
            # Recency boost
            age_hours = (time.time() - ep.timestamp) / 3600
            recency = max(0.0, 1.0 - age_hours / 168)
            score = similarity * 0.7 + recency * 0.3
            if score > 0.15:
                scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def get_semantic_knowledge(self, query: str) -> list[SemanticMemory]:
        """Retrieve semantic knowledge relevant to a query using embedding similarity."""
        if not self._semantic_memory:
            return []

        try:
            from .embeddings import EmbeddingEngine
            engine = EmbeddingEngine()
            query_fp = engine.fingerprint(query)
        except ImportError:
            # Fallback to word-overlap if embeddings unavailable
            query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
            if not query_words:
                return []
            scored = []
            for sm in self._semantic_memory:
                topic_words = set(re.findall(r'\b\w{3,}\b', sm.topic.lower()))
                pattern_words = set(re.findall(r'\b\w{3,}\b', sm.pattern.lower()))
                all_words = topic_words | pattern_words
                overlap = len(query_words & all_words)
                if overlap > 0:
                    score = overlap / len(query_words) * sm.confidence
                    scored.append((score, sm))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [sm for _, sm in scored[:5]]

        scored: list[tuple[float, SemanticMemory]] = []
        for sm in self._semantic_memory:
            sm_fp = getattr(sm, '_embedding_fp', None)
            if sm_fp is None:
                sm_fp = engine.fingerprint(f"{sm.topic} {sm.pattern}")
                sm._embedding_fp = sm_fp

            similarity = engine.similarity(query_fp, sm_fp)
            score = similarity * sm.confidence
            if score > 0.1:
                scored.append((score, sm))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [sm for _, sm in scored[:5]]

    def _consolidate(self, episode: EpisodicMemory) -> None:
        """
        Consolidate episodic memory into semantic memory.

        Like hippocampal replay training the neocortex, this
        extracts patterns from individual episodes and builds
        general knowledge that persists across episodes.
        """
        # Extract topic keywords from the query
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "do", "does",
            "did", "will", "can", "could", "should", "would", "may",
            "how", "what", "why", "when", "where", "which", "who",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
        }
        words = [
            w.lower() for w in re.findall(r'\b\w+\b', episode.query)
            if w.lower() not in stop_words and len(w) > 2
        ]
        if len(words) < 2:
            return

        topic = " ".join(sorted(set(words))[:5])

        # Check if we already have a semantic memory for this topic
        existing = None
        for sm in self._semantic_memory:
            sm_words = set(re.findall(r'\b\w{3,}\b', sm.topic.lower()))
            query_words = set(words)
            if len(sm_words & query_words) / max(len(sm_words | query_words), 1) > 0.5:
                existing = sm
                break

        if existing:
            # Update existing semantic memory
            existing.source_count += 1
            # Confidence increases with more sources, up to a cap
            existing.confidence = min(
                0.95,
                existing.confidence + (1.0 - existing.confidence) * 0.1
            )
            existing.last_updated = time.time()
        else:
            # Create new semantic memory
            key_phrases = episode.key_evidence[:3]
            pattern = f"{episode.decision}: {'; '.join(key_phrases)}" if key_phrases else episode.decision
            self._semantic_memory.append(SemanticMemory(
                topic=topic,
                pattern=pattern[:200],
                confidence=episode.confidence * 0.8,
                source_count=1,
                last_updated=time.time(),
            ))

        # Trim if over capacity
        if len(self._semantic_memory) > self._max_semantic:
            # Remove least-updated
            self._semantic_memory.sort(key=lambda x: x.last_updated)
            self._semantic_memory = self._semantic_memory[-self._max_semantic:]

    @property
    def episodic_count(self) -> int:
        return len(self._episodic_memory)

    @property
    def semantic_count(self) -> int:
        return len(self._semantic_memory)
