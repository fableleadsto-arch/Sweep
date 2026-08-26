"""
Dopaminergic Reward Prediction, Salience Modulation, and Inhibitory Gating — the Midbrain relay layer.

These three mechanisms control signal routing and attention:

1. DOPAMINERGIC REWARD PREDICTION (VTA + Substantia Nigra):
   Predict the VALUE of processing a signal before doing it.
   Like dopamine neurons that fire in anticipation of reward,
   this boosts high-value signals early and suppresses low-value ones.
   Uses past outcomes to predict future value of similar signals.

2. SALIENCE-BASED ATTENTION MODULATION (Superior Colliculus):
   Not just routing but ACTIVELY amplifying important signals
   and suppressing noise. Uses temporal context, novelty detection,
   and relevance scoring to dynamically adjust attention weights.

3. INHIBITORY GATING — Thalamic Reticular Nucleus:
   Block signals that are currently irrelevant. Like selective
   attention in hearing (cocktail party effect), this maintains
   a "relevance mask" that suppresses currently unimportant channels.

Signal flow in the midbrain:

    Filtered Evidence (from Hindbrain)
        ↓
    Reward Prediction: score expected value of each signal
        ↓
    Salience Modulation: amplify important, suppress noise
        ↓
    Inhibitory Gating: block currently irrelevant channels
        ↓
    Output to Forebrain (with value/salience context)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ──────────────────────────────────────────────────────────────────
# DOPAMINERGIC REWARD PREDICTION
# ──────────────────────────────────────────────────────────────────


@dataclass
class ValuePrediction:
    """A prediction about the value of processing a signal."""
    signal_id: str
    predicted_value: float         # 0.0-1.0: expected value of processing this
    prediction_confidence: float   # how confident in this value prediction
    value_components: dict[str, float]  # breakdown of value factors
    reasoning: str                 # why this value was predicted
    timestamp: float = field(default_factory=time.time)


class DopaminergicSystem:
    """
    Predict the value of processing signals before full analysis.

    Like the ventral tegmental area (VTA) and substantia nigra that
    produce dopamine and signal reward prediction, this module:

    1. Maintains a value memory: which types of signals produced good outcomes
    2. Predicts the value of new signals based on their features
    3. Updates predictions based on actual outcomes (reward prediction error)
    4. Signals "wanting" (incentive salience) vs "liking" (actual value)

    The key insight from dopamine neuroscience:
    - Dopamine doesn't signal pleasure — it signals PREDICTED pleasure
    - When prediction is wrong (better or worse than expected), dopamine spikes
    - This prediction error drives learning

    In Sweep:
    - High predicted value → boost signal early, allocate more compute
    - Low predicted value → dampen signal, allocate less compute
    - Prediction error → update value memory for future predictions
    """

    def __init__(self) -> None:
        # Value memory: feature_pattern → average_value
        self._value_memory: dict[str, dict[str, float]] = {}
        # Prediction history for learning
        self._prediction_history: list[dict[str, Any]] = []
        # Learning parameters
        self._learning_rate = 0.15
        self._decay_rate = 0.02
        # Statistics
        self._total_predictions = 0
        self._total_updates = 0

    def predict_value(
        self,
        signal_features: dict[str, Any],
        signal_id: str = "",
    ) -> ValuePrediction:
        """
        Predict the value of processing this signal.

        Called BEFORE expensive processing. Uses learned value memory
        and feature-based heuristics to estimate expected value.
        """
        self._total_predictions += 1

        # Extract feature pattern for memory lookup
        pattern = self._extract_pattern(signal_features)

        # Check value memory
        remembered_value = self._value_memory.get(pattern)
        if remembered_value:
            base_value = remembered_value.get("avg_value", 0.5)
            confidence = min(0.9, remembered_value.get("confidence", 0.5))
        else:
            base_value = 0.5
            confidence = 0.3

        # Feature-based value components
        components = self._compute_value_components(signal_features)

        # Combine remembered value with feature-based estimate
        if remembered_value:
            predicted_value = base_value * 0.6 + components["combined"] * 0.4
        else:
            predicted_value = components["combined"]

        predicted_value = max(0.0, min(1.0, predicted_value))

        reasoning = self._build_value_reasoning(
            predicted_value, components, bool(remembered_value)
        )

        return ValuePrediction(
            signal_id=signal_id,
            predicted_value=predicted_value,
            prediction_confidence=confidence,
            value_components=components,
            reasoning=reasoning,
        )

    def update_from_outcome(
        self,
        prediction: ValuePrediction,
        actual_value: float,
    ) -> float:
        """
        Update value memory based on actual outcome.

        This implements reward prediction error:
        - If actual > predicted: positive error → boost value memory
        - If actual < predicted: negative error → reduce value memory
        - Error magnitude drives learning speed

        Returns the prediction error (for downstream learning).
        """
        self._total_updates += 1

        prediction_error = actual_value - prediction.predicted_value

        # Update value memory
        pattern = self._extract_pattern(prediction.value_components)
        if pattern not in self._value_memory:
            self._value_memory[pattern] = {
                "avg_value": actual_value,
                "confidence": 0.5,
                "update_count": 1,
            }
        else:
            mem = self._value_memory[pattern]
            # Incremental update
            mem["avg_value"] = mem["avg_value"] * (1 - self._learning_rate) + actual_value * self._learning_rate
            mem["update_count"] += 1
            # Confidence increases with more updates
            mem["confidence"] = min(0.95, 0.5 + mem["update_count"] * 0.05)

        # Record for history
        self._prediction_history.append({
            "signal_id": prediction.signal_id,
            "predicted": prediction.predicted_value,
            "actual": actual_value,
            "error": prediction_error,
            "timestamp": time.time(),
        })

        # Keep history bounded
        if len(self._prediction_history) > 500:
            self._prediction_history = self._prediction_history[-500:]

        return prediction_error

    def _extract_pattern(self, features: dict[str, Any]) -> str:
        """Extract a pattern key from signal features."""
        parts = []
        for key in sorted(features.keys()):
            val = features[key]
            if isinstance(val, str):
                parts.append(f"{key}:{val[:20]}")
            elif isinstance(val, (int, float)):
                parts.append(f"{key}:{val:.2f}")
            elif isinstance(val, bool):
                parts.append(f"{key}:{val}")
        return "|".join(parts[:8])

    def _compute_value_components(self, features: dict[str, Any]) -> dict[str, float]:
        """Compute value components from signal features."""
        components = {}

        # Source credibility value
        source = str(features.get("source", "")).lower()
        trusted = ["wikipedia", "github", "arxiv", "nature", "pubmed", "edu", "gov"]
        if any(t in source for t in trusted):
            components["source_trust"] = 0.8
        elif source and source != "unknown":
            components["source_trust"] = 0.5
        else:
            components["source_trust"] = 0.3

        # Information density value
        text = str(features.get("text", ""))
        word_count = len(text.split())
        if word_count > 50:
            components["density"] = 0.8
        elif word_count > 20:
            components["density"] = 0.6
        elif word_count > 5:
            components["density"] = 0.4
        else:
            components["density"] = 0.2

        # Recency value
        if features.get("has_recency"):
            components["recency"] = 0.8
        elif features.get("has_date"):
            components["recency"] = 0.6
        else:
            components["recency"] = 0.4

        # Specificity value (has specific data)
        if re.search(r'\d+', text):
            components["specificity"] = 0.7
        else:
            components["specificity"] = 0.4

        # Combined value
        if components:
            components["combined"] = sum(components.values()) / len(components)
        else:
            components["combined"] = 0.5

        return components

    def _build_value_reasoning(
        self,
        predicted_value: float,
        components: dict[str, float],
        from_memory: bool,
    ) -> str:
        """Build human-readable reasoning for value prediction."""
        parts = []
        if from_memory:
            parts.append("based on learned value memory")
        if components.get("source_trust", 0) > 0.6:
            parts.append("high-trust source")
        if components.get("density", 0) > 0.6:
            parts.append("information-dense")
        if components.get("recency", 0) > 0.6:
            parts.append("recent/relevant")
        if predicted_value > 0.7:
            parts.append("high expected value")
        elif predicted_value < 0.3:
            parts.append("low expected value")
        return "; ".join(parts) if parts else "default value estimate"

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_predictions": self._total_predictions,
            "total_updates": self._total_updates,
            "value_memory_size": len(self._value_memory),
            "history_size": len(self._prediction_history),
        }


# ──────────────────────────────────────────────────────────────────
# SALIENCE-BASED ATTENTION MODULATION
# ──────────────────────────────────────────────────────────────────


@dataclass
class SalienceResult:
    """Result of salience modulation."""
    signal_id: str
    original_attention: float      # attention before modulation
    modulated_attention: float     # attention after modulation
    modulation_factor: float       # how much it changed
    amplification_reason: str      # why it was amplified/suppressed


class SalienceModulator:
    """
    Actively amplify important signals and suppress noise.

    Like the superior colliculus that directs eye movements and
    attention based on stimulus salience, this module:

    1. Computes intrinsic salience (how inherently important is this signal)
    2. Applies contextual modulation (what's happening in the current reasoning)
    3. Amplifies high-salience signals, suppresses low-salience ones
    4. Adapts modulation based on reasoning history

    The key insight: attention is not fixed — it dynamically shifts
    based on what's currently important. A signal that's noise in one
    context might be critical in another.
    """

    def __init__(self) -> None:
        # Attention history for adaptive modulation
        self._attention_history: list[dict[str, Any]] = []
        # Channel activity: which "channels" are currently active
        self._channel_activity: dict[str, float] = {}
        # Learning: which modulation decisions worked well
        self._modulation_rewards: list[float] = []

    def modulate(
        self,
        signal_id: str,
        base_attention: float,
        signal_features: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> SalienceResult:
        """
        Modulate attention weight for a signal based on salience.

        Combines intrinsic salience with contextual factors to produce
        a final attention weight that reflects current importance.
        """
        original = base_attention

        # ── Intrinsic salience: how inherently important ──
        intrinsic = self._compute_intrinsic_salience(signal_features)

        # ── Contextual modulation: what's currently relevant ──
        contextual = self._compute_contextual_modulation(
            signal_features, context or {}
        )

        # ── Channel competition: suppress signals in noisy channels ──
        channel = signal_features.get("source", "default")
        channel_suppression = self._compute_channel_suppression(channel)

        # ── Novelty bonus: new/unexpected signals get attention ──
        novelty = self._compute_novelty_bonus(signal_features)

        # ── Combine modulation factors ──
        modulation_factor = (
            intrinsic * 0.35
            + contextual * 0.30
            + (1.0 - channel_suppression) * 0.15
            + novelty * 0.20
        )

        modulated = max(0.05, min(1.0, base_attention * modulation_factor))

        # Determine reason
        if modulated > original * 1.2:
            reason = "amplified: high salience"
        elif modulated < original * 0.8:
            reason = "suppressed: low salience or noisy channel"
        else:
            reason = "maintained: appropriate attention level"

        # Update channel activity
        self._channel_activity[channel] = (
            self._channel_activity.get(channel, 0.5) * 0.9
            + modulated * 0.1
        )

        # Record for history
        self._attention_history.append({
            "signal_id": signal_id,
            "original": original,
            "modulated": modulated,
            "intrinsic": intrinsic,
            "contextual": contextual,
            "timestamp": time.time(),
        })

        return SalienceResult(
            signal_id=signal_id,
            original_attention=original,
            modulated_attention=modulated,
            modulation_factor=modulation_factor,
            amplification_reason=reason,
        )

    def _compute_intrinsic_salience(self, features: dict[str, Any]) -> float:
        """Compute how inherently salient a signal is."""
        score = 0.5

        # Source authority
        source = str(features.get("source", "")).lower()
        if any(t in source for t in ["wikipedia", "arxiv", "nature", "edu", "gov"]):
            score += 0.2

        # Information density
        text = str(features.get("text", ""))
        if len(text.split()) > 30:
            score += 0.15

        # Causal/contradictory language (high salience)
        if re.search(r'(because|therefore|however|contradict|refute)', text, re.IGNORECASE):
            score += 0.15

        # Recency
        if features.get("has_recency"):
            score += 0.1

        return max(0.0, min(1.0, score))

    def _compute_contextual_modulation(
        self,
        features: dict[str, Any],
        context: dict[str, Any],
    ) -> float:
        """Modulate based on current reasoning context."""
        score = 0.5

        # If we're looking for contradictions, contradiction signals get boost
        if context.get("seeking_contradictions"):
            text = str(features.get("text", ""))
            if re.search(r'(however|but|although|contradict)', text, re.IGNORECASE):
                score += 0.3

        # If credibility is in question, source-heavy signals get boost
        if context.get("credibility_concern"):
            if features.get("source") and features.get("source") != "unknown":
                score += 0.25

        # If temporal context matters, recent signals get boost
        if context.get("temporal_focus"):
            if features.get("has_recency"):
                score += 0.2

        # Evidence count: if we have little evidence, each piece matters more
        evidence_count = context.get("evidence_count", 10)
        if evidence_count < 3:
            score += 0.15  # scarce evidence → each piece is more salient

        return max(0.0, min(1.0, score))

    def _compute_channel_suppression(self, channel: str) -> float:
        """
        Suppress signals from channels that have been noisy.

        Like habituation in the brain (ignoring repetitive stimuli),
        this reduces attention to channels that consistently produce
        low-value signals.
        """
        activity = self._channel_activity.get(channel, 0.5)
        # If channel has been producing low-attention signals, suppress it
        if activity < 0.3:
            return 0.3  # suppress noisy channel
        elif activity > 0.7:
            return 0.0  # don't suppress active channel
        return 0.1  # mild suppression for average channels

    def _compute_novelty_bonus(self, features: dict[str, Any]) -> float:
        """Give a bonus to novel/unexpected signals."""
        score = 0.3  # base novelty

        # Signals with unique features get novelty bonus
        if features.get("has_citations"):
            score += 0.2
        if features.get("has_url"):
            score += 0.1
        if features.get("has_entities"):
            score += 0.15

        return max(0.0, min(1.0, score))

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "history_size": len(self._attention_history),
            "active_channels": len(self._channel_activity),
        }


# ──────────────────────────────────────────────────────────────────
# INHIBITORY GATING — Thalamic Reticular Nucleus
# ──────────────────────────────────────────────────────────────────


class InhibitoryGate:
    """
    Block signals that are currently irrelevant.

    Like the thalamic reticular nucleus (TRN) that forms a shell
    around the thalamus and selectively inhibits thalamic relay
    neurons, this module:

    1. Maintains a relevance mask: which topics/channels are currently relevant
    2. Applies inhibition to signals outside the current focus
    3. Implements the "cocktail party effect": focus on one signal stream
       while suppressing others
    4. Adapts the mask based on reasoning context

    The TRN is often called the "brain's attention switchboard" —
    it determines which signals reach the cortex and which are blocked.
    """

    def __init__(self) -> None:
        # Relevance mask: topic/channel → relevance score
        self._relevance_mask: dict[str, float] = {}
        # Active focus: what we're currently attending to
        self._active_focus: list[str] = []
        # Inhibition history
        self._inhibition_history: list[dict[str, Any]] = []
        # Mask learning rate
        self._mask_learning_rate = 0.2

    def apply_gate(
        self,
        signal_id: str,
        signal_features: dict[str, Any],
        base_attention: float,
        context: dict[str, Any] | None = None,
    ) -> tuple[float, str]:
        """
        Apply inhibitory gating to a signal.

        Returns:
            (gated_attention, reason) where gated_attention is the
            attention after inhibitory gating, and reason explains
            the gating decision.
        """
        # Extract the channel/topic for this signal
        channel = signal_features.get("source", "unknown")
        text = str(signal_features.get("text", "")).lower()

        # Check relevance mask
        channel_relevance = self._relevance_mask.get(channel, 0.5)

        # Check active focus alignment
        focus_alignment = 0.0
        if self._active_focus:
            signal_words = set(re.findall(r'\b\w{4,}\b', text))
            focus_words = set()
            for f in self._active_focus:
                focus_words.update(set(f.lower().split()))
            if signal_words and focus_words:
                overlap = len(signal_words & focus_words)
                focus_alignment = overlap / max(len(signal_words), 1)

        # Compute inhibition level
        if focus_alignment > 0.3:
            # Signal aligns with current focus → low inhibition
            inhibition = 0.1
            reason = "aligned with current focus"
        elif channel_relevance > 0.7:
            # High-relevance channel → low inhibition
            inhibition = 0.15
            reason = "high-relevance channel"
        elif channel_relevance < 0.3:
            # Low-relevance channel → high inhibition
            inhibition = 0.6
            reason = "low-relevance channel suppressed"
        else:
            # Default: mild inhibition based on attention
            inhibition = 0.3 * (1.0 - base_attention)
            reason = "default inhibitory gating"

        # Apply context-based modulation
        if context:
            if context.get("focus_topic"):
                topic = context["focus_topic"].lower()
                if topic in text:
                    inhibition *= 0.3  # strongly reduce inhibition for focused topic
                    reason = f"matches focus topic: {context['focus_topic']}"

        gated = max(0.05, base_attention * (1.0 - inhibition))

        # Record
        self._inhibition_history.append({
            "signal_id": signal_id,
            "channel": channel,
            "inhibition": inhibition,
            "gated_attention": gated,
            "timestamp": time.time(),
        })

        return gated, reason

    def update_focus(self, focus_topics: list[str]) -> None:
        """Update the active focus topics."""
        self._active_focus = focus_topics

    def update_relevance(self, channel: str, relevance: float) -> None:
        """Update channel relevance based on outcomes."""
        old = self._relevance_mask.get(channel, 0.5)
        self._relevance_mask[channel] = (
            old * (1.0 - self._mask_learning_rate)
            + relevance * self._mask_learning_rate
        )

    def suppress_channel(self, channel: str, amount: float = 0.5) -> None:
        """Explicitly suppress a channel (e.g., after bad output)."""
        old = self._relevance_mask.get(channel, 0.5)
        self._relevance_mask[channel] = max(0.0, old - amount)

    def boost_channel(self, channel: str, amount: float = 0.3) -> None:
        """Explicitly boost a channel (e.g., after good output)."""
        old = self._relevance_mask.get(channel, 0.5)
        self._relevance_mask[channel] = min(1.0, old + amount)

    @property
    def active_focus(self) -> list[str]:
        return list(self._active_focus)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "relevance_mask_size": len(self._relevance_mask),
            "active_focus_count": len(self._active_focus),
            "inhibition_history_size": len(self._inhibition_history),
            "avg_inhibition": (
                sum(h["inhibition"] for h in self._inhibition_history[-20:])
                / min(20, len(self._inhibition_history))
                if self._inhibition_history else 0.0
            ),
        }
