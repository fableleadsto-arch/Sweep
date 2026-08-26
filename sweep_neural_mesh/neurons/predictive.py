"""
Predictive Coding, Reflexive Shortcuts, and Energy Gating — the Hindbrain survival layer.

These three mechanisms operate at the lowest level of the brain,
processing input BEFORE it reaches higher centers:

1. PREDICTIVE CODING (Cerebellum + Brainstem):
   Generate a prediction about incoming data BEFORE full processing.
   Compare prediction with actual results. Only process the DIFFERENCE
   (prediction error). This is how the brain processes 90% of input
   unconsciously — it predicts, then corrects only when wrong.

2. REFLEXIVE SHORTCUTS (Brainstem Reflexes):
   For well-known patterns, bypass ALL higher processing entirely.
   Like pulling your hand from a hot stove before feeling pain.
   In Sweep: if we've seen this exact query pattern and know the answer,
   skip the full pipeline and return immediately.

3. ENERGY GATING (Hypothalamus + Reticular Formation):
   Monitor system load and defer non-urgent processing when resources
   are scarce. Like sleep conserving energy, or the brain deprioritizing
   non-essential functions during stress.

Signal flow in the hindbrain:

    Raw Input
        ↓
    Predictive Coding: generate hypothesis
        ↓
    Reflexive Check: match against known patterns
        ↓  (if match → shortcut, skip higher processing)
    Energy Gate: check resource availability
        ↓  (if overloaded → defer/drop low-priority)
    Output to Midbrain (with prediction context)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ──────────────────────────────────────────────────────────────────
# PREDICTIVE CODING
# ──────────────────────────────────────────────────────────────────


@dataclass
class Prediction:
    """A prediction about incoming data."""
    predicted_type: str            # what we think this input is about
    predicted_relevance: float     # 0.0-1.0: how relevant we predict it is
    predicted_sources: list[str]   # which sources we expect to find
    confidence: float              # 0.0-1.0: how confident in our prediction
    hypothesis: str                # natural language hypothesis
    timestamp: float = field(default_factory=time.time)


@dataclass
class PredictionError:
    """The difference between prediction and reality."""
    predicted: Prediction
    actual_type: str
    actual_relevance: float
    error_magnitude: float         # 0.0-1.0: how wrong was the prediction
    error_components: list[str]    # which aspects were wrong
    learning_delta: float          # how much to update the model


class PredictiveCoder:
    """
    Generate predictions about incoming data before full processing.

    Like the cerebellum that predicts sensory consequences of actions,
    this module:

    1. Analyzes input patterns to generate a hypothesis
    2. Predicts what type of content it is and how relevant
    3. Provides the prediction to downstream centers so they can
       compare against actual results
    4. Learns from prediction errors to improve future predictions

    This saves compute by allowing downstream centers to focus on
    processing the DIFFERENCE between prediction and reality, rather
    than processing everything from scratch.
    """

    def __init__(self) -> None:
        # Learned patterns: query_pattern → prediction
        self._pattern_predictions: dict[str, dict[str, Any]] = {}
        # Prediction history for learning
        self._prediction_history: list[dict[str, Any]] = []
        # Total predictions and errors
        self._total_predictions = 0
        self._total_errors = 0
        # Error accumulator for learning
        self._learning_rate = 0.1

    def predict(self, query: str, evidence: list[dict]) -> Prediction:
        """
        Generate a prediction about incoming data.

        Called BEFORE any expensive processing. Uses pattern matching
        and learned associations to predict:
        - What type of query this is
        - How relevant the evidence will be
        - Which sources are likely involved
        """
        self._total_predictions += 1

        query_lower = query.lower().strip()
        word_count = len(query_lower.split())

        # ── Step 1: Check for learned pattern match ──
        learned = self._check_learned_patterns(query_lower)
        if learned:
            return learned

        # ── Step 2: Generate prediction from heuristics ──
        predicted_type = self._predict_type(query_lower)
        predicted_relevance = self._predict_relevance(query_lower, word_count)
        predicted_sources = self._predict_sources(query_lower, evidence)

        # Confidence based on how well we can predict
        confidence = 0.5  # base
        if learned:
            confidence = 0.8
        elif predicted_relevance > 0.7:
            confidence = 0.65  # high relevance → more predictable
        elif word_count > 5:
            confidence = 0.55  # longer queries are more specific

        hypothesis = self._generate_hypothesis(
            predicted_type, predicted_relevance, predicted_sources
        )

        prediction = Prediction(
            predicted_type=predicted_type,
            predicted_relevance=predicted_relevance,
            predicted_sources=predicted_sources,
            confidence=confidence,
            hypothesis=hypothesis,
        )

        return prediction

    def compute_error(
        self,
        prediction: Prediction,
        actual_relevance: float,
        actual_sources: list[str],
    ) -> PredictionError:
        """
        Compute the error between prediction and reality.

        This drives learning: large errors update the model more.
        """
        # Relevance error
        relevance_error = abs(prediction.predicted_relevance - actual_relevance)

        # Source prediction error
        predicted_source_set = set(prediction.predicted_sources)
        actual_source_set = set(actual_sources)
        if predicted_source_set or actual_source_set:
            source_overlap = len(predicted_source_set & actual_source_set)
            source_total = len(predicted_source_set | actual_source_set)
            source_error = 1.0 - (source_overlap / source_total if source_total > 0 else 0.0)
        else:
            source_error = 0.0

        # Overall error magnitude
        error_magnitude = relevance_error * 0.6 + source_error * 0.4

        # Error components
        error_components = []
        if relevance_error > 0.2:
            error_components.append("relevance_mispredicted")
        if source_error > 0.3:
            error_components.append("sources_mispredicted")
        if error_magnitude < 0.1:
            error_components.append("accurate_prediction")

        # Learning delta: how much to update patterns
        learning_delta = error_magnitude * self._learning_rate

        return PredictionError(
            predicted=prediction,
            actual_type="",
            actual_relevance=actual_relevance,
            error_magnitude=error_magnitude,
            error_components=error_components,
            learning_delta=learning_delta,
        )

    def learn_from_error(self, error: PredictionError) -> None:
        """
        Update prediction model based on error.

        Like cerebellar learning that refines motor predictions,
        this improves future predictions by adjusting pattern weights.
        """
        if error.error_magnitude > 0.3:
            self._total_errors += 1

        # Record for pattern learning
        self._prediction_history.append({
            "predicted_type": error.predicted.predicted_type,
            "predicted_relevance": error.predicted.predicted_relevance,
            "actual_relevance": error.actual_relevance,
            "error": error.error_magnitude,
            "timestamp": time.time(),
        })

        # Keep history bounded
        if len(self._prediction_history) > 1000:
            self._prediction_history = self._prediction_history[-1000:]

    def _check_learned_patterns(self, query: str) -> Prediction | None:
        """Check if we've seen this query pattern before."""
        # Simple pattern matching: extract keywords
        keywords = set(re.findall(r'\b\w{3,}\b', query))
        if not keywords:
            return None

        best_match = None
        best_score = 0.0

        for pattern_key, pattern_data in self._pattern_predictions.items():
            pattern_words = set(pattern_key.split())
            overlap = len(keywords & pattern_words)
            total = len(keywords | pattern_words)
            if total == 0:
                continue

            similarity = overlap / total
            if similarity > best_score and similarity > 0.6:
                best_score = similarity
                best_match = pattern_data

        if best_match:
            return Prediction(
                predicted_type=best_match.get("type", "unknown"),
                predicted_relevance=best_match.get("relevance", 0.5),
                predicted_sources=best_match.get("sources", []),
                confidence=best_match.get("confidence", 0.5) * best_score,
                hypothesis=f"Learned pattern match ({best_score:.0%} similarity)",
            )

        return None

    def _predict_type(self, query: str) -> str:
        """Predict the type/category of the query."""
        if re.search(r'\b(how|step|method|process|guide)\b', query):
            return "how_to"
        if re.search(r'\b(what|define|meaning|explain)\b', query):
            return "definition"
        if re.search(r'\b(why|reason|cause|because)\b', query):
            return "explanation"
        if re.search(r'\b(compare|difference|versus|vs|better)\b', query):
            return "comparison"
        if re.search(r'\b(bug|error|fix|issue|problem|crash)\b', query):
            return "troubleshooting"
        if re.search(r'\b(best|recommend|suggest|should)\b', query):
            return "recommendation"
        return "general"

    def _predict_relevance(self, query: str, word_count: int) -> float:
        """Predict how relevant evidence will be."""
        score = 0.5

        # Question words indicate specific intent
        if re.search(r'^(how|what|why|when|where|which|who)\b', query):
            score += 0.15

        # Technical terms indicate high-value queries
        tech_terms = [
            "python", "javascript", "api", "database", "security",
            "performance", "algorithm", "machine", "learning", "docker",
            "kubernetes", "terraform", "aws", "azure", "linux",
        ]
        if any(t in query for t in tech_terms):
            score += 0.15

        # Longer queries are more specific
        if word_count > 8:
            score += 0.1
        elif word_count < 3:
            score -= 0.15

        return max(0.0, min(1.0, score))

    def _predict_sources(self, query: str, evidence: list[dict]) -> list[str]:
        """Predict which sources will be relevant."""
        sources = []
        if re.search(r'\b(paper|research|study|journal)\b', query):
            sources.append("academic")
        if re.search(r'\b(documentation|docs|api|reference)\b', query):
            sources.append("official_docs")
        if re.search(r'\b(github|repo|code|source)\b', query):
            sources.append("code_repository")
        if re.search(r'\b(news|latest|recent|update)\b', query):
            sources.append("news")
        return sources

    def _generate_hypothesis(
        self,
        predicted_type: str,
        predicted_relevance: float,
        predicted_sources: list[str],
    ) -> str:
        """Generate a natural language hypothesis."""
        type_descriptions = {
            "how_to": "user seeks a procedural explanation",
            "definition": "user seeks a conceptual definition",
            "explanation": "user seeks causal reasoning",
            "comparison": "user seeks comparative analysis",
            "troubleshooting": "user needs problem resolution",
            "recommendation": "user needs decision support",
            "general": "user has a general inquiry",
        }
        desc = type_descriptions.get(predicted_type, "general inquiry")
        source_str = f" from {', '.join(predicted_sources)}" if predicted_sources else ""
        return f"Predicted: {desc}{source_str} (relevance: {predicted_relevance:.0%})"

    @property
    def accuracy(self) -> float:
        """Prediction accuracy over time."""
        if self._total_predictions == 0:
            return 0.0
        return 1.0 - (self._total_errors / self._total_predictions)


# ──────────────────────────────────────────────────────────────────
# REFLEXIVE SHORTCUTS
# ──────────────────────────────────────────────────────────────────


@dataclass
class ReflexiveMatch:
    """A match against a known reflexive pattern."""
    pattern_id: str
    response: str
    confidence: float
    shortcut_path: list[str]      # which centers to skip to
    match_quality: float          # 0.0-1.0: how well does it match


class ReflexiveSystem:
    """
    Bypass full processing for well-known patterns.

    Like brainstem reflexes that respond to stimuli before conscious
    awareness (e.g., pulling hand from hot stove), this system:

    1. Maintains a database of known query patterns and their optimal paths
    2. Matches incoming queries against these patterns
    3. If matched, returns the cached response path immediately
    4. Learns new reflexive patterns from successful reasoning episodes

    This is the "muscle memory" of Sweep's brain — routine queries
    get handled automatically, freeing compute for novel queries.
    """

    def __init__(self) -> None:
        # Learned reflexive patterns
        self._patterns: dict[str, dict[str, Any]] = {}
        # Usage statistics
        self._shortcut_hits = 0
        self._shortcut_misses = 0
        # Pattern learning threshold
        self._min_episodes = 3  # need this many similar episodes to form reflex

    def check_reflex(
        self,
        query: str,
        prediction: Prediction,
    ) -> ReflexiveMatch | None:
        """
        Check if this query matches a known reflexive pattern.

        Returns a ReflexiveMatch if matched, None if full processing needed.
        """
        query_lower = query.lower().strip()
        query_words = set(re.findall(r'\b\w{3,}\b', query_lower))

        best_match = None
        best_quality = 0.0

        for pattern_id, pattern in self._patterns.items():
            pattern_words = set(pattern.get("keywords", set()))
            if not pattern_words or not query_words:
                continue

            overlap = len(query_words & pattern_words)
            total = len(query_words | pattern_words)
            similarity = overlap / total if total > 0 else 0.0

            # Also check query type match
            type_match = (
                pattern.get("query_type") == prediction.predicted_type
                if pattern.get("query_type")
                else True
            )

            # Combined quality score
            quality = similarity * 0.7 + (0.3 if type_match else 0.0)

            if quality > best_quality and quality > 0.5:
                best_quality = quality
                best_match = pattern

        if best_match:
            self._shortcut_hits += 1
            return ReflexiveMatch(
                pattern_id=best_match["id"],
                response=best_match.get("response", ""),
                confidence=best_match.get("confidence", 0.5) * best_quality,
                shortcut_path=best_match.get("shortcut_path", []),
                match_quality=best_quality,
            )

        self._shortcut_misses += 1
        return None

    def learn_pattern(
        self,
        query: str,
        query_type: str,
        successful_path: list[str],
        confidence: float,
    ) -> None:
        """
        Learn a new reflexive pattern from a successful reasoning episode.

        Like motor learning that converts deliberate practice into
        automatic reflexes, this stores successful processing paths
        for future shortcut use.
        """
        query_lower = query.lower().strip()
        keywords = set(re.findall(r'\b\w{3,}\b', query_lower))

        # Generate pattern ID from keywords
        pattern_id = "_".join(sorted(keywords)[:5])

        if pattern_id in self._patterns:
            # Strengthen existing pattern
            existing = self._patterns[pattern_id]
            existing["confidence"] = min(
                0.95,
                existing["confidence"] + (1.0 - existing["confidence"]) * 0.15,
            )
            existing["use_count"] = existing.get("use_count", 0) + 1
            existing["last_used"] = time.time()
        else:
            # Create new pattern
            self._patterns[pattern_id] = {
                "id": pattern_id,
                "keywords": keywords,
                "query_type": query_type,
                "shortcut_path": successful_path,
                "confidence": confidence * 0.7,  # start conservative
                "use_count": 1,
                "created_at": time.time(),
                "last_used": time.time(),
            }

    def decay_unused(self, decay_rate: float = 0.01) -> None:
        """Decay patterns that haven't been used recently."""
        now = time.time()
        to_remove = []

        for pattern_id, pattern in self._patterns.items():
            last_used = pattern.get("last_used", 0)
            hours_since = (now - last_used) / 3600 if last_used > 0 else 1000

            if hours_since > 168:  # 1 week
                pattern["confidence"] -= decay_rate
                if pattern["confidence"] < 0.1:
                    to_remove.append(pattern_id)

        for pid in to_remove:
            del self._patterns[pid]

    @property
    def hit_rate(self) -> float:
        """How often reflexive shortcuts are used."""
        total = self._shortcut_hits + self._shortcut_misses
        return self._shortcut_hits / total if total > 0 else 0.0

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)


# ──────────────────────────────────────────────────────────────────
# ENERGY GATING
# ──────────────────────────────────────────────────────────────────


class EnergyState(Enum):
    """System energy/load states."""
    FRESH = "fresh"          # plenty of resources
    NORMAL = "normal"        # nominal load
    BUSY = "busy"            # elevated load, defer low-priority
    STRESSED = "stressed"    # high load, only process critical
    EXHAUSTED = "exhausted"  # critical, minimal processing only


class EnergyGating:
    """
    Monitor system load and gate processing accordingly.

    Like the hypothalamus regulating energy expenditure and the
    reticular formation controlling arousal levels, this module:

    1. Tracks processing time, memory usage, and queue depth
    2. Classifies system state: fresh → normal → busy → stressed → exhausted
    3. Gates processing: deferred, dropped, or throttled based on state
    4. Recovers gracefully when load decreases

    This prevents Sweep from consuming all resources on low-priority
    queries when high-priority work is pending.
    """

    def __init__(self) -> None:
        self._state = EnergyState.FRESH
        self._processing_times: list[float] = []
        self._max_history = 100
        self._queue_depth = 0
        self._active_queries = 0

        # Thresholds (in ms)
        self._busy_threshold_ms = 100.0
        self._stressed_threshold_ms = 250.0
        self._exhausted_threshold_ms = 500.0

    def check_energy(self, query_priority: float = 0.5) -> tuple[EnergyState, str]:
        """
        Check current energy state and whether to proceed.

        Returns:
            (energy_state, reason) where reason explains the gating decision.
        """
        # Update state based on recent processing times
        self._update_state()

        # Gate based on state and priority
        if self._state == EnergyState.FRESH:
            return self._state, "plenty of resources, proceed"
        elif self._state == EnergyState.NORMAL:
            return self._state, "nominal load, proceed"
        elif self._state == EnergyState.BUSY:
            if query_priority < 0.3:
                return self._state, "busy: deferring low-priority query"
            return self._state, "busy but query has sufficient priority"
        elif self._state == EnergyState.STRESSED:
            if query_priority < 0.6:
                return self._state, "stressed: dropping low-priority query"
            return self._state, "stressed but query is high-priority"
        else:  # EXHAUSTED
            if query_priority < 0.9:
                return self._state, "exhausted: only critical queries proceed"
            return self._state, "exhausted but query is critical"

    def should_process(self, query_priority: float = 0.5) -> bool:
        """Quick check: should we process this query now?"""
        state, _ = self.check_energy(query_priority)
        return state in (EnergyState.FRESH, EnergyState.NORMAL, EnergyState.BUSY)

    def record_processing_time(self, time_ms: float) -> None:
        """Record a processing time for load tracking."""
        self._processing_times.append(time_ms)
        if len(self._processing_times) > self._max_history:
            self._processing_times = self._processing_times[-self._max_history:]

    def increment_queue(self) -> None:
        """Record a new query entering the queue."""
        self._queue_depth += 1

    def decrement_queue(self) -> None:
        """Record a query completing."""
        self._queue_depth = max(0, self._queue_depth - 1)

    def _update_state(self) -> None:
        """Update energy state based on recent processing times."""
        if not self._processing_times:
            self._state = EnergyState.FRESH
            return

        # Use recent average (last 10)
        recent = self._processing_times[-10:]
        avg_time = sum(recent) / len(recent)

        # Also factor in queue depth
        queue_factor = min(1.0, self._queue_depth / 10.0)

        # Combined load score
        load_score = avg_time / self._exhausted_threshold_ms + queue_factor * 0.3

        if load_score < 0.2:
            self._state = EnergyState.FRESH
        elif load_score < 0.5:
            self._state = EnergyState.NORMAL
        elif load_score < 0.75:
            self._state = EnergyState.BUSY
        elif load_score < 1.0:
            self._state = EnergyState.STRESSED
        else:
            self._state = EnergyState.EXHAUSTED

    @property
    def state(self) -> EnergyState:
        """Current energy state."""
        self._update_state()
        return self._state

    @property
    def stats(self) -> dict[str, Any]:
        """Energy gating statistics."""
        recent_avg = (
            sum(self._processing_times[-10:]) / len(self._processing_times[-10:])
            if self._processing_times else 0.0
        )
        return {
            "state": self._state.value,
            "avg_processing_time_ms": round(recent_avg, 2),
            "queue_depth": self._queue_depth,
            "history_size": len(self._processing_times),
        }
