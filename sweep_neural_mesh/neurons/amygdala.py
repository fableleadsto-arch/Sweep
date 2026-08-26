"""
Amygdala — Emotional Valence Tagging for Evidence.

The amygdala in the human brain assigns emotional significance to stimuli:
- Threatening information gets priority processing
- Novel/surprising events get encoded more strongly
- Emotional memories are recalled more easily

In Sweep, the amygdala tags evidence with emotional valence:
- HIGH urgency: contradicts known facts, contains errors, is critical
- MEDIUM urgency: novel, surprising, or unusual
- LOW urgency: routine, expected, mundane

This valence feeds into attention gating (midbrain) and memory encoding (forebrain):
- High-valence evidence gets more processing time
- High-valence memories are retained longer
- High-valence contradictions trigger deeper investigation

Architecture:

    Evidence Signal
        ↓
    ┌─────────────────────────────────────┐
    │  AMYGDALA                           │
    │                                     │
    │  Valence Detection:                 │
    │  - Threat (contradiction, error)    │
    │  - Novelty (unexpected, unusual)    │
    │  - Reward (confirms, supports)      │
    │  - Arousal (intensity of signal)    │
    │                                     │
    │  Output: ValenceScore               │
    │  - valence: -1.0 to 1.0            │
    │  - arousal: 0.0 to 1.0             │
    │  - category: threat/novelty/reward  │
    │  - priority_multiplier: 1.0–3.0    │
    └─────────────────────────────────────┘
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValenceCategory(Enum):
    """Emotional valence categories."""
    THREAT = "threat"           # contradicts, errors, dangerous claims
    NOVELTY = "novelty"         # new, unusual, surprising
    REWARD = "reward"           # confirms, supports, good outcomes
    NEUTRAL = "neutral"         # routine, expected, mundane


@dataclass
class ValenceScore:
    """Emotional valence assessment of a piece of evidence."""
    valence: float              # -1.0 (negative/threatening) to 1.0 (positive/rewarding)
    arousal: float              # 0.0 (calm) to 1.0 (highly arousing/urgent)
    category: ValenceCategory
    priority_multiplier: float  # 1.0 (normal) to 3.0 (urgent)
    reasoning: str              # why this valence was assigned
    timestamp: float = field(default_factory=time.time)


@dataclass
class EmotionalMemory:
    """An emotionally tagged memory for priority recall."""
    text: str
    valence_score: ValenceScore
    embedding_bits: int = 0     # SimHash fingerprint for recall
    created_at: float = field(default_factory=time.time)
    recall_count: int = 0
    last_recalled: float = 0.0


class Amygdala:
    """
    Emotional valence tagging for evidence.

    Like the biological amygdala, this module:
    1. DETECTS emotional significance of evidence
    2. CATEGORIES it as threat/novelty/reward/neutral
    3. COMPUTES arousal (how much attention it deserves)
    4. AMPLIFIES priority for high-valence evidence
    5. ENCODES emotional memories more strongly

    The valence score feeds into:
    - Midbrain attention gating (arousal → processing time)
    - Forebrain memory encoding (valence → retention strength)
    - Metacognition (threats → uncertainty signals)

    Key insight: not all evidence is equal. A contradiction to a known fact
    deserves 3x more processing time than a routine confirmation.
    """

    def __init__(self) -> None:
        self._valuation_history: list[ValenceScore] = []
        self._emotional_memories: list[EmotionalMemory] = []
        self._max_memories = 500
        # Known threat patterns
        self._threat_patterns: list[str] = [
            r'\b(false|wrong|incorrect|error|mistake|refuted|debunked)\b',
            r'\b(dangerous|harmful|risky|unsafe|toxic|poison)\b',
            r'\b(failed|failure|crash|broken|corrupt|malicious)\b',
            r'\b(warning|alert|critical|urgent|emergency)\b',
            r'\b(contradicts?|conflicts?|disagrees?|opposes?)\b',
        ]
        # Known novelty patterns
        self._novelty_patterns: list[str] = [
            r'\b(breaking|new|first|novel|unprecedented|discover)\b',
            r'\b(unexpected|surprising|unusual|rare|uncommon)\b',
            r'\b(revolutionary|groundbreaking|cutting.?edge|state.of.the.art)\b',
            r'\b(but|however|although|despite|contrary|paradox)\b',
        ]
        # Known reward patterns
        self._reward_patterns: list[str] = [
            r'\b(confirmed|verified|validated|proven|demonstrated)\b',
            r'\b(success|successful|works|effective|efficient|optimal)\b',
            r'\b(improved|better|enhanced|optimized| breakthrough)\b',
            r'\b(supports?|corroborates?|agrees?|consistent)\b',
        ]

    def evaluate(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> ValenceScore:
        """
        Evaluate the emotional valence of a piece of evidence.

        Returns a ValenceScore with valence, arousal, category, and priority.
        """
        text_lower = text.lower()
        context = context or {}

        # Score each dimension
        threat_score = self._detect_threat(text_lower)
        novelty_score = self._detect_novelty(text_lower)
        reward_score = self._detect_reward(text_lower)

        # Determine primary category
        scores = {
            ValenceCategory.THREAT: threat_score,
            ValenceCategory.NOVELTY: novelty_score,
            ValenceCategory.REWARD: reward_score,
        }
        primary_category = max(scores, key=scores.get)
        primary_score = scores[primary_category]

        # If all low, it's neutral
        if primary_score < 0.2:
            primary_category = ValenceCategory.NEUTRAL
            valence = 0.0
            arousal = 0.1
            priority = 1.0
            reasoning = "routine evidence, no emotional significance"
        elif primary_category == ValenceCategory.THREAT:
            valence = -primary_score
            arousal = min(1.0, primary_score * 1.3)
            priority = 1.0 + primary_score * 2.0
            reasoning = f"threat detected (score={primary_score:.2f}): contradicts or errors"
        elif primary_category == ValenceCategory.NOVELTY:
            valence = primary_score * 0.3  # novelty is mildly positive
            arousal = min(1.0, primary_score * 1.1)
            priority = 1.0 + primary_score * 1.5
            reasoning = f"novelty detected (score={primary_score:.2f}): new or unusual information"
        else:  # REWARD
            valence = primary_score
            arousal = primary_score * 0.5  # reward is calming
            priority = 1.0 + primary_score * 0.5
            reasoning = f"reward detected (score={primary_score:.2f}): confirms or supports"

        # Context adjustments
        if context.get("is_contradiction", False):
            arousal = min(1.0, arousal + 0.3)
            priority = min(3.0, priority + 0.5)
        if context.get("source_trust", 0.5) < 0.3:
            arousal = min(1.0, arousal + 0.2)

        # Clamp values
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        priority = max(1.0, min(3.0, priority))

        result = ValenceScore(
            valence=valence,
            arousal=arousal,
            category=primary_category,
            priority_multiplier=priority,
            reasoning=reasoning,
        )

        self._valuation_history.append(result)
        return result

    def _detect_threat(self, text: str) -> float:
        """Detect threatening content."""
        matches = sum(1 for p in self._threat_patterns if re.search(p, text))
        return min(1.0, matches * 0.35)

    def _detect_novelty(self, text: str) -> float:
        """Detect novel/unusual content."""
        matches = sum(1 for p in self._novelty_patterns if re.search(p, text))
        return min(1.0, matches * 0.3)

    def _detect_reward(self, text: str) -> float:
        """Detect rewarding/confirming content."""
        matches = sum(1 for p in self._reward_patterns if re.search(p, text))
        return min(1.0, matches * 0.3)

    def encode_emotional_memory(
        self,
        text: str,
        valence: ValenceScore,
        embedding_bits: int = 0,
    ) -> EmotionalMemory:
        """
        Encode an emotionally tagged memory for priority recall.

        High-valence memories are retained longer and recalled more easily.
        """
        memory = EmotionalMemory(
            text=text[:500],
            valence_score=valence,
            embedding_bits=embedding_bits,
        )
        self._emotional_memories.append(memory)

        # Trim if over capacity (keep highest arousal)
        if len(self._emotional_memories) > self._max_memories:
            self._emotional_memories.sort(key=lambda m: m.valence_score.arousal, reverse=True)
            self._emotional_memories = self._emotional_memories[:self._max_memories]

        return memory

    def recall_emotional(
        self,
        query_embedding_bits: int,
        top_k: int = 5,
    ) -> list[EmotionalMemory]:
        """
        Recall emotionally tagged memories similar to a query.

        Uses bit-level Hamming distance for fast matching.
        High-arousal memories get a boost in ranking.
        """
        if not self._emotional_memories or query_embedding_bits == 0:
            return []

        scored: list[tuple[float, EmotionalMemory]] = []
        for mem in self._emotional_memories:
            if mem.embedding_bits == 0:
                continue
            # Hamming distance
            distance = bin(mem.embedding_bits ^ query_embedding_bits).count("1")
            similarity = 1.0 - (distance / 128.0)

            # Arousal boost: high-arousal memories are more recallable
            arousal_boost = 1.0 + mem.valence_score.arousal * 0.5
            score = similarity * arousal_boost

            if score > 0.3:
                scored.append((score, mem))
                mem.recall_count += 1
                mem.last_recalled = time.time()

        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    @property
    def stats(self) -> dict[str, Any]:
        category_counts = {}
        for v in self._valuation_history:
            cat = v.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        avg_arousal = (
            sum(v.arousal for v in self._valuation_history)
            / len(self._valuation_history)
            if self._valuation_history else 0.0
        )
        return {
            "total_valuations": len(self._valuation_history),
            "category_distribution": category_counts,
            "avg_arousal": round(avg_arousal, 4),
            "emotional_memories": len(self._emotional_memories),
        }
