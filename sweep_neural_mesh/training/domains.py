"""
Expertise Domains — Per-domain tracking, scoring, and curriculum management.

§3: 17 minimum domains, extensible.
§21: Per-domain expertise scores.
§22: Confidence calibration tracking.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DomainScore:
    """Score for a single expertise domain."""
    domain: str
    attempts: int = 0
    correct: int = 0
    total_confidence: float = 0.0
    total_correct_confidence: float = 0.0
    total_wrong_confidence: float = 0.0
    wrong_count: int = 0
    high_conf_wrong: int = 0
    current_level: int = 1
    mastery_threshold: float = 0.90
    consecutive_mastery: int = 0
    mastery_required: int = 5
    last_improvement_iteration: int = 0
    error_history: dict[str, int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.attempts)

    @property
    def avg_confidence(self) -> float:
        return self.total_confidence / max(1, self.attempts)

    @property
    def calibration_gap(self) -> float:
        """Difference between avg confidence and actual accuracy."""
        return self.avg_confidence - self.accuracy

    @property
    def is_mastered(self) -> bool:
        return self.consecutive_mastery >= self.mastery_required

    def record(self, correct: bool, confidence: float, error_type: str = "") -> None:
        self.attempts += 1
        self.total_confidence += confidence
        if correct:
            self.correct += 1
            self.total_correct_confidence += confidence
        else:
            self.wrong_count += 1
            self.total_wrong_confidence += confidence
            if confidence >= 0.8:
                self.high_conf_wrong += 1
            if error_type:
                self.error_history[error_type] = self.error_history.get(error_type, 0) + 1

    def check_mastery(self) -> bool:
        if self.accuracy >= self.mastery_threshold and self.attempts >= 10:
            self.consecutive_mastery += 1
        else:
            self.consecutive_mastery = 0
        return self.is_mastered

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "accuracy": round(self.accuracy, 4),
            "attempts": self.attempts,
            "correct": self.correct,
            "current_level": self.current_level,
            "avg_confidence": round(self.avg_confidence, 4),
            "calibration_gap": round(self.calibration_gap, 4),
            "high_conf_wrong": self.high_conf_wrong,
            "mastery_streak": self.consecutive_mastery,
            "is_mastered": self.is_mastered,
            "error_history": self.error_history,
        }


DEFAULT_DOMAINS = [
    "logic",
    "reasoning",
    "deduction",
    "induction",
    "transitivity",
    "relational_reasoning",
    "temporal_reasoning",
    "spatial_reasoning",
    "evidence_evaluation",
    "contradiction_detection",
    "ambiguity_resolution",
    "uncertainty",
    "pattern_recognition",
    "multi_step_planning",
    "graph_reasoning",
    "causal_reasoning",
    "novel_structure_reasoning",
]


class ExpertiseTracker:
    """
    Tracks per-domain expertise scores and curriculum progress.

    §21: Per-domain scores.
    §18: Curriculum level management.
    §22: Confidence calibration.
    """

    def __init__(
        self,
        domains: list[str] | None = None,
        mastery_threshold: float = 0.90,
        mastery_required: int = 5,
    ) -> None:
        self._domains = domains or DEFAULT_DOMAINS
        self._scores: dict[str, DomainScore] = {}
        for d in self._domains:
            self._scores[d] = DomainScore(
                domain=d,
                mastery_threshold=mastery_threshold,
                mastery_required=mastery_required,
            )

    def get_score(self, domain: str) -> DomainScore:
        if domain not in self._scores:
            self._scores[domain] = DomainScore(domain=domain)
        return self._scores[domain]

    def record_result(
        self,
        domain: str,
        correct: bool,
        confidence: float,
        error_type: str = "",
    ) -> None:
        self.get_score(domain).record(correct, confidence, error_type)

    def check_all_mastery(self) -> dict[str, bool]:
        return {d: self._scores[d].check_mastery() for d in self._domains}

    def get_weakest_domains(self, n: int = 3) -> list[str]:
        scored = [(d, self._scores[d].accuracy) for d in self._domains if self._scores[d].attempts > 0]
        scored.sort(key=lambda x: x[1])
        return [d for d, _ in scored[:n]]

    def get_strongest_domains(self, n: int = 3) -> list[str]:
        scored = [(d, self._scores[d].accuracy) for d in self._domains if self._scores[d].attempts > 0]
        scored.sort(key=lambda x: -x[1])
        return [d for d, _ in scored[:n]]

    @property
    def overall_accuracy(self) -> float:
        total_a = sum(s.attempts for s in self._scores.values())
        total_c = sum(s.correct for s in self._scores.values())
        return total_c / max(1, total_a)

    @property
    def overall_confidence_calibration(self) -> float:
        total_a = sum(s.attempts for s in self._scores.values())
        total_gap = sum(s.calibration_gap * s.attempts for s in self._scores.values())
        return total_gap / max(1, total_a)

    def get_domain_level(self, domain: str) -> int:
        return self.get_score(domain).current_level

    def advance_level(self, domain: str) -> int:
        score = self.get_score(domain)
        if score.is_mastered and score.current_level < 6:
            score.current_level += 1
            score.consecutive_mastery = 0
        return score.current_level

    def export_scores(self) -> dict[str, dict[str, Any]]:
        """Return per-domain scores as {domain: {score, level, consecutive_correct, ...}}."""
        result = {}
        for domain, score in self._scores.items():
            result[domain] = {
                "score": score.accuracy,
                "level": score.current_level,
                "attempts": score.attempts,
                "correct": score.correct,
                "consecutive_correct": score.consecutive_mastery,
                "is_mastered": score.is_mastered,
                "calibration_gap": round(score.calibration_gap, 4),
            }
        return result

    def mastery_status(self) -> dict[str, int]:
        """Count mastered, practicing, and novice domains."""
        mastered = 0
        practicing = 0
        novice = 0
        for score in self._scores.values():
            if score.is_mastered:
                mastered += 1
            elif score.attempts > 0:
                practicing += 1
            else:
                novice += 1
        return {"mastered": mastered, "practicing": practicing, "novice": novice}

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_accuracy": round(self.overall_accuracy, 4),
            "overall_calibration_gap": round(self.overall_confidence_calibration, 4),
            "domains": {d: s.to_dict() for d, s in self._scores.items()},
        }

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: str | Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d, s_data in data.get("domains", {}).items():
            if d in self._scores:
                s = self._scores[d]
                s.attempts = s_data.get("attempts", 0)
                s.correct = s_data.get("correct", 0)
                s.current_level = s_data.get("current_level", 1)
                s.total_confidence = s_data.get("avg_confidence", 0) * s.attempts
                s.high_conf_wrong = s_data.get("high_conf_wrong", 0)
                s.consecutive_mastery = s_data.get("mastery_streak", 0)
                s.error_history = s_data.get("error_history", {})
