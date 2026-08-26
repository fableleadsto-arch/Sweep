"""
Confidence Calibration — Learn that "I don't know" can be correct.

§22: Measure confidence, correctness, calibration, uncertainty.
Penalize high confidence + wrong answer more heavily than low confidence + wrong.
Reward appropriate uncertainty + insufficient evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CalibrationRecord:
    """A single calibration observation."""
    confidence: float
    correct: bool
    domain: str
    difficulty: int


class ConfidenceCalibrator:
    """
    Tracks and improves confidence calibration.

    §22: Sweep must learn that "I don't know" can be the correct answer.
    """

    def __init__(self) -> None:
        self._records: list[CalibrationRecord] = []
        self._bins = 10
        self._bin_data: dict[int, dict[str, Any]] = {}

    def record(self, confidence: float, correct: bool, domain: str = "", difficulty: int = 1) -> None:
        self._records.append(CalibrationRecord(
            confidence=confidence, correct=correct,
            domain=domain, difficulty=difficulty,
        ))

    def expected_calibration_error(self) -> float:
        """Compute Expected Calibration Error (ECE)."""
        if not self._records:
            return 0.0

        bin_size = 1.0 / self._bins
        total = len(self._records)
        ece = 0.0

        for i in range(self._bins):
            low = i * bin_size
            high = (i + 1) * bin_size
            bin_records = [r for r in self._records if low <= r.confidence < high]
            if not bin_records:
                continue
            bin_acc = sum(1 for r in bin_records if r.correct) / len(bin_records)
            bin_conf = sum(r.confidence for r in bin_records) / len(bin_records)
            ece += len(bin_records) / total * abs(bin_acc - bin_conf)

        return ece

    def overconfidence_penalty(self) -> float:
        """Penalize high confidence + wrong answer."""
        high_conf_wrong = [
            r for r in self._records
            if r.confidence >= 0.8 and not r.correct
        ]
        if not self._records:
            return 0.0
        return len(high_conf_wrong) / len(self._records)

    def appropriate_uncertainty_reward(self) -> float:
        """Reward appropriate uncertainty + insufficient evidence."""
        low_conf_correct = [
            r for r in self._records
            if r.confidence <= 0.5 and r.correct
        ]
        if not self._records:
            return 0.0
        return len(low_conf_correct) / len(self._records)

    def calibration_summary(self) -> dict[str, Any]:
        return {
            "total_records": len(self._records),
            "ece": round(self.expected_calibration_error(), 4),
            "overconfidence_penalty": round(self.overconfidence_penalty(), 4),
            "appropriate_uncertainty_reward": round(self.appropriate_uncertainty_reward(), 4),
            "avg_confidence": round(sum(r.confidence for r in self._records) / max(1, len(self._records)), 4),
            "overall_accuracy": round(sum(1 for r in self._records if r.correct) / max(1, len(self._records)), 4),
        }
