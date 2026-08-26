"""
Dashboard — Training metrics, progress, and reporting.

§32: Dashboard displaying domains, strength/weakness, tasks run,
mastered vs practice vs novices, path to 100%.
§35: No hard-coded "final percentage" — path to 100%.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sweep_neural_mesh.training.domains import ExpertiseTracker
from sweep_neural_mesh.training.calibration import ConfidenceCalibrator
from sweep_neural_mesh.training.versioning import VersionManager


@dataclass
class DashboardSnapshot:
    """Point-in-time snapshot of training state."""
    timestamp: float
    iteration: int
    domain_scores: dict[str, Any]
    overall_accuracy: float
    calibration_ece: float
    tasks_attempted: int
    tasks_correct: int
    version: str
    mastery_progress: dict[str, Any]


class Dashboard:
    """
    §32: Dashboard displaying domains, strength/weakness, tasks run,
    mastered vs practice vs novices, path to 100%.
    §35: No hard-coded "final percentage".
    """

    def __init__(
        self,
        expertise: ExpertiseTracker,
        calibrator: ConfidenceCalibrator,
        version_manager: VersionManager,
    ) -> None:
        self._expertise = expertise
        self._calibrator = calibrator
        self._version_manager = version_manager
        self._snapshots: list[DashboardSnapshot] = []
        self._start_time = time.time()

    def take_snapshot(self) -> DashboardSnapshot:
        scores = self._expertise.export_scores()
        cal_summary = self._calibrator.calibration_summary()
        mastery = self._expertise.mastery_status()

        snapshot = DashboardSnapshot(
            timestamp=time.time(),
            iteration=0,
            domain_scores=scores,
            overall_accuracy=cal_summary["overall_accuracy"],
            calibration_ece=cal_summary["ece"],
            tasks_attempted=cal_summary["total_records"],
            tasks_correct=int(cal_summary["total_records"] * cal_summary["overall_accuracy"]),
            version=self._version_manager.current_version,
            mastery_progress=mastery,
        )
        self._snapshots.append(snapshot)
        return snapshot

    def render(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("  SWEEP TRAINING DASHBOARD")
        lines.append("=" * 60)
        lines.append("")

        mastery = self._expertise.mastery_status()
        lines.append("MASTERY STATUS:")
        lines.append(f"  Mastered:      {mastery['mastered']}")
        lines.append(f"  Practicing:    {mastery['practicing']}")
        lines.append(f"  Novice:        {mastery['novice']}")
        lines.append("")

        lines.append("DOMAIN SCORES:")
        scores = self._expertise.export_scores()
        sorted_scores = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
        for domain, data in sorted_scores:
            level = data["level"]
            score = data["score"]
            consecutive = data["consecutive_correct"]
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            lines.append(f"  {domain:<28} L{level} {bar} {score:.1%} ({consecutive}x)")
        lines.append("")

        weakest = self._expertise.get_weakest_domains(3)
        strongest = self._expertise.get_strongest_domains(3)

        lines.append("WEAKEST DOMAINS:")
        for w in weakest:
            lines.append(f"  - {w}")
        lines.append("")

        lines.append("STRONGEST DOMAINS:")
        for s in strongest:
            lines.append(f"  - {s}")
        lines.append("")

        cal_summary = self._calibrator.calibration_summary()
        lines.append("CONFIDENCE CALIBRATION:")
        lines.append(f"  ECE:                       {cal_summary['ece']:.4f}")
        lines.append(f"  Overconfidence penalty:     {cal_summary['overconfidence_penalty']:.4f}")
        lines.append(f"  Appropriate uncertainty:    {cal_summary['appropriate_uncertainty_reward']:.4f}")
        lines.append("")

        lines.append("PROGRESS:")
        elapsed = time.time() - self._start_time
        lines.append(f"  Tasks attempted:            {cal_summary['total_records']}")
        lines.append(f"  Overall accuracy:           {cal_summary['overall_accuracy']:.1%}")
        lines.append(f"  Version:                    {self._version_manager.current_version}")
        lines.append(f"  Elapsed:                    {elapsed:.0f}s")
        lines.append("")

        lines.append("PATH TO 100%:")
        total_domains = len(scores)
        mastered_count = mastery["mastered"]
        if total_domains > 0:
            pct = mastered_count / total_domains * 100
            lines.append(f"  {mastered_count}/{total_domains} domains mastered ({pct:.0f}%)")
        lines.append("  No hard-coded ceiling. Training continues until all domains mastered.")
        lines.append("=" * 60)

        return "\n".join(lines)

    def export_report(self) -> dict[str, Any]:
        return {
            "snapshot_count": len(self._snapshots),
            "latest": self.render(),
            "version": self._version_manager.current_version,
            "expertise": self._expertise.export_scores(),
            "calibration": self._calibrator.calibration_summary(),
            "mastery": self._expertise.mastery_status(),
        }

    def save_report(self, path: str | Path) -> None:
        report = self.export_report()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
