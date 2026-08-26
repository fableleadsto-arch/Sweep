"""
Model Versioning — Every improvement creates a version.

§26: Record architecture version, training dataset version, benchmark results.
Never overwrite the previous model.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelVersion:
    """A snapshot of the model state."""
    version_id: str
    iteration: int
    timestamp: float
    training_tasks: int
    verified_experiences: int
    benchmark_accuracy: float
    domain_scores: dict[str, float]
    regression_passed: bool
    changes: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VersionManager:
    """
    Manages model versions.

    §26: Never overwrite the previous model.
    §27: Before/after evaluation.
    """

    def __init__(self, storage_dir: str | Path = "sweep_neural_mesh/training/versions") -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._versions: list[ModelVersion] = []
        self._current_version = "v0.0"
        self._version_counter = 0

    def create_version(
        self,
        iteration: int,
        training_tasks: int,
        verified_experiences: int,
        benchmark_accuracy: float,
        domain_scores: dict[str, float],
        regression_passed: bool,
        changes: str = "",
    ) -> ModelVersion:
        """Create a new version snapshot."""
        self._version_counter += 1
        self._current_version = f"v0.{self._version_counter}"

        version = ModelVersion(
            version_id=self._current_version,
            iteration=iteration,
            timestamp=time.time(),
            training_tasks=training_tasks,
            verified_experiences=verified_experiences,
            benchmark_accuracy=benchmark_accuracy,
            domain_scores=domain_scores,
            regression_passed=regression_passed,
            changes=changes,
        )
        self._versions.append(version)
        self._save_version(version)
        return version

    def get_previous_accuracy(self) -> float:
        if len(self._versions) >= 2:
            return self._versions[-2].benchmark_accuracy
        return 0.0

    @property
    def current_version(self) -> str:
        return self._current_version

    @property
    def version_history(self) -> list[ModelVersion]:
        return self._versions.copy()

    def _save_version(self, version: ModelVersion) -> None:
        path = self._storage_dir / f"{version.version_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "version_id": version.version_id,
                "iteration": version.iteration,
                "timestamp": version.timestamp,
                "training_tasks": version.training_tasks,
                "verified_experiences": version.verified_experiences,
                "benchmark_accuracy": version.benchmark_accuracy,
                "domain_scores": version.domain_scores,
                "regression_passed": version.regression_passed,
                "changes": version.changes,
            }, f, indent=2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_version": self._current_version,
            "total_versions": len(self._versions),
            "history": [
                {"version": v.version_id, "accuracy": v.benchmark_accuracy,
                 "iteration": v.iteration, "regression": v.regression_passed}
                for v in self._versions
            ],
        }
