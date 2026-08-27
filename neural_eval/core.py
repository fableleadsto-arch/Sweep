"""
Neural Evaluation Benchmark — Sweep vs OpenAI o1 Reference.

Controlled evaluation of Sweep's neural mesh architecture under
scientifically rigorous conditions. No deterministic solvers in
the inference path.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════

OPENAI_O1_GRAPHWALKS_BFS = 62.0
OPENAI_O1_GRAPHWALKS_PARENTS = 50.9

TRAIN_SEED = 42
VAL_SEED = 137
TEST_SEED = 2026
GENERALIZATION_SEED = 9999


@dataclass
class Task:
    task_id: str
    domain: str
    difficulty: int
    input_text: str
    expected_output: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    task_id: str
    predicted: str
    expected: str
    correct: bool
    confidence: float
    latency_ms: float
    reasoning_steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    name: str
    tasks: list[Task] = field(default_factory=list)
    results: list[Result] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correct) / len(self.results) * 100

    @property
    def mean_latency(self) -> float:
        if not self.results:
            return 0.0
        return np.mean([r.latency_ms for r in self.results])

    @property
    def median_latency(self) -> float:
        if not self.results:
            return 0.0
        return float(np.median([r.latency_ms for r in self.results]))

    @property
    def std_latency(self) -> float:
        if not self.results:
            return 0.0
        return float(np.std([r.latency_ms for r in self.results]))

    @property
    def mean_confidence(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.confidence for r in self.results]))

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tasks": len(self.tasks),
            "accuracy_pct": round(self.accuracy, 2),
            "mean_latency_ms": round(self.mean_latency, 2),
            "median_latency_ms": round(self.median_latency, 2),
            "std_latency_ms": round(self.std_latency, 2),
            "mean_confidence": round(self.mean_confidence, 4),
        }
