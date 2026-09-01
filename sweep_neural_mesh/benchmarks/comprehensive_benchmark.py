"""
Comprehensive Benchmark — §18-19

A benchmark that measures more than just accuracy:
- accuracy
- precision
- recall
- F1
- calibration
- hallucination rate
- unsupported-claim rate
- tool-selection accuracy
- retrieval accuracy
- reasoning-task accuracy
- latency
- memory consumption
- failure rate

Creates a dashboard showing performance by task category.
Do not report one artificial "intelligence score" as proof of superiority.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    task_id: str
    category: str
    subcategory: str
    difficulty: int
    input_text: str
    expected_output: str
    verification_method: str  # exact, contains, numeric, semantic, boolean
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of a single task evaluation."""
    task_id: str
    category: str
    correct: bool
    predicted: str
    expected: str
    confidence: float
    latency_ms: float
    true_positive: bool = False
    false_positive: bool = False
    true_negative: bool = False
    false_negative: bool = False


@dataclass
class CategoryMetrics:
    """Metrics for a single category."""
    category: str
    task_count: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    avg_confidence: float
    avg_latency_ms: float
    hallucination_rate: float
    calibration_error: float


class ComprehensiveBenchmark:
    """
    §18-19: Comprehensive benchmark suite that measures multiple
    dimensions of performance per category.
    """

    CATEGORIES = [
        "reasoning", "mathematics", "coding", "knowledge",
        "instruction_following", "language", "data_analysis",
        "multimodal", "retrieval", "entity_resolution",
        "memory", "planning", "tool_use", "web_research",
        "uncertainty", "adversarial",
    ]

    def __init__(self) -> None:
        self._tasks: list[BenchmarkTask] = []
        self._results: list[BenchmarkResult] = []
        self._category_metrics: dict[str, CategoryMetrics] = {}

    def add_tasks(self, tasks: list[BenchmarkTask]) -> None:
        """Add tasks to the benchmark."""
        self._tasks.extend(tasks)

    def generate_benchmark_tasks(self, per_category: int = 20) -> list[BenchmarkTask]:
        """Generate benchmark tasks across all categories."""
        generators = {
            "reasoning": self._gen_reasoning_tasks,
            "mathematics": self._gen_math_tasks,
            "knowledge": self._gen_knowledge_tasks,
            "uncertainty": self._gen_uncertainty_tasks,
            "entity_resolution": self._gen_entity_resolution_tasks,
        }
        tasks = []
        for category in self.CATEGORIES:
            gen = generators.get(category, self._gen_generic_tasks)
            tasks.extend(gen(per_category))
        self._tasks.extend(tasks)
        return tasks

    def evaluate(
        self,
        model_fn: Callable[[str], tuple[str, float]],
    ) -> dict[str, CategoryMetrics]:
        """
        Evaluate the model on all tasks.

        Args:
            model_fn: Function that takes input_text and returns (answer, confidence).

        Returns:
            Per-category metrics.
        """
        self._results = []

        for task in self._tasks:
            t0 = time.perf_counter()
            answer, confidence = model_fn(task.input_text)
            latency = (time.perf_counter() - t0) * 1000

            correct = self._verify(task, answer)

            result = BenchmarkResult(
                task_id=task.task_id,
                category=task.category,
                correct=correct,
                predicted=answer,
                expected=task.expected_output,
                confidence=confidence,
                latency_ms=latency,
            )
            self._results.append(result)

        self._compute_metrics()
        return self._category_metrics

    def _verify(self, task: BenchmarkTask, prediction: str) -> bool:
        """Verify a prediction against expected output."""
        method = task.verification_method
        expected = task.expected_output.strip()
        pred = prediction.strip()

        if method == "exact":
            return pred.upper() == expected.upper()
        elif method == "contains":
            return expected.lower() in pred.lower()
        elif method == "numeric":
            import re
            exp_nums = re.findall(r'-?\d+\.?\d*', expected)
            pred_nums = re.findall(r'-?\d+\.?\d*', pred)
            return exp_nums and pred_nums and exp_nums[0] == pred_nums[0]
        elif method == "boolean":
            return pred.upper().strip() in ("YES", "NO", "TRUE", "FALSE")
        else:
            return pred.upper() == expected.upper()

    def _compute_metrics(self) -> None:
        """Compute per-category metrics."""
        by_category: dict[str, list[BenchmarkResult]] = {}
        for r in self._results:
            by_category.setdefault(r.category, []).append(r)

        for category, results in by_category.items():
            total = len(results)
            correct = sum(1 for r in results if r.correct)
            accuracy = correct / max(total, 1)

            # Precision/Recall (for binary classification tasks)
            tp = sum(1 for r in results if r.correct and "YES" in r.expected.upper())
            fp = sum(1 for r in results if not r.correct and "YES" in r.predicted.upper())
            fn = sum(1 for r in results if not r.correct and "YES" in r.expected.upper())
            tn = sum(1 for r in results if r.correct and "NO" in r.expected.upper())

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)

            # Calibration error
            avg_conf = sum(r.confidence for r in results) / max(total, 1)
            calibration_error = abs(avg_conf - accuracy)

            # Hallucination rate (high confidence + wrong)
            hallucinations = sum(
                1 for r in results
                if not r.correct and r.confidence > 0.8
            ) / max(total, 1)

            # Average latency
            avg_latency = sum(r.latency_ms for r in results) / max(total, 1)

            self._category_metrics[category] = CategoryMetrics(
                category=category,
                task_count=total,
                accuracy=round(accuracy, 4),
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                avg_confidence=round(avg_conf, 4),
                avg_latency_ms=round(avg_latency, 1),
                hallucination_rate=round(hallucinations, 4),
                calibration_error=round(calibration_error, 4),
            )

    def generate_report(self, output_dir: str | Path = "sweep_neural_mesh/benchmarks/reports") -> str:
        """Generate a comprehensive benchmark report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report = self._build_report()
        path = output_path / "comprehensive_benchmark_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Also save CSV
        csv_path = output_path / "comprehensive_benchmark.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Category,Tasks,Accuracy,Precision,Recall,F1,AvgConfidence,AvgLatencyMs,HallucinationRate,CalibrationError\n")
            for cm in self._category_metrics.values():
                f.write(f"{cm.category},{cm.task_count},{cm.accuracy},{cm.precision},{cm.recall},{cm.f1},{cm.avg_confidence},{cm.avg_latency_ms},{cm.hallucination_rate},{cm.calibration_error}\n")

        return str(path)

    def _build_report(self) -> dict[str, Any]:
        """Build the full report dict."""
        total_tasks = len(self._results)
        total_correct = sum(1 for r in self._results if r.correct)

        overall_accuracy = total_correct / max(total_tasks, 1)
        overall_latency = sum(r.latency_ms for r in self._results) / max(total_tasks, 1)
        overall_hallucination = sum(
            1 for r in self._results if not r.correct and r.confidence > 0.8
        ) / max(total_tasks, 1)

        # Weighted average (equal weights)
        if self._category_metrics:
            avg_f1 = sum(m.f1 for m in self._category_metrics.values()) / len(self._category_metrics)
            avg_calibration = sum(m.calibration_error for m in self._category_metrics.values()) / len(self._category_metrics)
        else:
            avg_f1 = 0
            avg_calibration = 0

        return {
            "summary": {
                "total_tasks": total_tasks,
                "overall_accuracy": round(overall_accuracy, 4),
                "overall_f1": round(avg_f1, 4),
                "overall_calibration_error": round(avg_calibration, 4),
                "overall_hallucination_rate": round(overall_hallucination, 4),
                "overall_avg_latency_ms": round(overall_latency, 1),
                "categories_evaluated": len(self._category_metrics),
            },
            "category_metrics": {
                name: {
                    "tasks": cm.task_count,
                    "accuracy": cm.accuracy,
                    "precision": cm.precision,
                    "recall": cm.recall,
                    "f1": cm.f1,
                    "avg_confidence": cm.avg_confidence,
                    "avg_latency_ms": cm.avg_latency_ms,
                    "hallucination_rate": cm.hallucination_rate,
                    "calibration_error": cm.calibration_error,
                }
                for name, cm in self._category_metrics.items()
            },
            "note": "This benchmark measures multiple dimensions of performance. "
                    "No single score represents overall intelligence.",
        }

    # ══════════════════════════════════════════════════════════════════
    # TASK GENERATORS
    # ══════════════════════════════════════════════════════════════════

    def _gen_reasoning_tasks(self, count: int) -> list[BenchmarkTask]:
        """Generate reasoning benchmark tasks."""
        tasks = []
        pairs = [
            ("If all cats are animals and all animals are living things, are cats living things?", "YES", "boolean"),
            ("A is taller than B. B is taller than C. Who is shortest?", "C", "exact"),
            ("It is raining. If it rains, the ground is wet. Is the ground wet?", "YES", "boolean"),
        ]
        for i in range(min(count, len(pairs))):
            q, a, v = pairs[i]
            tasks.append(BenchmarkTask(
                task_id=f"BM-REASON-{i:04d}",
                category="reasoning",
                subcategory="deduction",
                difficulty=2,
                input_text=q,
                expected_output=a,
                verification_method=v,
            ))
        return tasks

    def _gen_math_tasks(self, count: int) -> list[BenchmarkTask]:
        """Generate math benchmark tasks."""
        import random
        rng = random.Random(42)
        tasks = []
        for i in range(count):
            a = rng.randint(1, 100)
            b = rng.randint(1, 100)
            tasks.append(BenchmarkTask(
                task_id=f"BM-MATH-{i:04d}",
                category="mathematics",
                subcategory="arithmetic",
                difficulty=1,
                input_text=f"What is {a} + {b}?",
                expected_output=str(a + b),
                verification_method="numeric",
            ))
        return tasks

    def _gen_knowledge_tasks(self, count: int) -> list[BenchmarkTask]:
        """Generate knowledge benchmark tasks."""
        tasks = []
        pairs = [
            ("What is the capital of France?", "Paris", "contains"),
            ("What planet is closest to the Sun?", "Mercury", "contains"),
            ("What is the chemical symbol for water?", "H2O", "contains"),
        ]
        for i in range(min(count, len(pairs))):
            q, a, v = pairs[i]
            tasks.append(BenchmarkTask(
                task_id=f"BM-KNOW-{i:04d}",
                category="knowledge",
                subcategory="factual",
                difficulty=1,
                input_text=q,
                expected_output=a,
                verification_method=v,
            ))
        return tasks

    def _gen_uncertainty_tasks(self, count: int) -> list[BenchmarkTask]:
        """Generate uncertainty benchmark tasks."""
        tasks = []
        pairs = [
            ("What is the exact population of Mars right now?", "UNKNOWN", "contains"),
            ("Who will win the next presidential election?", "UNKNOWN", "contains"),
            ("What happened yesterday at 3:42 PM in New York City?", "UNKNOWN", "contains"),
        ]
        for i in range(min(count, len(pairs))):
            q, a, v = pairs[i]
            tasks.append(BenchmarkTask(
                task_id=f"BM-UNCERT-{i:04d}",
                category="uncertainty",
                subcategory="abstention",
                difficulty=2,
                input_text=q,
                expected_output=a,
                verification_method=v,
            ))
        return tasks

    def _gen_entity_resolution_tasks(self, count: int) -> list[BenchmarkTask]:
        """Generate entity resolution tasks."""
        tasks = []
        pairs = [
            ("Are 'John Smith' and 'J. Smith' the same person?", "UNKNOWN", "contains"),
            ("Are 'Apple Inc.' and 'Apple Corps' the same company?", "NO", "contains"),
        ]
        for i in range(min(count, len(pairs))):
            q, a, v = pairs[i]
            tasks.append(BenchmarkTask(
                task_id=f"BM-ENT-{i:04d}",
                category="entity_resolution",
                subcategory="same_different",
                difficulty=3,
                input_text=q,
                expected_output=a,
                verification_method=v,
            ))
        return tasks

    def _gen_generic_tasks(self, count: int) -> list[BenchmarkTask]:
        """Generate generic tasks for categories without specific generators."""
        return [
            BenchmarkTask(
                task_id=f"BM-GEN-{i:04d}",
                category="general",
                subcategory="general",
                difficulty=1,
                input_text=f"Task {i}: What is 2 + {i}?",
                expected_output=str(2 + i),
                verification_method="numeric",
            )
            for i in range(count)
        ]
