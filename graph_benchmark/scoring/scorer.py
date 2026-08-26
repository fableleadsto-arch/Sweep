"""
Scorer — Evaluates model outputs against ground truth.

Metrics:
- Exact accuracy (primary)
- Precision, Recall, F1 (set-based tasks)
- Exact-set accuracy
- Path accuracy
- Reachability accuracy
- Error classification
"""
from __future__ import annotations

import math
import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskResult:
    """Result for a single task."""
    task_id: str
    task_type: str
    difficulty: str
    graph_id: str
    ground_truth: Any
    prediction: str
    correct: bool
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    error_type: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryMetrics:
    """Aggregate metrics for a task type or difficulty."""
    name: str
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_f1: float = 0.0
    avg_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    error_breakdown: dict[str, int] = field(default_factory=dict)


@dataclass
class BenchmarkResults:
    """Full benchmark results."""
    system: str
    overall: CategoryMetrics
    by_task_type: dict[str, CategoryMetrics]
    by_difficulty: dict[str, CategoryMetrics]
    by_context_size: dict[str, CategoryMetrics]
    task_results: list[TaskResult]
    environment: dict[str, Any]
    statistics: dict[str, Any]


def _parse_set_answer(answer: str) -> set[str]:
    """Parse a comma-separated node list into a set."""
    answer = answer.strip()
    if answer.upper() == "NONE" or answer == "":
        return set()
    parts = [p.strip() for p in re.split(r'[,;\n]', answer) if p.strip()]
    return set(parts)


def _classify_error(
    task_type: str,
    ground_truth: Any,
    prediction: str,
) -> str:
    """Classify the type of error."""
    if task_type == "bfs":
        gt = set(ground_truth) if isinstance(ground_truth, list) else set()
        pred = _parse_set_answer(prediction)
        missed = gt - pred
        false_pos = pred - gt
        if missed and not false_pos:
            return "missed_edge"
        if false_pos and not missed:
            return "false_edge"
        if missed and false_pos:
            return "incomplete_aggregation"
        return ""

    elif task_type == "reachability":
        gt = ground_truth.upper()
        pred = prediction.strip().upper()
        if gt == "YES" and pred == "NO":
            return "missed_edge"
        if gt == "NO" and pred == "YES":
            return "false_edge"
        return "context_failure"

    elif task_type == "shortest_path":
        return "path_reconstruction_failure"

    elif task_type == "common_descendants":
        gt = set(ground_truth) if isinstance(ground_truth, list) else set()
        pred = _parse_set_answer(prediction)
        missed = gt - pred
        false_pos = pred - gt
        if missed and not false_pos:
            return "incomplete_aggregation"
        if false_pos and not missed:
            return "distractor_contamination"
        if missed and false_pos:
            return "incorrect_branch_integration"
        return ""

    elif task_type == "common_ancestors":
        gt = set(ground_truth) if isinstance(ground_truth, list) else set()
        pred = _parse_set_answer(prediction)
        missed = gt - pred
        if missed:
            return "incomplete_aggregation"
        return "distractor_contamination" if (pred - gt) else ""

    elif task_type == "parent_reconstruction":
        gt = set(ground_truth) if isinstance(ground_truth, list) else set()
        pred = _parse_set_answer(prediction)
        missed = gt - pred
        if missed:
            return "missed_edge"
        return "false_edge" if (pred - gt) else ""

    elif task_type == "multi_hop_chain":
        return "path_reconstruction_failure"

    elif task_type == "contradictory":
        return "distractor_contamination"

    return "output_formatting_failure"


def _compute_set_metrics(predicted: set[str], expected: set[str]) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for set-based tasks."""
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    if not predicted or not expected:
        return 0.0, 0.0, 0.0
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _exact_match(ground_truth: Any, prediction: str) -> bool:
    """Check if prediction matches ground truth."""
    gt_str = str(ground_truth).strip().upper()
    pred_str = prediction.strip().upper()

    if gt_str == pred_str:
        return True

    # Set-based comparison
    if isinstance(ground_truth, list):
        gt_set = set(ground_truth)
        pred_set = _parse_set_answer(prediction)
        return gt_set == pred_set

    # Numeric comparison
    try:
        if float(gt_str) == float(pred_str):
            return True
    except (ValueError, TypeError):
        pass

    return False


class BenchmarkScorer:
    """
    Scores benchmark results and generates reports.

    Usage:
        scorer = BenchmarkScorer()
        scorer.add_result(task_result)
        results = scorer.compute_results()
    """

    def __init__(self) -> None:
        self._results: list[TaskResult] = []

    def score_task(
        self,
        task_id: str,
        task_type: str,
        difficulty: str,
        graph_id: str,
        ground_truth: Any,
        prediction: str,
        latency_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> TaskResult:
        """Score a single task."""
        correct = _exact_match(ground_truth, prediction)

        precision, recall, f1 = 0.0, 0.0, 0.0
        if isinstance(ground_truth, list):
            expected_set = set(ground_truth)
            predicted_set = _parse_set_answer(prediction)
            precision, recall, f1 = _compute_set_metrics(predicted_set, expected_set)

        error_type = "" if correct else _classify_error(task_type, ground_truth, prediction)

        result = TaskResult(
            task_id=task_id,
            task_type=task_type,
            difficulty=difficulty,
            graph_id=graph_id,
            ground_truth=ground_truth,
            prediction=prediction,
            correct=correct,
            precision=precision,
            recall=recall,
            f1=f1,
            error_type=error_type,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )
        self._results.append(result)
        return result

    def add_result(self, result: TaskResult) -> None:
        """Add a pre-computed result."""
        self._results.append(result)

    def _compute_category(self, name: str, tasks: list[TaskResult]) -> CategoryMetrics:
        """Compute aggregate metrics for a category."""
        if not tasks:
            return CategoryMetrics(name=name)

        total = len(tasks)
        correct = sum(1 for t in tasks if t.correct)
        latencies = [t.latency_ms for t in tasks if t.latency_ms > 0]

        precisions = [t.precision for t in tasks if isinstance(t.ground_truth, list)]
        recalls = [t.recall for t in tasks if isinstance(t.ground_truth, list)]
        f1s = [t.f1 for t in tasks if isinstance(t.ground_truth, list)]

        error_breakdown: dict[str, int] = {}
        for t in tasks:
            if t.error_type:
                error_breakdown[t.error_type] = error_breakdown.get(t.error_type, 0) + 1

        return CategoryMetrics(
            name=name,
            total=total,
            correct=correct,
            accuracy=correct / total,
            avg_precision=statistics.mean(precisions) if precisions else 0.0,
            avg_recall=statistics.mean(recalls) if recalls else 0.0,
            avg_f1=statistics.mean(f1s) if f1s else 0.0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            median_latency_ms=statistics.median(latencies) if latencies else 0.0,
            p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else 0.0,
            error_breakdown=error_breakdown,
        )

    def compute_results(
        self,
        system_name: str = "sweep",
        environment: dict[str, Any] | None = None,
    ) -> BenchmarkResults:
        """Compute all aggregate metrics."""
        overall = self._compute_category("overall", self._results)

        by_type: dict[str, list[TaskResult]] = {}
        for r in self._results:
            by_type.setdefault(r.task_type, []).append(r)
        by_task_type = {k: self._compute_category(k, v) for k, v in by_type.items()}

        by_diff_raw: dict[str, list[TaskResult]] = {}
        for r in self._results:
            by_diff_raw.setdefault(r.difficulty, []).append(r)
        by_difficulty = {k: self._compute_category(k, v) for k, v in by_diff_raw.items()}

        # Compute confidence intervals
        latencies = [r.latency_ms for r in self._results if r.latency_ms > 0]
        accuracies_by_graph: dict[str, list[int]] = {}
        for r in self._results:
            accuracies_by_graph.setdefault(r.graph_id, []).append(1 if r.correct else 0)
        graph_accs = [sum(v) / len(v) for v in accuracies_by_graph.values() if v]

        stats = {
            "mean_latency_ms": statistics.mean(latencies) if latencies else 0.0,
            "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
            "std_latency_ms": statistics.stdev(latencies) if len(latencies) >= 2 else 0.0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else 0.0,
        }

        # 95% CI for accuracy
        n = len(self._results)
        p = overall.accuracy
        if n > 0:
            se = math.sqrt(p * (1 - p) / n)
            stats["accuracy_ci_lower"] = max(0.0, p - 1.96 * se)
            stats["accuracy_ci_upper"] = min(1.0, p + 1.96 * se)
        else:
            stats["accuracy_ci_lower"] = 0.0
            stats["accuracy_ci_upper"] = 0.0

        # Error analysis summary
        error_counts: dict[str, int] = {}
        for r in self._results:
            if r.error_type:
                error_counts[r.error_type] = error_counts.get(r.error_type, 0) + 1
        stats["error_analysis"] = error_counts
        stats["total_errors"] = sum(error_counts.values())

        # Source analysis: local processing vs logic gathering
        local_errors = 0
        logic_errors = 0
        for r in self._results:
            if r.error_type in ("missed_edge", "false_edge", "distractor_contamination", "context_failure"):
                local_errors += 1
            elif r.error_type in ("incomplete_aggregation", "incorrect_branch_integration"):
                logic_errors += 1
        stats["local_processing_errors"] = local_errors
        stats["logic_gathering_errors"] = logic_errors

        return BenchmarkResults(
            system=system_name,
            overall=overall,
            by_task_type=by_task_type,
            by_difficulty=by_difficulty,
            by_context_size={},  # filled by runner
            task_results=self._results,
            environment=environment or {},
            statistics=stats,
        )

    def to_json(self, results: BenchmarkResults, path: str) -> None:
        """Save results to JSON."""

        def _cat_dict(c: CategoryMetrics) -> dict:
            return {
                "name": c.name, "total": c.total, "correct": c.correct,
                "accuracy": round(c.accuracy, 4),
                "avg_precision": round(c.avg_precision, 4),
                "avg_recall": round(c.avg_recall, 4),
                "avg_f1": round(c.avg_f1, 4),
                "avg_latency_ms": round(c.avg_latency_ms, 3),
                "median_latency_ms": round(c.median_latency_ms, 3),
                "p95_latency_ms": round(c.p95_latency_ms, 3),
                "error_breakdown": c.error_breakdown,
            }

        data = {
            "system": results.system,
            "overall": _cat_dict(results.overall),
            "by_task_type": {k: _cat_dict(v) for k, v in results.by_task_type.items()},
            "by_difficulty": {k: _cat_dict(v) for k, v in results.by_difficulty.items()},
            "by_context_size": {k: _cat_dict(v) for k, v in results.by_context_size.items()},
            "statistics": results.statistics,
            "environment": results.environment,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def generate_report(self, results: BenchmarkResults) -> str:
        """Generate human-readable report."""
        lines = []
        lines.append("=" * 72)
        lines.append("  SWEEP GRAPH REASONING BENCHMARK — FINAL REPORT")
        lines.append("=" * 72)
        lines.append("")
        lines.append(f"System: {results.system}")
        lines.append(f"Total tasks: {results.overall.total}")
        lines.append(f"Overall accuracy: {results.overall.accuracy:.1%}")
        ci_lo = results.statistics.get("accuracy_ci_lower", 0)
        ci_hi = results.statistics.get("accuracy_ci_upper", 0)
        lines.append(f"95% CI: [{ci_lo:.1%}, {ci_hi:.1%}]")
        lines.append("")

        # Per task type
        lines.append("─" * 72)
        lines.append("  BY TASK TYPE")
        lines.append("─" * 72)
        lines.append(f"  {'Task Type':<25s} {'Acc':>7s} {'P':>7s} {'R':>7s} {'F1':>7s} {'Lat':>8s}")
        lines.append("  " + "─" * 60)
        for name, cat in sorted(results.by_task_type.items()):
            lines.append(
                f"  {name:<25s} {cat.accuracy:>6.1%} {cat.avg_precision:>6.3f} "
                f"{cat.avg_recall:>6.3f} {cat.avg_f1:>6.3f} {cat.median_latency_ms:>7.1f}ms"
            )
        lines.append("")

        # Per difficulty
        lines.append("─" * 72)
        lines.append("  BY DIFFICULTY")
        lines.append("─" * 72)
        lines.append(f"  {'Difficulty':<25s} {'Acc':>7s} {'Tasks':>7s} {'Errors':>7s}")
        lines.append("  " + "─" * 45)
        for name, cat in sorted(results.by_difficulty.items()):
            lines.append(
                f"  {name:<25s} {cat.accuracy:>6.1%} {cat.total:>7d} {cat.total - cat.correct:>7d}"
            )
        lines.append("")

        # Efficiency
        lines.append("─" * 72)
        lines.append("  EFFICIENCY")
        lines.append("─" * 72)
        lines.append(f"  Median latency:  {results.statistics.get('median_latency_ms', 0):.1f} ms")
        lines.append(f"  Mean latency:    {results.statistics.get('mean_latency_ms', 0):.1f} ms")
        lines.append(f"  p95 latency:     {results.statistics.get('p95_latency_ms', 0):.1f} ms")
        lines.append(f"  Std latency:     {results.statistics.get('std_latency_ms', 0):.1f} ms")
        lines.append("")

        # Error analysis
        lines.append("─" * 72)
        lines.append("  ERROR ANALYSIS")
        lines.append("─" * 72)
        ea = results.statistics.get("error_analysis", {})
        total_errors = results.statistics.get("total_errors", 0)
        lines.append(f"  Total errors: {total_errors}")
        for err_type, count in sorted(ea.items(), key=lambda x: -x[1]):
            lines.append(f"    {err_type:<35s} {count:>5d}")
        lines.append("")
        lines.append(f"  Local processing errors: {results.statistics.get('local_processing_errors', 0)}")
        lines.append(f"  Logic gathering errors:  {results.statistics.get('logic_gathering_errors', 0)}")
        lines.append("")
        lines.append("=" * 72)

        return "\n".join(lines)
