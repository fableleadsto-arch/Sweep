"""
Sweep Runner — Runs Sweep's GraphReasoningEngine on graph reasoning tasks.

Measures accuracy, latency, memory, and throughput.
Supports ablation variants with reduced cores, neurons, or connectivity.
"""
from __future__ import annotations

import json
import time
import tracemalloc
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graph_benchmark.runners.graph_engine import GraphReasoningEngine, GraphAnswer
from graph_benchmark.generator.task_generator import Task
from graph_benchmark.scoring.scorer import BenchmarkScorer, TaskResult


class SweepGraphRunner:
    """
    Runs Sweep on graph reasoning tasks.

    Uses GraphReasoningEngine (graph algorithms + neural mesh) instead of
    raw ReasoningCortex (which only does claim-evidence evaluation).

    Args:
        enable_ml: Whether to enable ML engines.
        variant: Ablation variant name (None = full).
        mesh_fraction: Fraction of mesh capacity (0.0 to 1.0).
    """

    def __init__(
        self,
        enable_ml: bool = False,
        variant: str | None = None,
        mesh_fraction: float = 1.0,
    ) -> None:
        self._variant = variant or "full"
        self._mesh_fraction = mesh_fraction
        self._results: list[TaskResult] = []
        use_neural = variant not in ("no_logic_gatherer", "simplified_logic")
        self._engine = GraphReasoningEngine(use_neural_mesh=use_neural)

    def run_single(self, task: Task) -> TaskResult:
        """Run a single task and return the result."""
        t0 = time.perf_counter()
        answer = self._engine.solve(prompt=task.prompt, graph_text=task.graph_text)
        latency_ms = (time.perf_counter() - t0) * 1000

        prediction = self._format_prediction(answer, task)

        scorer = BenchmarkScorer()
        task_result = scorer.score_task(
            task_id=task.id,
            task_type=task.task_type,
            difficulty=task.difficulty,
            graph_id=task.graph_id,
            ground_truth=task.ground_truth,
            prediction=prediction,
            latency_ms=latency_ms,
            metadata={
                "variant": self._variant,
                "mesh_fraction": self._mesh_fraction,
                "engine_confidence": answer.confidence,
                "engine_reasoning": answer.reasoning,
                "engine_method": answer.method,
            },
        )
        return task_result

    def _format_prediction(self, answer: GraphAnswer, task: Task) -> str:
        """Format GraphAnswer into a string for scoring."""
        task_type = task.task_type

        if task_type == "bfs":
            if isinstance(answer.value, list):
                if not answer.value:
                    return "NONE"
                return ", ".join(answer.value)
            return str(answer.value)

        if task_type == "reachability":
            return str(answer.value).upper()

        if task_type == "shortest_path":
            return str(answer.value)

        if task_type == "common_descendants":
            if isinstance(answer.value, list):
                if not answer.value:
                    return "NONE"
                return ", ".join(answer.value)
            return str(answer.value)

        if task_type == "common_ancestors":
            if isinstance(answer.value, list):
                if not answer.value:
                    return "NONE"
                return ", ".join(answer.value)
            return str(answer.value)

        if task_type == "parent_reconstruction":
            if isinstance(answer.value, list):
                if not answer.value:
                    return "NONE"
                return ", ".join(answer.value)
            return str(answer.value)

        if task_type == "multi_hop_chain":
            return str(answer.value).upper()

        if task_type == "contradictory":
            return str(answer.value).upper()

        if task_type == "parallel_branches":
            if isinstance(answer.value, list):
                if not answer.value:
                    return "NONE"
                return ", ".join(answer.value)
            return str(answer.value)

        return str(answer.value)

    def run_all(self, tasks: list[Task], verbose: bool = True) -> dict[str, Any]:
        """Run all tasks and return results."""
        self._results = []
        tracemalloc.start()
        t_start = time.perf_counter()

        for i, task in enumerate(tasks):
            result = self.run_single(task)
            self._results.append(result)
            if verbose and (i + 1) % 50 == 0:
                print(f"  Sweep [{self._variant}]: {i+1}/{len(tasks)} tasks completed")

        t_total = time.perf_counter() - t_start
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        scorer = BenchmarkScorer()
        for r in self._results:
            scorer.add_result(r)

        total = len(self._results)
        correct = sum(1 for r in self._results if r.correct)

        return {
            "system": f"sweep_{self._variant}",
            "variant": self._variant,
            "mesh_fraction": self._mesh_fraction,
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "total_time_s": round(t_total, 2),
            "avg_latency_ms": round(sum(r.latency_ms for r in self._results) / max(1, total), 3),
            "peak_memory_mb": round(peak_memory / (1024 * 1024), 2),
            "results": self._results,
        }
