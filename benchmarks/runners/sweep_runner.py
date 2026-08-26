"""
Sweep Benchmark Runner — runs ReasoningCortex on test cases.

Measures:
- Accuracy (decision match + answer match)
- Latency (per-case and aggregate)
- Throughput (cases per second)
- Memory usage
"""
from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sweep_neural_mesh.neurons.cortex import ReasoningCortex
from benchmarks.dataset.generate import TestCase


class SweepRunner:
    """
    Runs Sweep's ReasoningCortex on benchmark test cases.

    Collects per-case results and aggregate statistics.
    """

    def __init__(self, enable_ml: bool = False) -> None:
        self._cortex = ReasoningCortex(enable_ml=enable_ml)
        self._results: list[dict[str, Any]] = []

    def run_single(self, case: TestCase) -> dict[str, Any]:
        """Run a single test case and return the result."""
        t0 = time.perf_counter()
        result = self._cortex.reason(
            query=case.query,
            evidence=case.evidence,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "id": case.id,
            "category": case.category,
            "difficulty": case.difficulty,
            "query": case.query,
            "expected_decision": case.expected_decision,
            "expected_answer": case.expected_answer,
            "actual_decision": result.decision,
            "actual_confidence": result.confidence,
            "actual_reasoning": result.reasoning,
            "latency_ms": round(latency_ms, 3),
            "decision_correct": result.decision == case.expected_decision,
        }

    def run_all(self, cases: list[TestCase], verbose: bool = True) -> dict[str, Any]:
        """Run all test cases and return aggregate results."""
        self._results = []
        tracemalloc.start()
        t_start = time.perf_counter()

        for i, case in enumerate(cases):
            result = self.run_single(case)
            self._results.append(result)
            if verbose and (i + 1) % 100 == 0:
                print(f"  Sweep: {i+1}/{len(cases)} cases completed")

        t_total = time.perf_counter() - t_start
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Aggregate stats
        total = len(self._results)
        correct_decisions = sum(1 for r in self._results if r["decision_correct"])
        latencies = [r["latency_ms"] for r in self._results]

        # Per-category stats
        category_stats: dict[str, dict[str, Any]] = {}
        for cat in set(r["category"] for r in self._results):
            cat_results = [r for r in self._results if r["category"] == cat]
            cat_correct = sum(1 for r in cat_results if r["decision_correct"])
            cat_latencies = [r["latency_ms"] for r in cat_results]
            category_stats[cat] = {
                "total": len(cat_results),
                "correct": cat_correct,
                "accuracy": cat_correct / len(cat_results) if cat_results else 0.0,
                "avg_latency_ms": round(sum(cat_latencies) / len(cat_latencies), 3) if cat_latencies else 0.0,
                "p50_latency_ms": round(sorted(cat_latencies)[len(cat_latencies) // 2], 3) if cat_latencies else 0.0,
                "p95_latency_ms": round(sorted(cat_latencies)[int(len(cat_latencies) * 0.95)], 3) if cat_latencies else 0.0,
            }

        # Per-difficulty stats
        difficulty_stats: dict[str, dict[str, Any]] = {}
        for diff in set(r["difficulty"] for r in self._results):
            diff_results = [r for r in self._results if r["difficulty"] == diff]
            diff_correct = sum(1 for r in diff_results if r["decision_correct"])
            difficulty_stats[diff] = {
                "total": len(diff_results),
                "correct": diff_correct,
                "accuracy": diff_correct / len(diff_results) if diff_results else 0.0,
            }

        summary = {
            "system": "sweep",
            "total_cases": total,
            "correct_decisions": correct_decisions,
            "accuracy": correct_decisions / total if total > 0 else 0.0,
            "total_latency_ms": round(t_total * 1000, 3),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 3) if latencies else 0.0,
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 3) if latencies else 0.0,
            "p99_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 3) if latencies else 0.0,
            "throughput_per_sec": round(total / t_total, 1) if t_total > 0 else 0.0,
            "peak_memory_mb": round(peak_memory / (1024 * 1024), 2),
            "by_category": category_stats,
            "by_difficulty": difficulty_stats,
        }

        return {
            "summary": summary,
            "results": self._results,
        }

    def save(self, data: dict[str, Any], path: str | Path) -> None:
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
