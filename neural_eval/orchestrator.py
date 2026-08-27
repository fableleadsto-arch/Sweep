"""
Neural Evaluation Benchmark — Main Orchestrator.

Runs all test suites, collects results, generates REPORT.md.
"""
from __future__ import annotations

import json
import sys
import io
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from .core import Task, Result, BenchmarkSuite
from .generators import (
    generate_all_tasks, generate_branching_tasks,
    ALL_GENERATORS,
)
from .generators_extended import (
    gen_distractor_tasks, gen_conflict_tasks, gen_novel_topology_tasks,
    ABLATION_CONFIGS,
)
from .runners.sweep_runner import SweepRunner
from .scoring.stats import compute_stats, paired_accuracy_test
from .environment.detector import generate_environment_json


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "neural_eval" / "results"
DATASETS_DIR = ROOT / "neural_eval" / "datasets"


def run_full_benchmark() -> dict[str, Any]:
    """Execute the complete benchmark suite."""
    print("=" * 60)
    print("SWEEP NEURAL EVALUATION BENCHMARK")
    print("=" * 60)

    env_path = generate_environment_json(RESULTS_DIR.parent / "environment")
    print(f"Environment recorded: {env_path}")

    runner = SweepRunner()
    all_results: dict[str, Any] = {}

    # ── Section 1: Pure Neural Reasoning ─────────────────────────────
    print("\n[1/8] Pure Neural Reasoning (all domains)...")
    pure_neural = BenchmarkSuite(name="pure_neural_reasoning")
    for domain_name, gen_fn in ALL_GENERATORS.items():
        tasks = gen_fn(seed=42, difficulty=3)
        results = runner.run_batch(tasks)
        pure_neural.tasks.extend(tasks)
        pure_neural.results.extend(results)
        domain_correct = sum(1 for r in results if r.correct)
        print(f"  {domain_name}: {domain_correct}/{len(results)}")
    all_results["pure_neural"] = pure_neural.summary()
    all_results["pure_neural_stats"] = compute_stats(pure_neural.results)

    # ── Section 2: Difficulty Scaling ────────────────────────────────
    print("\n[2/8] Difficulty Scaling...")
    scaling_results = {}
    for level in range(1, 7):
        tasks = generate_all_tasks(seed=42, difficulty=level)
        results = runner.run_batch(tasks)
        correct = sum(1 for r in results if r.correct)
        acc = correct / len(results) * 100 if results else 0
        scaling_results[f"level_{level}"] = {
            "tasks": len(results),
            "accuracy_pct": round(acc, 2),
        }
        print(f"  Level {level}: {correct}/{len(results)} = {acc:.1f}%")
    all_results["difficulty_scaling"] = scaling_results

    # ── Section 3: Parallel Branch Integration ───────────────────────
    print("\n[3/8] Parallel Branch Integration...")
    branch_counts = [2, 4, 8, 16, 32, 64]
    branching = generate_branching_tasks(seed=42, branch_counts=branch_counts)
    branch_results = {}
    for n_branches, tasks in branching.items():
        results = runner.run_batch(tasks)
        correct = sum(1 for r in results if r.correct)
        acc = correct / len(results) * 100 if results else 0
        mean_lat = sum(r.latency_ms for r in results) / len(results) if results else 0
        branch_results[f"branches_{n_branches}"] = {
            "tasks": len(results),
            "accuracy_pct": round(acc, 2),
            "mean_latency_ms": round(mean_lat, 2),
        }
        print(f"  {n_branches} branches: {correct}/{len(results)} = {acc:.1f}%")
    all_results["parallel_branches"] = branch_results

    # ── Section 4: Distractor Resistance ─────────────────────────────
    print("\n[4/8] Distractor Resistance...")
    distractor_results = {}
    for relevance in [1.0, 0.5, 0.25, 0.1]:
        tasks = gen_distractor_tasks(seed=42, difficulty=3, relevance_pct=relevance)
        results = runner.run_batch(tasks)
        correct = sum(1 for r in results if r.correct)
        acc = correct / len(results) * 100 if results else 0
        distractor_results[f"relevance_{int(relevance*100)}pct"] = {
            "tasks": len(results),
            "accuracy_pct": round(acc, 2),
            "mean_latency_ms": round(sum(r.latency_ms for r in results) / len(results), 2) if results else 0,
        }
        print(f"  {int(relevance*100)}% relevant: {correct}/{len(results)} = {acc:.1f}%")
    all_results["distractor_resistance"] = distractor_results

    # ── Section 5: Conflict Resolution ───────────────────────────────
    print("\n[5/8] Conflict Resolution...")
    conflict_suite = BenchmarkSuite(name="conflict_resolution")
    for level in range(1, 7):
        tasks = gen_conflict_tasks(seed=42, difficulty=level)
        results = runner.run_batch(tasks)
        conflict_suite.tasks.extend(tasks)
        conflict_suite.results.extend(results)
        correct = sum(1 for r in results if r.correct)
        print(f"  Level {level}: {correct}/{len(results)}")
    all_results["conflict_resolution"] = conflict_suite.summary()
    all_results["conflict_resolution_stats"] = compute_stats(conflict_suite.results)

    # ── Section 6: Novel Topologies ──────────────────────────────────
    print("\n[6/8] Novel Topology Generalization...")
    novel_suite = BenchmarkSuite(name="novel_topology")
    for level in range(1, 7):
        tasks = gen_novel_topology_tasks(seed=9999, difficulty=level)
        results = runner.run_batch(tasks)
        novel_suite.tasks.extend(tasks)
        novel_suite.results.extend(results)
        correct = sum(1 for r in results if r.correct)
        print(f"  Level {level}: {correct}/{len(results)}")
    all_results["novel_topology"] = novel_suite.summary()
    all_results["novel_topology_stats"] = compute_stats(novel_suite.results)

    # ── Section 7: Ablation Study ────────────────────────────────────
    print("\n[7/8] Ablation Study...")
    ablation_tasks = generate_all_tasks(seed=42, difficulty=3)
    ablation_results = {}
    for config_name, config in ABLATION_CONFIGS.items():
        ablation_runner = SweepRunner()
        results = ablation_runner.run_batch(ablation_tasks)
        correct = sum(1 for r in results if r.correct)
        acc = correct / len(results) * 100 if results else 0
        mean_lat = sum(r.latency_ms for r in results) / len(results) if results else 0
        ablation_results[config_name] = {
            "description": config["description"],
            "tasks": len(results),
            "accuracy_pct": round(acc, 2),
            "mean_latency_ms": round(mean_lat, 2),
        }
        print(f"  {config_name}: {correct}/{len(results)} = {acc:.1f}%")
    all_results["ablation"] = ablation_results

    # ── Section 8: Generalization (unseen seed) ──────────────────────
    print("\n[8/8] Generalization (unseen seed)...")
    gen_tasks = generate_all_tasks(seed=9999, difficulty=3)
    gen_results = runner.run_batch(gen_tasks)
    gen_correct = sum(1 for r in gen_results if r.correct)
    gen_acc = gen_correct / len(gen_results) * 100 if gen_results else 0
    all_results["generalization"] = {
        "tasks": len(gen_results),
        "accuracy_pct": round(gen_acc, 2),
        "stats": compute_stats(gen_results),
    }
    print(f"  Unseen seed: {gen_correct}/{len(gen_results)} = {gen_acc:.1f}%")

    # ── Save results ─────────────────────────────────────────────────
    results_path = RESULTS_DIR / "benchmark_results.json"
    results_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved: {results_path}")

    return all_results


if __name__ == "__main__":
    run_full_benchmark()
