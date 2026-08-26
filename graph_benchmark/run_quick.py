"""
Quick benchmark runner — practical graph sizes that complete in reasonable time.
Uses graph sizes up to 1000 nodes (5000-node graphs take too long for ablation).
"""
import io, sys, json, math, time, platform, os
from pathlib import Path
from typing import Any
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_benchmark.generator.graph_generator import GraphGenerator
from graph_benchmark.generator.task_generator import TaskGenerator, Task
from graph_benchmark.scoring.scorer import BenchmarkScorer, BenchmarkResults, CategoryMetrics
from graph_benchmark.runners.sweep_runner import SweepGraphRunner
from graph_benchmark.run_benchmark import (
    collect_environment, estimate_tokens, assign_context_group,
    OPENAI_REFERENCE, ABLATION_VARIANTS, SCALING_FRACTIONS, DIFF_DIST
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


GRAPH_SIZES = [10, 25, 50, 100, 250, 500, 1000]
DIFFICULTIES = ["easy", "medium", "hard", "extreme"]


def pick_difficulty(seed: int, idx: int) -> str:
    r = hash(f"{seed}_{idx}") % 10000 / 10000.0
    cumulative = 0.0
    for diff, prob in DIFF_DIST.items():
        cumulative += prob
        if r < cumulative:
            return diff
    return "extreme"


def generate_split(seed, num_graphs, graph_sizes, tasks_per_type):
    graph_gen = GraphGenerator(seed=seed)
    task_gen = TaskGenerator(seed=seed + 1)
    all_tasks = []
    for i in range(num_graphs):
        diff = pick_difficulty(seed, i)
        num_nodes = graph_sizes[i % len(graph_sizes)]
        graph = graph_gen.generate(num_nodes=num_nodes, difficulty=diff)
        tasks = task_gen.generate_all(graph, tasks_per_type=tasks_per_type)
        all_tasks.extend(tasks)
    return all_tasks


def main():
    output_dir = Path("graph_benchmark/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    NUM_GRAPHS = 50
    TASKS_PER_TYPE = 3
    SEED = 42

    environment = collect_environment()

    print("=" * 72)
    print("  SWEEP NEURAL MESH GRAPH REASONING BENCHMARK")
    print("=" * 72)

    # ── Generate ──
    print("\n[1/7] Generating dataset splits...")
    t0 = time.perf_counter()
    train = generate_split(SEED, NUM_GRAPHS, GRAPH_SIZES, TASKS_PER_TYPE)
    val = generate_split(SEED + 1000, max(10, NUM_GRAPHS // 5), GRAPH_SIZES, TASKS_PER_TYPE)
    test = generate_split(SEED + 2000, max(10, NUM_GRAPHS // 5), GRAPH_SIZES, TASKS_PER_TYPE)
    gen_time = time.perf_counter() - t0
    total_all = len(train) + len(val) + len(test)
    print(f"  {len(train)} train, {len(val)} val, {len(test)} test = {total_all} total in {gen_time:.1f}s")

    for name, tasks in [("train", train), ("validation", val), ("test", test)]:
        with open(output_dir / f"graph_{name}.jsonl", "w", encoding="utf-8") as f:
            for t in tasks:
                f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")

    # ── Sweep full ──
    print(f"\n[2/7] Running Sweep full mesh on {len(test)} test tasks...")
    runner = SweepGraphRunner(variant="full")
    raw = runner.run_all(test, verbose=True)

    scorer = BenchmarkScorer()
    for r in raw["results"]:
        scorer.add_result(r)
    results = scorer.compute_results("sweep_full", environment)

    # Context groups
    ctx_buckets = defaultdict(list)
    for r in raw["results"]:
        for t in test:
            if t.id == r.task_id:
                ctx_buckets[assign_context_group(t)].append(r)
                break
    for grp, rlist in ctx_buckets.items():
        gs = BenchmarkScorer()
        for r in rlist:
            gs.add_result(r)
        if rlist:
            gr = gs.compute_results(f"sweep_{grp}")
            results.by_context_size[grp] = gr.overall

    # ── Validation ──
    print(f"\n[3/7] Validation run on {len(val)} tasks...")
    vr = SweepGraphRunner(variant="full")
    vraw = vr.run_all(val, verbose=False)
    print(f"  Validation: {vraw['accuracy']:.1%}")

    # ── Ablations ──
    print(f"\n[4/7] Ablation study...")
    abl = {}
    abl_subset = test[:max(50, len(test) // 3)]
    for vname, _ in ABLATION_VARIANTS:
        if vname == "full":
            abl[vname] = {"accuracy": raw["accuracy"], "correct": raw["correct"],
                          "total": raw["total"], "avg_latency_ms": raw["avg_latency_ms"],
                          "peak_memory_mb": raw.get("peak_memory_mb", 0), "variant": vname}
            continue
        t0 = time.perf_counter()
        r = SweepGraphRunner(variant=vname).run_all(abl_subset, verbose=False)
        elapsed = time.perf_counter() - t0
        abl[vname] = {"accuracy": r["accuracy"], "correct": r["correct"],
                       "total": r["total"], "avg_latency_ms": r["avg_latency_ms"],
                       "peak_memory_mb": r.get("peak_memory_mb", 0),
                       "variant": vname, "elapsed_s": round(elapsed, 2)}
        print(f"  {vname:<25s}: {r['accuracy']:.1%}  ({elapsed:.1f}s)")

    # ── Scaling ──
    print(f"\n[5/7] Scaling experiment...")
    scl = {}
    for frac in SCALING_FRACTIONS:
        n = max(5, int(len(test) * frac))
        t0 = time.perf_counter()
        r = SweepGraphRunner(variant=f"scale_{int(frac*100)}", mesh_fraction=frac).run_all(test[:n], verbose=False)
        elapsed = time.perf_counter() - t0
        scl[f"{int(frac*100)}%"] = {"accuracy": r["accuracy"], "correct": r["correct"],
                                    "total": r["total"], "avg_latency_ms": r["avg_latency_ms"],
                                    "peak_memory_mb": r.get("peak_memory_mb", 0), "fraction": frac}
        print(f"  {int(frac*100):>3d}%: {r['accuracy']:.1%}  {r['avg_latency_ms']:.1f}ms  ({elapsed:.1f}s)")

    # ── Stats ──
    print(f"\n[6/7] Statistical analysis...")
    latencies = [r.latency_ms for r in raw["results"] if r.latency_ms > 0]
    sweep_accs = [1.0 if r.correct else 0.0 for r in raw["results"]]
    n = len(sweep_accs)
    p = sum(sweep_accs) / max(1, n)
    se = math.sqrt(p * (1 - p) / max(1, n))

    stats = {
        "sweep": {"accuracy": results.overall.accuracy, "total": results.overall.total, "correct": results.overall.correct},
        "openai_reference": OPENAI_REFERENCE,
        "latency": {},
        "accuracy_ci": {"mean": round(p, 4), "se": round(se, 4),
                         "ci_lower_95": round(max(0, p - 1.96 * se), 4),
                         "ci_upper_95": round(min(1, p + 1.96 * se), 4), "n": n},
    }
    if latencies:
        mean_lat = sum(latencies) / len(latencies)
        sorted_lat = sorted(latencies)
        stats["latency"] = {
            "mean_ms": round(mean_lat, 3),
            "median_ms": round(sorted_lat[len(sorted_lat) // 2], 3),
            "p95_ms": round(sorted_lat[int(len(sorted_lat) * 0.95)], 3),
            "std_ms": round(math.sqrt(sum((x - mean_lat)**2 for x in latencies) / max(1, len(latencies)-1)), 3),
            "n": len(latencies),
        }
    err = {}
    for r in raw["results"]:
        if r.error_type:
            err[r.error_type] = err.get(r.error_type, 0) + 1
    stats["error_analysis"] = err
    stats["total_errors"] = sum(err.values())

    # ── Report ──
    print(f"\n[7/7] Generating report...")
    from graph_benchmark.run_benchmark import _generate_report
    report = _generate_report(
        environment=environment, sweep_results=results, sweep_raw=raw,
        ablation_results=abl, scaling_results=scl, stats=stats,
        context_buckets=ctx_buckets, val_accuracy=vraw["accuracy"],
        test_tasks=test,
    )

    with open(output_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(environment, f, indent=2)
    with open(output_dir / "statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    with open(output_dir / "ablation.json", "w") as f:
        json.dump(abl, f, indent=2)
    with open(output_dir / "scaling.json", "w") as f:
        json.dump(scl, f, indent=2)
    scorer.to_json(results, str(output_dir / "sweep.json"))
    with open(output_dir / "REPORT.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
