from __future__ import annotations

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

"""
Sweep Benchmark -- Main Entry Point

Runs the full benchmark: generate dataset -> run Sweep -> run GPT-4o -> compare -> report.
"""
import argparse
import json
import time
from pathlib import Path

# Ensure we can import from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.dataset.generate import BenchmarkDataset
from benchmarks.runners.sweep_runner import SweepRunner
from benchmarks.runners.openai_runner import OpenAIRunner
from benchmarks.evaluation.scorer import BenchmarkScorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Neural Mesh Benchmark")
    parser.add_argument("--cases", type=int, default=1000, help="Number of test cases (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SWEEP NEURAL MESH BENCHMARK")
    print("=" * 70)
    print()

    # ── Step 1: Generate Dataset ──
    print("[1/5] Generating benchmark dataset...")
    t0 = time.perf_counter()
    dataset = BenchmarkDataset(seed=args.seed)
    cases = dataset.generate()

    # Trim to requested count if needed
    if len(cases) > args.cases:
        cases = cases[:args.cases]

    dataset.save(output_dir / "dataset.json")
    gen_time = time.perf_counter() - t0
    print(f"       Generated {len(cases)} test cases in {gen_time:.2f}s")
    print(f"       Categories: {len(dataset.CATEGORIES)}")
    print(f"       Stats: {json.dumps(dataset.stats, indent=0)}")
    print()

    # ── Step 2: Run Sweep ──
    print("[2/5] Running Sweep ReasoningCortex...")
    t0 = time.perf_counter()
    sweep_runner = SweepRunner(enable_ml=False)
    sweep_data = sweep_runner.run_all(cases, verbose=args.verbose)
    sweep_time = time.perf_counter() - t0
    sweep_runner.save(sweep_data, output_dir / "sweep_results.json")
    print(f"       Sweep completed in {sweep_time:.2f}s")
    print(f"       Accuracy: {sweep_data['summary']['accuracy']:.1%}")
    print(f"       Avg latency: {sweep_data['summary']['avg_latency_ms']:.2f}ms")
    print(f"       Throughput: {sweep_data['summary']['throughput_per_sec']:.0f} req/sec")
    print()

    # ── Step 3: Run GPT-4o (estimated) ──
    print("[3/5] Estimating GPT-4o performance from published numbers...")
    t0 = time.perf_counter()
    gpt4o_runner = OpenAIRunner(seed=args.seed)
    gpt4o_data = gpt4o_runner.run_all(cases)
    gpt4o_time = time.perf_counter() - t0
    gpt4o_runner.save(gpt4o_data, output_dir / "gpt4o_results.json")
    print(f"       GPT-4o estimation completed in {gpt4o_time:.2f}s")
    print(f"       Estimated accuracy: {gpt4o_data['summary']['accuracy']:.1%}")
    print(f"       Estimated avg latency: {gpt4o_data['summary']['avg_latency_ms']:.1f}ms")
    print()

    # ── Step 4: Compare ──
    print("[4/5] Comparing results...")
    scorer = BenchmarkScorer()
    comparison = scorer.compare(sweep_data, gpt4o_data)
    scorer.save_report(comparison, output_dir / "comparison.json")
    scorer.save_text_report(comparison, output_dir / "REPORT.txt")
    print("       Comparison complete")
    print()

    # ── Step 5: Print Report ──
    print("[5/5] Benchmark Report:")
    print()
    report = scorer.generate_report(comparison)
    print(report)

    # Save metadata
    metadata = {
        "benchmark_version": "1.0.0",
        "total_cases": len(cases),
        "sweep_time_s": round(sweep_time, 2),
        "gpt4o_time_s": round(gpt4o_time, 2),
        "total_time_s": round(sweep_time + gpt4o_time, 2),
        "output_dir": str(output_dir),
        "files": {
            "dataset": str(output_dir / "dataset.json"),
            "sweep_results": str(output_dir / "sweep_results.json"),
            "gpt4o_results": str(output_dir / "gpt4o_results.json"),
            "comparison": str(output_dir / "comparison.json"),
            "report": str(output_dir / "REPORT.txt"),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print(f"Results saved to: {output_dir}/")
    print(f"  - dataset.json ({len(cases)} test cases)")
    print(f"  - sweep_results.json")
    print(f"  - gpt4o_results.json")
    print(f"  - comparison.json")
    print(f"  - REPORT.txt (human-readable)")


if __name__ == "__main__":
    main()
