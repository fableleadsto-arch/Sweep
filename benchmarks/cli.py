#!/usr/bin/env python3
"""
sweep-benchmark — Scientific Benchmark CLI for Sweep Neural Engine.

Usage:
    sweep-benchmark run --suite full
    sweep-benchmark run --suite reasoning
    sweep-benchmark run --suite multimodal
    sweep-benchmark run --suite sweep-specific
    sweep-benchmark run --suite quick
    sweep-benchmark full
    sweep-benchmark compare
    sweep-benchmark report
    sweep-benchmark contamination-check
    sweep-benchmark ablation
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.core.engine import BenchmarkEngine, BenchmarkConfig
from benchmarks.core.ablation import AblationStudy
from benchmarks.core.normalization import PromptNormalizer
from benchmarks.contamination.controller import ContaminationController
from benchmarks.reports.generator import ReportGenerator
from benchmarks.evaluators.scorer import BenchmarkScorer
from benchmarks.metrics.statistics import BenchmarkMetrics

logger = logging.getLogger("sweep_benchmark")


def cmd_run(args: argparse.Namespace) -> None:
    """Run the benchmark suite."""
    print("=" * 70)
    print("  SWEEP NEURAL ENGINE SCIENTIFIC BENCHMARK")
    print("  Version 2.0.0")
    print("=" * 70)
    print()

    # Load config
    config = BenchmarkConfig(
        suite=args.suite,
        cases_per_task=args.cases,
        seed=args.seed,
        multi_run_count=args.runs,
        output_dir=args.output,
        verbose=args.verbose,
    )

    # Initialize engine
    engine = BenchmarkEngine(config)

    # Step 1: Environment report
    print("ENVIRONMENT REPORT")
    print("-" * 50)
    env_dict = engine.environment.to_dict()
    for k, v in env_dict.items():
        print(f"  {k}: {v}")
    print()

    # Step 2: Load tasks
    print(f"[1/8] Loading tasks for suite '{args.suite}'...")
    task_filter = None
    if args.suite == "reasoning":
        task_filter = ["reasoning", "mathematics", "planning", "uncertainty"]
    elif args.suite == "multimodal":
        task_filter = ["multimodal", "data_analysis"]
    elif args.suite == "sweep-specific":
        task_filter = ["sweep_specific", "entity_resolution", "evidence_reasoning"]
    elif args.suite == "quick":
        task_filter = ["reasoning", "mathematics", "coding"]

    tasks = engine.load_tasks(task_filter=task_filter)
    print(f"  Loaded {len(tasks)} tasks across {len(set(t.category.value for t in tasks))} categories")
    print()

    # Step 3: Integrity check
    print("[2/8] Running integrity checks...")
    integrity = engine.check_integrity()
    print(f"  Integrity: {integrity['overall']}")
    print(f"  Dataset hash: {integrity.get('dataset_hash', 'N/A')}")
    print(f"  Total tasks: {integrity.get('total_tasks', 0)}")
    if not integrity["overall"]:
        print("  WARNING: Integrity check FAILED. Results may be INVALID.")
        print("  Marking run as INVALID per Section 41.")
    print()

    # Step 4: Contamination check
    print("[3/8] Running contamination checks...")
    contamination_ctrl = ContaminationController()
    contamination_report = contamination_ctrl.check_integrity(tasks)
    print(f"  Contamination integrity: {contamination_report.integrity}")
    print(f"  Hashed tasks: {contamination_report.hashed_tasks}")
    print(f"  Hidden tests: {contamination_report.hidden_test_count}")
    if contamination_report.issues:
        print("  Issues:")
        for issue in contamination_report.issues:
            print(f"    - {issue}")
    print()

    # Step 5: Run Sweep
    print("[4/8] Running Sweep Neural Mesh...")
    sweep_stats = {}
    try:
        from benchmarks.adapters.sweep_adapter import SweepAdapter
        sweep_adapter = SweepAdapter(enable_ml=args.enable_ml)

        if args.multi_run and args.runs > 1:
            print(f"  Running {args.runs} iterations for statistical robustness...")
            multi_results = engine.run_multi(sweep_adapter, "sweep_neural_mesh", tasks, runs=args.runs)
            # Use the last run for primary results
            sweep_results = engine.run(sweep_adapter, "sweep_neural_mesh", tasks)
        else:
            sweep_results = engine.run(sweep_adapter, "sweep_neural_mesh", tasks)

        sweep_stats = engine.compute_statistics("sweep_neural_mesh")
        print(f"  Sweep accuracy: {sweep_stats.get('accuracy', 0):.1%}")
        print(f"  Sweep mean latency: {sweep_stats.get('mean_latency_ms', 0):.1f}ms")

        # Print calibration
        cal = sweep_stats.get("calibration", {})
        if cal:
            print(f"  Sweep Brier score: {cal.get('brier_score', 0):.3f}")
            print(f"  Sweep ECE: {cal.get('expected_calibration_error', 0):.3f}")
    except Exception as e:
        print(f"  Sweep runner failed: {e}")
        sweep_stats = {}
    print()

    # Step 6: Run external models (if configured)
    print("[5/8] Running external model comparisons...")
    models_run = ["sweep_neural_mesh"]

    # Try OpenAI
    try:
        from benchmarks.adapters.openai_adapter import OpenAIAdapter
        openai_adapter = OpenAIAdapter(model_id=args.openai_model or "gpt-4o")
        if openai_adapter.health_check():
            engine.run(openai_adapter, "gpt4o", tasks)
            models_run.append("gpt4o")
            print(f"  GPT-4o completed")
        else:
            print("  GPT-4o: no API key (skipping)")
    except Exception as e:
        print(f"  GPT-4o: {e}")

    # Try Anthropic
    try:
        from benchmarks.adapters.anthropic_adapter import AnthropicAdapter
        anthropic_adapter = AnthropicAdapter(model_id=args.anthropic_model or "claude-sonnet-4-20250514")
        if anthropic_adapter.health_check():
            engine.run(anthropic_adapter, "claude", tasks)
            models_run.append("claude")
            print(f"  Claude completed")
        else:
            print("  Claude: no API key (skipping)")
    except Exception as e:
        print(f"  Claude: {e}")

    # Try Google
    try:
        from benchmarks.adapters.google_adapter import GoogleAdapter
        google_adapter = GoogleAdapter(model_id=args.google_model or "gemini-2.5-pro")
        if google_adapter.health_check():
            engine.run(google_adapter, "gemini", tasks)
            models_run.append("gemini")
            print(f"  Gemini completed")
        else:
            print("  Gemini: no API key (skipping)")
    except Exception as e:
        print(f"  Gemini: {e}")
    print()

    # Step 7: Statistical analysis
    print("[6/8] Running statistical analysis...")
    all_stats = {}
    for model_name in models_run:
        all_stats[model_name] = engine.compute_statistics(model_name)

    comparison = None
    if len(models_run) > 1:
        comparison = engine.compare_models(models_run)
        print(f"  Compared {len(models_run)} models")
        for comp in comparison.get("pairwise_comparisons", []):
            sig = "YES" if comp.get("significant") else "NO"
            print(f"  {comp['model_a']} vs {comp['model_b']}: "
                  f"diff={comp.get('absolute_difference', 0)*100:.2f}pp, "
                  f"significant={sig}")
    print()

    # Step 8: Ablation study
    ablation_results = None
    if args.ablation:
        print("[7/8] Running ablation study...")
        try:
            from benchmarks.adapters.sweep_adapter import SweepAdapter
            sweep_adapter = SweepAdapter(enable_ml=args.enable_ml)
            ablation_study = AblationStudy()
            ablation_results = ablation_study.run(
                tasks, sweep_adapter, engine,
            )
            print("  Ablation study complete")
            for comp, desc in ablation_results.get("summary", {}).items():
                print(f"    {comp}: {desc}")
        except Exception as e:
            print(f"  Ablation failed: {e}")
    else:
        print("[7/8] Skipping ablation (use --ablation to enable)")
    print()

    # Step 9: Generate reports
    print("[8/8] Generating reports...")
    reporter = ReportGenerator(args.output)
    manifest = engine.generate_manifest()
    files = reporter.generate_all(
        statistics=all_stats,
        comparison=comparison,
        ablation=ablation_results,
        contamination=contamination_report.to_dict(),
        manifest=manifest,
        environment=env_dict,
    )
    for fmt, path in files.items():
        print(f"  {fmt}: {path}")

    # Generate honesty report
    if args.honesty:
        honesty_report = reporter.generate_honesty_report(all_stats)
        print()
        print(honesty_report)

    print()

    # Terminal summary (Section 47)
    reporter.print_terminal_summary(all_stats, comparison)

    print(f"\nResults saved to: {args.output}/")


def cmd_full(args: argparse.Namespace) -> None:
    """Alias for 'run --suite full' with all features enabled."""
    args.suite = "full"
    args.ablation = True
    args.honesty = True
    args.multi_run = True
    cmd_run(args)


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare existing results."""
    results_dir = Path(args.results_dir)
    print("Comparing existing benchmark results...")
    print(f"  Results directory: {results_dir}")

    # Load all result files
    stats = {}
    for json_file in results_dir.glob("*_results.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            model_name = json_file.stem.replace("_results", "")
            stats[model_name] = data.get("summary", data)
        except Exception as e:
            print(f"  Error loading {json_file}: {e}")

    if not stats:
        print("  No results found.")
        return

    engine = BenchmarkEngine()
    engine._results = {k: [] for k in stats}
    print(f"  Found {len(stats)} result sets")

    reporter = ReportGenerator(args.output)
    reporter.print_terminal_summary(stats)


def cmd_contamination_check(args: argparse.Namespace) -> None:
    """Run contamination checks only."""
    print("Running contamination checks...")
    config = BenchmarkConfig(suite="full", cases_per_task=200)
    engine = BenchmarkEngine(config)
    tasks = engine.load_tasks()

    ctrl = ContaminationController()
    report = ctrl.check_integrity(tasks)

    print(f"  Integrity: {report.integrity}")
    print(f"  Total tasks: {report.total_tasks}")
    print(f"  Hashed: {report.hashed_tasks}")
    print(f"  Hidden tests: {report.hidden_test_count}")
    print(f"  Private: {report.private_count}")
    print(f"  Public: {report.public_count}")
    print(f"  Fresh: {report.fresh_count}")
    print(f"  Dataset hash: {report.dataset_hash}")
    if report.issues:
        print("  Issues:")
        for issue in report.issues:
            print(f"    - {issue}")

    # Verify holdout inaccessibility
    holdout_ids = [t.id for t in tasks if t.is_hidden_test]
    accessible_ids = [t.id for t in tasks if not t.is_hidden_test]
    if ctrl.verify_holdout_inaccessible(holdout_ids, accessible_ids):
        print("  Holdout inaccessibility: PASS")
    else:
        print("  Holdout inaccessibility: FAIL — CRITICAL CONTAMINATION")

    # Save integrity manifest
    manifest = ctrl.generate_integrity_manifest()
    print(f"  Integrity manifest saved")


def cmd_report(args: argparse.Namespace) -> None:
    """Generate report from existing results."""
    results_dir = Path(args.results_dir)
    print(f"Generating report from {results_dir}...")

    # Load stats
    stats = {}
    report_file = results_dir / "final_report.json"
    if report_file.exists():
        data = json.loads(report_file.read_text(encoding="utf-8"))
        stats = data.get("statistics", {})

    if not stats:
        print("  No statistics found. Run 'sweep-benchmark run' first.")
        return

    reporter = ReportGenerator(args.output)
    files = reporter.generate_all(statistics=stats)
    for fmt, path in files.items():
        print(f"  {fmt}: {path}")

    reporter.print_terminal_summary(stats)


def cmd_ablation(args: argparse.Namespace) -> None:
    """Run ablation study."""
    print("Running ablation study...")
    config = BenchmarkConfig(suite="full", cases_per_task=args.cases)
    engine = BenchmarkEngine(config)
    tasks = engine.load_tasks()

    from benchmarks.adapters.sweep_adapter import SweepAdapter
    sweep_adapter = SweepAdapter()

    study = AblationStudy()
    results = study.run(tasks, sweep_adapter, engine)

    print("\nAblation Results:")
    print("-" * 50)
    for comp, desc in results.get("summary", {}).items():
        print(f"  {comp}: {desc}")

    print("\nConfiguration Details:")
    for name, data in results.get("configurations", {}).items():
        print(f"  {name}: accuracy={data.get('accuracy', 0):.1%}, latency={data.get('avg_latency_ms', 0):.1f}ms")

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "ablation_results.json"
    manifest_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to: {manifest_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sweep Neural Engine Scientific Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sweep-benchmark run --suite full
  sweep-benchmark run --suite reasoning --cases 100
  sweep-benchmark run --suite quick --verbose
  sweep-benchmark full
  sweep-benchmark compare
  sweep-benchmark contamination-check
  sweep-benchmark report
  sweep-benchmark ablation
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ── run ──
    run_parser = subparsers.add_parser("run", help="Run the benchmark suite")
    run_parser.add_argument("--suite", default="full",
                           choices=["full", "reasoning", "multimodal", "sweep_specific", "quick"],
                           help="Benchmark suite to run")
    run_parser.add_argument("--cases", type=int, default=200,
                           help="Number of cases per task category")
    run_parser.add_argument("--seed", type=int, default=42,
                           help="Random seed for reproducibility")
    run_parser.add_argument("--runs", type=int, default=5,
                           help="Number of multi-run iterations")
    run_parser.add_argument("--output", default="benchmarks/reports",
                           help="Output directory for reports")
    run_parser.add_argument("--verbose", action="store_true",
                           help="Verbose output")
    run_parser.add_argument("--ablation", action="store_true",
                           help="Run ablation study")
    run_parser.add_argument("--honesty", action="store_true",
                           help="Generate honesty report")
    run_parser.add_argument("--multi-run", action="store_true",
                           help="Enable multi-run testing for stochastic evaluation")
    run_parser.add_argument("--enable-ml", action="store_true",
                           help="Enable ML engines (embeddings, NER, sentiment)")
    run_parser.add_argument("--openai-model", default=None,
                           help="OpenAI model ID (default: gpt-4o)")
    run_parser.add_argument("--anthropic-model", default=None,
                           help="Anthropic model ID")
    run_parser.add_argument("--google-model", default=None,
                           help="Google model ID")
    run_parser.set_defaults(func=cmd_run)

    # ── full ──
    full_parser = subparsers.add_parser("full", help="Run full benchmark with all features")
    full_parser.add_argument("--cases", type=int, default=200)
    full_parser.add_argument("--seed", type=int, default=42)
    full_parser.add_argument("--runs", type=int, default=5)
    full_parser.add_argument("--output", default="benchmarks/reports")
    full_parser.add_argument("--verbose", action="store_true")
    full_parser.add_argument("--enable-ml", action="store_true")
    full_parser.add_argument("--openai-model", default=None)
    full_parser.add_argument("--anthropic-model", default=None)
    full_parser.add_argument("--google-model", default=None)
    full_parser.set_defaults(func=cmd_full)

    # ── compare ──
    compare_parser = subparsers.add_parser("compare", help="Compare existing results")
    compare_parser.add_argument("--results-dir", default="benchmarks/results",
                               help="Directory containing results")
    compare_parser.add_argument("--output", default="benchmarks/reports",
                               help="Output directory for comparison report")
    compare_parser.set_defaults(func=cmd_compare)

    # ── report ──
    report_parser = subparsers.add_parser("report", help="Generate report from existing results")
    report_parser.add_argument("--results-dir", default="benchmarks/reports",
                              help="Directory containing results")
    report_parser.add_argument("--output", default="benchmarks/reports",
                              help="Output directory")
    report_parser.set_defaults(func=cmd_report)

    # ── contamination-check ──
    cont_parser = subparsers.add_parser("contamination-check",
                                        help="Run contamination integrity checks")
    cont_parser.set_defaults(func=cmd_contamination_check)

    # ── ablation ──
    ablation_parser = subparsers.add_parser("ablation",
                                            help="Run ablation study")
    ablation_parser.add_argument("--cases", type=int, default=200)
    ablation_parser.add_argument("--output", default="benchmarks/reports")
    ablation_parser.set_defaults(func=cmd_ablation)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    args.func(args)


if __name__ == "__main__":
    main()
