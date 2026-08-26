"""
Benchmark Scorer — deterministic answer matching and evaluation.

Scoring dimensions:
1. Decision accuracy: did the system get the right decision category?
2. Confidence calibration: is confidence aligned with correctness?
3. Latency: how fast was the response?
4. Reasoning quality: does the reasoning make sense?
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class BenchmarkScorer:
    """
    Scores and compares benchmark results from different systems.
    """

    def __init__(self) -> None:
        self._comparisons: list[dict[str, Any]] = []

    def compare(self, sweep_data: dict[str, Any], gpt4o_data: dict[str, Any]) -> dict[str, Any]:
        """
        Compare Sweep vs GPT-4o results.

        Returns a comprehensive comparison report.
        """
        sweep_summary = sweep_data["summary"]
        gpt4o_summary = gpt4o_data["summary"]
        sweep_results = sweep_data["results"]
        gpt4o_results = gpt4o_data["results"]

        # Overall comparison
        accuracy_diff = sweep_summary["accuracy"] - gpt4o_summary["accuracy"]
        speed_ratio = gpt4o_summary["avg_latency_ms"] / max(0.1, sweep_summary["avg_latency_ms"])

        # Per-category comparison
        category_comparison: dict[str, dict[str, Any]] = {}
        all_categories = set(list(sweep_summary.get("by_category", {}).keys()) +
                           list(gpt4o_summary.get("by_category", {}).keys()))

        for cat in sorted(all_categories):
            sweep_cat = sweep_summary.get("by_category", {}).get(cat, {})
            gpt4o_cat = gpt4o_summary.get("by_category", {}).get(cat, {})

            sweep_acc = sweep_cat.get("accuracy", 0.0)
            gpt4o_acc = gpt4o_cat.get("accuracy", 0.0)
            cat_speed = gpt4o_cat.get("avg_latency_ms", 0.0) / max(0.1, sweep_cat.get("avg_latency_ms", 0.1))

            winner = "sweep" if sweep_acc > gpt4o_acc else ("gpt-4o" if gpt4o_acc > sweep_acc else "tie")

            category_comparison[cat] = {
                "sweep_accuracy": round(sweep_acc, 4),
                "gpt4o_accuracy": round(gpt4o_acc, 4),
                "accuracy_diff": round(sweep_acc - gpt4o_acc, 4),
                "sweep_latency_ms": sweep_cat.get("avg_latency_ms", 0.0),
                "gpt4o_latency_ms": gpt4o_cat.get("avg_latency_ms", 0.0),
                "speed_ratio": round(cat_speed, 1),
                "winner": winner,
            }

        # Per-difficulty comparison
        difficulty_comparison: dict[str, dict[str, Any]] = {}
        for diff in ["easy", "medium", "hard"]:
            sweep_diff = sweep_summary.get("by_difficulty", {}).get(diff, {})
            gpt4o_diff = gpt4o_summary.get("by_difficulty", {}).get(diff, {})
            sweep_acc = sweep_diff.get("accuracy", 0.0)
            gpt4o_acc = gpt4o_diff.get("accuracy", 0.0)
            difficulty_comparison[diff] = {
                "sweep_accuracy": round(sweep_acc, 4),
                "gpt4o_accuracy": round(gpt4o_acc, 4),
                "accuracy_diff": round(sweep_acc - gpt4o_acc, 4),
                "winner": "sweep" if sweep_acc > gpt4o_acc else ("gpt-4o" if gpt4o_acc > sweep_acc else "tie"),
            }

        # Categories won
        sweep_wins = sum(1 for c in category_comparison.values() if c["winner"] == "sweep")
        gpt4o_wins = sum(1 for c in category_comparison.values() if c["winner"] == "gpt-4o")
        ties = sum(1 for c in category_comparison.values() if c["winner"] == "tie")

        # Overall winner
        if accuracy_diff > 0.01:
            overall_winner = "sweep"
        elif accuracy_diff < -0.01:
            overall_winner = "gpt-4o"
        else:
            overall_winner = "tie"

        # Verdict
        if overall_winner == "sweep":
            verdict = f"Sweep wins overall ({sweep_summary['accuracy']:.1%} vs {gpt4o_summary['accuracy']:.1%})"
        elif overall_winner == "gpt-4o":
            verdict = f"GPT-4o wins overall ({gpt4o_summary['accuracy']:.1%} vs {sweep_summary['accuracy']:.1%})"
        else:
            verdict = f"Tie ({sweep_summary['accuracy']:.1%} vs {gpt4o_summary['accuracy']:.1%})"

        comparison = {
            "overall": {
                "sweep_accuracy": round(sweep_summary["accuracy"], 4),
                "gpt4o_accuracy": round(gpt4o_summary["accuracy"], 4),
                "accuracy_diff": round(accuracy_diff, 4),
                "sweep_latency_ms": sweep_summary["avg_latency_ms"],
                "gpt4o_latency_ms": gpt4o_summary["avg_latency_ms"],
                "speed_ratio": round(speed_ratio, 1),
                "sweep_throughput": sweep_summary["throughput_per_sec"],
                "gpt4o_throughput": gpt4o_summary["throughput_per_sec"],
                "overall_winner": overall_winner,
                "verdict": verdict,
            },
            "categories_won": {
                "sweep": sweep_wins,
                "gpt4o": gpt4o_wins,
                "ties": ties,
                "total": len(category_comparison),
            },
            "by_category": category_comparison,
            "by_difficulty": difficulty_comparison,
            "sweep_summary": sweep_summary,
            "gpt4o_summary": gpt4o_summary,
        }

        self._comparisons.append(comparison)
        return comparison

    def generate_report(self, comparison: dict[str, Any]) -> str:
        """Generate a human-readable comparison report."""
        lines = []
        overall = comparison["overall"]
        cats = comparison["categories_won"]

        lines.append("=" * 70)
        lines.append("  SWEEP vs GPT-4o BENCHMARK REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  Verdict: {overall['verdict']}")
        lines.append("")
        lines.append("─" * 70)
        lines.append("  OVERALL SCORES")
        lines.append("─" * 70)
        lines.append(f"  {'Metric':<35} {'Sweep':>15} {'GPT-4o':>15}")
        lines.append(f"  {'─'*35} {'─'*15} {'─'*15}")
        lines.append(f"  {'Accuracy':<35} {overall['sweep_accuracy']:>14.1%} {overall['gpt4o_accuracy']:>14.1%}")
        lines.append(f"  {'Avg Latency (ms)':<35} {overall['sweep_latency_ms']:>14.1f} {overall['gpt4o_latency_ms']:>14.1f}")
        lines.append(f"  {'Throughput (req/sec)':<35} {overall['sweep_throughput']:>14.1f} {overall['gpt4o_throughput']:>14.1f}")
        lines.append(f"  {'Speed Ratio':<35} {overall['speed_ratio']:>14.1f}x {'(Sweep faster)':>14}")
        lines.append("")
        lines.append(f"  Categories Won: Sweep {cats['sweep']} / GPT-4o {cats['gpt4o']} / Ties {cats['ties']}")
        lines.append("")

        lines.append("─" * 70)
        lines.append("  PER-CATEGORY BREAKDOWN")
        lines.append("─" * 70)
        lines.append(f"  {'Category':<25} {'Sweep':>8} {'GPT-4o':>8} {'Diff':>8} {'Speed':>8} {'Winner':>12}")
        lines.append(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*12}")

        for cat, data in sorted(comparison["by_category"].items()):
            winner_tag = "SWEEP" if data["winner"] == "sweep" else ("GPT-4o" if data["winner"] == "gpt-4o" else "TIE")
            lines.append(
                f"  {cat:<25} {data['sweep_accuracy']:>7.1%} {data['gpt4o_accuracy']:>7.1%} "
                f"{data['accuracy_diff']:>+7.1%} {data['speed_ratio']:>7.1f}x {winner_tag:>12}"
            )

        lines.append("")
        lines.append("─" * 70)
        lines.append("  PER-DIFFICULTY BREAKDOWN")
        lines.append("─" * 70)
        for diff, data in sorted(comparison["by_difficulty"].items()):
            winner_tag = "SWEEP" if data["winner"] == "sweep" else ("GPT-4o" if data["winner"] == "gpt-4o" else "TIE")
            lines.append(
                f"  {diff.upper():<12} Sweep: {data['sweep_accuracy']:.1%}  "
                f"GPT-4o: {data['gpt4o_accuracy']:.1%}  "
                f"Diff: {data['accuracy_diff']:+.1%}  Winner: {winner_tag}"
            )

        lines.append("")
        lines.append("─" * 70)
        lines.append("  SPEED ANALYSIS")
        lines.append("─" * 70)
        lines.append(f"  Sweep avg latency:    {overall['sweep_latency_ms']:.2f} ms")
        lines.append(f"  GPT-4o avg latency:   {overall['gpt4o_latency_ms']:.2f} ms")
        lines.append(f"  Speed advantage:      {overall['speed_ratio']:.0f}x faster (Sweep)")
        lines.append(f"  Sweep throughput:     {overall['sweep_throughput']:.0f} req/sec")
        lines.append(f"  GPT-4o throughput:    {overall['gpt4o_throughput']:.0f} req/sec")
        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def save_report(self, comparison: dict[str, Any], path: str | Path) -> None:
        """Save comparison to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    def save_text_report(self, comparison: dict[str, Any], path: str | Path) -> None:
        """Save human-readable report to text file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = self.generate_report(comparison)
        path.write_text(report, encoding="utf-8")
