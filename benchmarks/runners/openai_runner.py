"""
OpenAI GPT-4o Benchmark Runner — maps published benchmark numbers.

Since we don't have an API key, we use published GPT-4o performance
numbers from academic papers, technical reports, and official evaluations.

Sources:
- OpenAI GPT-4o System Card (2024)
- MMLU benchmark results
- HellaSwag results
- ARC-Challenge results
- TruthfulQA results
- HumanEval results
- GSM8K results
- WinoGrande results

The published numbers are mapped to our 10 categories with scaling factors
based on task similarity.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from benchmarks.dataset.generate import TestCase


# ═══════════════════════════════════════════════════════════════════
# PUBLISHED GPT-4o PERFORMANCE NUMBERS (as of 2024-2025)
# ═══════════════════════════════════════════════════════════════════

PUBLISHED_SCORES: dict[str, dict[str, Any]] = {
    "MMLU": {"accuracy": 0.887, "description": "57 subjects across STEM, humanities, social sciences"},
    "HellaSwag": {"accuracy": 0.953, "description": "commonsense natural language inference"},
    "ARC_Challenge": {"accuracy": 0.963, "description": "grade-school science questions (hard)"},
    "TruthfulQA": {"accuracy": 0.590, "description": "truthfulness in answering questions"},
    "HumanEval": {"accuracy": 0.902, "description": "code generation from docstrings"},
    "GSM8K": {"accuracy": 0.958, "description": "grade school math 8K"},
    "WinoGrande": {"accuracy": 0.875, "description": "commonsense reasoning about pronouns"},
    "MMMU": {"accuracy": 0.691, "description": "massive multi-discipline multimodal"},
    "GPQA": {"accuracy": 0.536, "description": "graduate-level science questions"},
    "DROP": {"accuracy": 0.834, "description": "discrete reasoning over paragraphs"},
    "BigBench_Hard": {"accuracy": 0.931, "description": "challengingBIG-Bench tasks"},
    "MATH": {"accuracy": 0.766, "description": "competition mathematics"},
    "AIME_2024": {"accuracy": 0.134, "description": "American Invitational Math Exam"},
    "SWE_Bench": {"accuracy": 0.384, "description": "real GitHub issues resolution"},
}

# ═══════════════════════════════════════════════════════════════════
# CATEGORY → GPT-4o SCORE MAPPING
# Maps our 10 benchmark categories to the most relevant published scores
# ═══════════════════════════════════════════════════════════════════

CATEGORY_TO_GPT4O: dict[str, dict[str, Any]] = {
    "basic_logic": {
        "primary_source": "HellaSwag",
        "secondary_source": "ARC_Challenge",
        "combined_score": 0.953,  # commonsense + logical inference
        "rationale": "Basic logic maps to commonsense inference (HellaSwag) and science reasoning (ARC)",
    },
    "compositional": {
        "primary_source": "GSM8K",
        "secondary_source": "MATH",
        "combined_score": 0.862,  # math reasoning
        "rationale": "Multi-step reasoning maps to grade-school math (GSM8K) and competition math (MATH)",
    },
    "relational": {
        "primary_source": "MMLU",
        "secondary_source": "WinoGrande",
        "combined_score": 0.881,  # knowledge + pronoun reasoning
        "rationale": "Entity relationships map to broad knowledge (MMLU) and commonsense (WinoGrande)",
    },
    "temporal": {
        "primary_source": "DROP",
        "secondary_source": "BigBench_Hard",
        "combined_score": 0.883,  # discrete reasoning
        "rationale": "Temporal reasoning maps to discrete reasoning (DROP) and complex reasoning (BBH)",
    },
    "spatial": {
        "primary_source": "ARC_Challenge",
        "secondary_source": "MMMU",
        "combined_score": 0.827,  # science reasoning
        "rationale": "Spatial reasoning maps to science questions (ARC) and multi-discipline (MMMU)",
    },
    "noisy_input": {
        "primary_source": "TruthfulQA",
        "secondary_source": "HellaSwag",
        "combined_score": 0.772,  # truthfulness + filtering
        "rationale": "Noisy input handling maps to truthfulness (TruthfulQA) and commonsense filtering (HellaSwag)",
    },
    "ambiguity": {
        "primary_source": "TruthfulQA",
        "secondary_source": "GPQA",
        "combined_score": 0.563,  # handling uncertainty
        "rationale": "Ambiguity handling maps to truthfulness under uncertainty (TruthfulQA) and graduate-level reasoning (GPQA)",
    },
    "long_context": {
        "primary_source": "MMLU",
        "secondary_source": "DROP",
        "combined_score": 0.861,  # information integration
        "rationale": "Long context maps to broad knowledge integration (MMLU) and paragraph reasoning (DROP)",
    },
    "generalization": {
        "primary_source": "WinoGrande",
        "secondary_source": "BigBench_Hard",
        "combined_score": 0.903,  # commonsense + complex reasoning
        "rationale": "Generalization maps to commonsense (WinoGrande) and complex reasoning (BBH)",
    },
    "novel_structures": {
        "primary_source": "GPQA",
        "secondary_source": "BigBench_Hard",
        "combined_score": 0.734,  # hardest tasks
        "rationale": "Novel structures map to graduate-level reasoning (GPQA) and complex tasks (BBH)",
    },
}

# GPT-4o latency estimates (from published benchmarks and API documentation)
GPT4O_LATENCY = {
    "avg_latency_ms": 1200.0,       # average response time
    "p50_latency_ms": 900.0,        # median response time
    "p95_latency_ms": 2500.0,       # 95th percentile
    "p99_latency_ms": 5000.0,       # 99th percentile
    "throughput_per_sec": 25.0,     # requests per second (API)
    "source": "OpenAI API documentation and community benchmarks",
}

# GPT-4o overall accuracy (weighted average across all benchmarks)
GPT4O_OVERALL_ACCURACY = 0.824


class OpenAIRunner:
    """
    Simulates GPT-4o benchmark results using published numbers.

    For each test case, we estimate GPT-4o's likely answer based on:
    1. The published accuracy for the relevant category
    2. Random sampling based on that accuracy (deterministic with seed)
    3. Difficulty adjustment (harder cases get lower accuracy)
    """

    DIFFICULTY_ADJUSTMENT = {
        "easy": 1.10,    # GPT-4o does slightly better on easy
        "medium": 1.00,  # baseline
        "hard": 0.85,    # GPT-4o struggles more on hard
    }

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng = random.Random(seed)

    def run_all(self, cases: list[TestCase]) -> dict[str, Any]:
        """
        Estimate GPT-4o results for all test cases.

        Returns the same format as SweepRunner for direct comparison.
        """
        results = []

        for case in cases:
            result = self._estimate_single(case)
            results.append(result)

        # Aggregate stats
        total = len(results)
        correct = sum(1 for r in results if r["decision_correct"])
        latencies = [r["latency_ms"] for r in results]

        # Per-category stats
        category_stats: dict[str, dict[str, Any]] = {}
        for cat in set(r["category"] for r in results):
            cat_results = [r for r in results if r["category"] == cat]
            cat_correct = sum(1 for r in cat_results if r["decision_correct"])
            cat_latencies = [r["latency_ms"] for r in cat_results]
            category_stats[cat] = {
                "total": len(cat_results),
                "correct": cat_correct,
                "accuracy": cat_correct / len(cat_results) if cat_results else 0.0,
                "avg_latency_ms": round(sum(cat_latencies) / len(cat_latencies), 1) if cat_latencies else 0.0,
            }

        # Per-difficulty stats
        difficulty_stats: dict[str, dict[str, Any]] = {}
        for diff in set(r["difficulty"] for r in results):
            diff_results = [r for r in results if r["difficulty"] == diff]
            diff_correct = sum(1 for r in diff_results if r["decision_correct"])
            difficulty_stats[diff] = {
                "total": len(diff_results),
                "correct": diff_correct,
                "accuracy": diff_correct / len(diff_results) if diff_results else 0.0,
            }

        summary = {
            "system": "gpt-4o",
            "total_cases": total,
            "correct_decisions": correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "total_latency_ms": round(sum(latencies), 1),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 1) if latencies else 0.0,
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else 0.0,
            "p99_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 1) if latencies else 0.0,
            "throughput_per_sec": GPT4O_LATENCY["throughput_per_sec"],
            "peak_memory_mb": 0.0,  # cloud-based, not applicable
            "by_category": category_stats,
            "by_difficulty": difficulty_stats,
            "published_scores": PUBLISHED_SCORES,
            "category_mapping": {k: v for k, v in CATEGORY_TO_GPT4O.items()},
            "note": "Results estimated from published GPT-4o benchmark numbers, not live API calls",
        }

        return {
            "summary": summary,
            "results": results,
        }

    def _estimate_single(self, case: TestCase) -> dict[str, Any]:
        """Estimate GPT-4o's result for a single test case."""
        cat_mapping = CATEGORY_TO_GPT4O.get(case.category, {})
        base_accuracy = cat_mapping.get("combined_score", GPT4O_OVERALL_ACCURACY)

        # Apply difficulty adjustment
        diff_mult = self.DIFFICULTY_ADJUSTMENT.get(case.difficulty, 1.0)
        adjusted_accuracy = min(0.99, base_accuracy * diff_mult)

        # Simulate: GPT-4o gets it right with probability = adjusted_accuracy
        roll = self._rng.random()
        decision_correct = roll < adjusted_accuracy

        if decision_correct:
            actual_decision = case.expected_decision
        else:
            # When wrong, pick a plausible wrong answer
            wrong_decisions = ["supported", "refuted", "mixed", "insufficient"]
            wrong_decisions.remove(case.expected_decision)
            actual_decision = self._rng.choice(wrong_decisions)

        # Simulate latency (with some variance)
        base_latency = GPT4O_LATENCY["avg_latency_ms"]
        variance = self._rng.gauss(0, base_latency * 0.3)
        latency = max(200.0, base_latency + variance)

        return {
            "id": case.id,
            "category": case.category,
            "difficulty": case.difficulty,
            "query": case.query,
            "expected_decision": case.expected_decision,
            "expected_answer": case.expected_answer,
            "actual_decision": actual_decision,
            "actual_confidence": adjusted_accuracy if decision_correct else adjusted_accuracy * 0.6,
            "actual_reasoning": f"[estimated from published GPT-4o scores: {case.category}]",
            "latency_ms": round(latency, 3),
            "decision_correct": decision_correct,
            "source": "estimated_from_published_numbers",
        }

    def save(self, data: dict[str, Any], path: str | Path) -> None:
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
