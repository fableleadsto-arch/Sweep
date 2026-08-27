"""
Scoring and Statistics — Accuracy, latency, confidence, significance.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from neural_eval.core import Result, BenchmarkSuite


def compute_stats(results: list[Result]) -> dict[str, Any]:
    """Compute comprehensive statistics for a set of results."""
    if not results:
        return {"error": "no results"}

    correct = [r for r in results if r.correct]
    wrong = [r for r in results if not r.correct]
    latencies = [r.latency_ms for r in results]
    confidences = [r.confidence for r in results]

    n = len(results)
    accuracy = len(correct) / n * 100

    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    std_lat = float(np.std(latencies))

    mean_conf = float(np.mean(confidences))

    ci_95 = 1.96 * math.sqrt(accuracy / 100 * (1 - accuracy / 100) / n) * 100

    correct_conf = [r.confidence for r in correct] if correct else [0]
    wrong_conf = [r.confidence for r in wrong] if wrong else [0]

    return {
        "n": n,
        "correct": len(correct),
        "wrong": len(wrong),
        "accuracy_pct": round(accuracy, 2),
        "ci_95_pct": round(ci_95, 2),
        "accuracy_range": f"{max(0, accuracy - ci_95):.2f}% - {min(100, accuracy + ci_95):.2f}%",
        "mean_latency_ms": round(mean_lat, 2),
        "median_latency_ms": round(median_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "mean_confidence": round(mean_conf, 4),
        "correct_mean_confidence": round(float(np.mean(correct_conf)), 4),
        "wrong_mean_confidence": round(float(np.mean(wrong_conf)), 4),
    }


def paired_accuracy_test(results_a: list[Result], results_b: list[Result]) -> dict[str, Any]:
    """Compare two systems on the same tasks (paired test)."""
    map_a = {r.task_id: r for r in results_a}
    map_b = {r.task_id: r for r in results_b}

    common_ids = sorted(set(map_a.keys()) & set(map_b.keys()))
    if len(common_ids) < 5:
        return {"error": "insufficient paired tasks", "n": len(common_ids)}

    both_correct = sum(1 for tid in common_ids if map_a[tid].correct and map_b[tid].correct)
    a_only = sum(1 for tid in common_ids if map_a[tid].correct and not map_b[tid].correct)
    b_only = sum(1 for tid in common_ids if not map_a[tid].correct and map_b[tid].correct)
    neither = sum(1 for tid in common_ids if not map_a[tid].correct and not map_b[tid].correct)

    n = len(common_ids)
    acc_a = sum(1 for tid in common_ids if map_a[tid].correct) / n * 100
    acc_b = sum(1 for tid in common_ids if map_b[tid].correct) / n * 100

    diff = acc_a - acc_b
    se = math.sqrt((acc_a / 100 * (1 - acc_a / 100) + acc_b / 100 * (1 - acc_b / 100)) / n) * 100

    z = diff / se if se > 0 else 0
    p_approx = 2 * (1 - _normal_cdf(abs(z))) if se > 0 else 1.0

    lat_a = [map_a[tid].latency_ms for tid in common_ids]
    lat_b = [map_b[tid].latency_ms for tid in common_ids]

    return {
        "n": n,
        "accuracy_a": round(acc_a, 2),
        "accuracy_b": round(acc_b, 2),
        "difference": round(diff, 2),
        "both_correct": both_correct,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
        "z_score": round(z, 3),
        "p_value_approx": round(p_approx, 4),
        "significant_005": p_approx < 0.05,
        "mean_latency_a_ms": round(float(np.mean(lat_a)), 2),
        "mean_latency_b_ms": round(float(np.mean(lat_b)), 2),
    }


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
