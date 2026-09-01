"""
Benchmark Metrics — statistical analysis for benchmark evaluation.

Implements:
- Brier score
- Expected Calibration Error (ECE)
- Confidence intervals (bootstrap + normal approximation)
- Effect sizes (Cohen's h, Cohen's d)
- Statistical significance (z-test for proportions, McNemar's test)
- Calibration curves
- Confusion matrix metrics (precision, recall, F1)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StatisticalResult:
    """Result of a statistical test."""
    metric_name: str
    value: float
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    p_value: float = 1.0
    significant: bool = False
    interpretation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric_name,
            "value": self.value,
            "confidence_interval": list(self.confidence_interval),
            "p_value": self.p_value,
            "significant": self.significant,
            "interpretation": self.interpretation,
        }


class BenchmarkMetrics:
    """Comprehensive metrics for benchmark evaluation."""

    # ── Brier Score ──

    @staticmethod
    def brier_score(confidences: list[float], outcomes: list[float]) -> float:
        """
        Compute Brier score for probabilistic predictions.

        Brier score = mean((confidence - outcome)^2)
        Range: [0, 1], lower is better.
        0 = perfect calibration, 1 = worst possible.
        """
        if len(confidences) != len(outcomes) or not confidences:
            return 0.0
        total = sum((c - o) ** 2 for c, o in zip(confidences, outcomes))
        return total / len(confidences)

    # ── Expected Calibration Error ──

    @staticmethod
    def expected_calibration_error(
        confidences: list[float],
        outcomes: list[float],
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Measures how well confidence matches actual accuracy across bins.
        Lower is better. 0 = perfectly calibrated.
        """
        if not confidences or not outcomes:
            return 0.0

        n = len(confidences)
        bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]

        for c, o in zip(confidences, outcomes):
            bin_idx = min(int(c * n_bins), n_bins - 1)
            bins[bin_idx].append((c, o))

        ece = 0.0
        for b in bins:
            if b:
                avg_conf = sum(c for c, _ in b) / len(b)
                avg_acc = sum(o for _, o in b) / len(b)
                ece += (len(b) / n) * abs(avg_conf - avg_acc)
        return ece

    @staticmethod
    def calibration_curve(
        confidences: list[float],
        outcomes: list[float],
        n_bins: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Generate calibration curve data.

        Returns list of bins with confidence, accuracy, and count.
        """
        if not confidences or not outcomes:
            return []

        n = len(confidences)
        bins: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]

        for c, o in zip(confidences, outcomes):
            bin_idx = min(int(c * n_bins), n_bins - 1)
            bins[bin_idx].append((c, o))

        curve = []
        for i, b in enumerate(bins):
            if b:
                avg_conf = sum(c for c, _ in b) / len(b)
                avg_acc = sum(o for _, o in b) / len(b)
                curve.append({
                    "bin_start": i / n_bins,
                    "bin_end": (i + 1) / n_bins,
                    "mean_confidence": avg_conf,
                    "mean_accuracy": avg_acc,
                    "count": len(b),
                    "fraction": len(b) / n,
                })
        return curve

    # ── Confidence Intervals ──

    @staticmethod
    def proportion_ci(
        successes: int,
        total: int,
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """
        Compute confidence interval for a proportion using normal approximation.

        Returns (lower, upper) bounds.
        """
        if total == 0:
            return (0.0, 1.0)
        p = successes / total
        # Wilson score interval (better for extreme proportions)
        z = BenchmarkMetrics._z_score(confidence)
        denom = 1 + z ** 2 / total
        center = (p + z ** 2 / (2 * total)) / denom
        spread = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * total)) / total) / denom
        return (max(0.0, center - spread), min(1.0, center + spread))

    @staticmethod
    def bootstrap_ci(
        values: list[float],
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> tuple[float, float]:
        """
        Compute bootstrap confidence interval for the mean.
        """
        if not values:
            return (0.0, 0.0)
        rng = random.Random(seed)
        means = []
        n = len(values)
        for _ in range(n_bootstrap):
            sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
            means.append(sum(sample) / len(sample))
        means.sort()
        alpha = (1 - confidence) / 2
        lower_idx = int(alpha * n_bootstrap)
        upper_idx = int((1 - alpha) * n_bootstrap) - 1
        return (means[lower_idx], means[upper_idx])

    @staticmethod
    def _z_score(confidence: float) -> float:
        """Get z-score for a given confidence level (two-tailed)."""
        # Common z-scores
        z_map = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }
        return z_map.get(confidence, 1.96)

    # ── Effect Sizes ──

    @staticmethod
    def cohens_h(p1: float, p2: float) -> float:
        """
        Compute Cohen's h for comparing two proportions.

        Small: 0.2, Medium: 0.5, Large: 0.8
        """
        return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

    @staticmethod
    def cohens_d(group1: list[float], group2: list[float]) -> float:
        """
        Compute Cohen's d for comparing two groups.

        Small: 0.2, Medium: 0.5, Large: 0.8
        """
        if not group1 or not group2:
            return 0.0
        mean1 = sum(group1) / len(group1)
        mean2 = sum(group2) / len(group2)
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0
        var1 = sum((x - mean1) ** 2 for x in group1) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in group2) / (n2 - 1)
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return 0.0
        return (mean1 - mean2) / pooled_std

    # ── Statistical Significance ──

    @staticmethod
    def z_test_proportions(
        successes_a: int, total_a: int,
        successes_b: int, total_b: int,
        significance_level: float = 0.05,
    ) -> StatisticalResult:
        """
        Two-proportion z-test.

        Tests whether the difference between two proportions is significant.
        """
        if total_a == 0 or total_b == 0:
            return StatisticalResult(
                metric_name="z_test_proportions",
                value=0.0,
                interpretation="Insufficient data for comparison",
            )

        p_a = successes_a / total_a
        p_b = successes_b / total_b
        p_pool = (successes_a + successes_b) / (total_a + total_b)

        se = math.sqrt(p_pool * (1 - p_pool) * (1 / total_a + 1 / total_b))
        if se == 0:
            return StatisticalResult(
                metric_name="z_test_proportions",
                value=0.0,
                interpretation="No variation — proportions are identical",
            )

        z = (p_a - p_b) / se
        # Two-tailed p-value using error function
        from math import erf, sqrt
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
        significant = p_value < significance_level

        # Effect size
        h = BenchmarkMetrics.cohens_h(p_a, p_b)

        # CI for the difference
        se_diff = math.sqrt(
            p_a * (1 - p_a) / total_a + p_b * (1 - p_b) / total_b
        )
        z_crit = BenchmarkMetrics._z_score(1 - significance_level)
        diff = p_a - p_b
        ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

        if significant:
            direction = "higher" if diff > 0 else "lower"
            interp = (
                f"Model A scored {abs(diff)*100:.1f} percentage points {direction} "
                f"than Model B; the difference was established as statistically "
                f"significant (p={p_value:.4f}, Cohen's h={h:.3f})."
            )
        else:
            interp = (
                f"Model A scored {abs(diff)*100:.1f} percentage points "
                f"{'higher' if diff > 0 else 'lower'} than Model B; "
                f"the difference was not established as statistically significant "
                f"(p={p_value:.4f})."
            )

        return StatisticalResult(
            metric_name="z_test_proportions",
            value=z,
            confidence_interval=ci,
            p_value=p_value,
            significant=significant,
            interpretation=interp,
        )

    @staticmethod
    def mcnemar_test(
        correct_a_only: int,
        correct_b_only: int,
        both_correct: int,
        both_wrong: int,
        significance_level: float = 0.05,
    ) -> StatisticalResult:
        """
        McNemar's test for paired binary outcomes.

        Tests whether two classifiers disagree significantly.
        """
        n = correct_a_only + correct_b_only + both_correct + both_wrong
        if n == 0:
            return StatisticalResult(
                metric_name="mcnemar_test",
                value=0.0,
                interpretation="No data for McNemar's test",
            )

        # McNemar's statistic with continuity correction
        b = correct_a_only  # A correct, B wrong
        c = correct_b_only  # B correct, A wrong
        if b + c == 0:
            return StatisticalResult(
                metric_name="mcnemar_test",
                value=0.0,
                interpretation="Models agree on all tasks — no disagreement to test",
            )

        chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0

        # p-value from chi-squared with 1 df
        from math import erf, sqrt
        p_value = 1 - erf(math.sqrt(chi2 / 2)) if chi2 > 0 else 1.0
        significant = p_value < significance_level

        interp = (
            f"McNemar's test: chi2={chi2:.3f}, p={p_value:.4f}. "
            f"{'Significant' if significant else 'Not significant'} disagreement."
        )

        return StatisticalResult(
            metric_name="mcnemar_test",
            value=chi2,
            p_value=p_value,
            significant=significant,
            interpretation=interp,
        )

    # ── Classification Metrics ──

    @staticmethod
    def precision_recall_f1(
        true_positives: int,
        false_positives: int,
        false_negatives: int,
    ) -> dict[str, float]:
        """Compute precision, recall, and F1 score."""
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # ── Abstention Metrics ──

    @staticmethod
    def abstention_metrics(
        predictions: list[str],
        ground_truth: list[str],
        abstention_token: str = "unknown",
    ) -> dict[str, float]:
        """
        Compute abstention-specific metrics.

        Measures:
        - abstention_rate: fraction of tasks where model abstained
        - selective_accuracy: accuracy on non-abstained tasks
        - false_certainty_rate: incorrect answers given with confidence
        """
        n = len(predictions)
        if n == 0:
            return {"abstention_rate": 0, "selective_accuracy": 0, "false_certainty_rate": 0}

        abstained = 0
        correct_abstained = 0
        correct_non_abstained = 0
        total_non_abstained = 0
        wrong_non_abstained = 0

        for pred, gt in zip(predictions, ground_truth):
            if pred.lower().strip() == abstention_token.lower():
                abstained += 1
                # Check if abstention was correct (i.e., ground truth is also unknown)
                if gt.lower().strip() in (abstention_token.lower(), "unknown", "insufficient"):
                    correct_abstained += 1
            else:
                total_non_abstained += 1
                if pred.lower().strip() == gt.lower().strip():
                    correct_non_abstained += 1
                else:
                    wrong_non_abstained += 1

        return {
            "abstention_rate": abstained / n,
            "selective_accuracy": correct_non_abstained / total_non_abstained if total_non_abstained > 0 else 0.0,
            "false_certainty_rate": wrong_non_abstained / n,
            "correct_abstention_rate": correct_abstained / abstained if abstained > 0 else 0.0,
        }

    # ── Hallucination Rate ──

    @staticmethod
    def hallucination_rate(
        predictions: list[str],
        ground_truth: list[str],
        evidence_texts: list[list[str]] | None = None,
    ) -> float:
        """
        Estimate hallucination rate.

        A hallucination is when the model confidently produces an answer
        that contradicts the evidence or ground truth.
        """
        if not predictions:
            return 0.0

        hallucinations = 0
        for pred, gt in zip(predictions, ground_truth):
            pred_lower = pred.lower().strip()
            gt_lower = gt.lower().strip()

            # Skip abstentions
            if pred_lower in ("unknown", "insufficient", "insufficient_evidence"):
                continue

            # If the answer is wrong and the model didn't abstain
            if pred_lower != gt_lower and pred_lower:
                hallucinations += 1

        return hallucinations / len(predictions)

    # ── Composite Report ──

    @staticmethod
    def compute_all_metrics(
        correct: int,
        total: int,
        confidences: list[float] | None = None,
        predictions: list[str] | None = None,
        ground_truth: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compute a comprehensive metrics report."""
        accuracy = correct / total if total > 0 else 0.0
        ci = BenchmarkMetrics.proportion_ci(correct, total)

        report: dict[str, Any] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "confidence_interval_95": list(ci),
        }

        if confidences and len(confidences) == total:
            outcomes = [1.0] * correct + [0.0] * (total - correct)
            # Align outcomes with confidences if possible
            if len(outcomes) == len(confidences):
                report["brier_score"] = BenchmarkMetrics.brier_score(confidences, outcomes)
                report["ece"] = BenchmarkMetrics.expected_calibration_error(confidences, outcomes)
                report["calibration_curve"] = BenchmarkMetrics.calibration_curve(confidences, outcomes)

        if predictions and ground_truth:
            abstention = BenchmarkMetrics.abstention_metrics(predictions, ground_truth)
            report["abstention"] = abstention
            report["hallucination_rate"] = BenchmarkMetrics.hallucination_rate(predictions, ground_truth)

        return report
