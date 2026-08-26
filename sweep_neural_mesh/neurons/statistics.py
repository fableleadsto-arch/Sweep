"""
Statistical Analysis — hypothesis testing, confidence intervals, correlation.

Implements the statistical backbone for Sweep's evidence evaluation:

    Confidence Intervals:
        CI = mean ± z * (σ / √n)

    Hypothesis Testing:
        t = (x̄ - μ) / (s / √n)

    Correlation:
        r = Σ((x_i - x̄)(y_i - ȳ)) / √(Σ(x_i - x̄)² * Σ(y_i - ȳ)²)

    Effect Size (Cohen's d):
        d = (x̄₁ - x̄₂) / s_pooled

    Z-score:
        z = (x - μ) / σ

All computations are logged for traceability.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("sweep.math.statistics")

# Standard normal Z-values for common confidence levels
_Z_TABLE = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
    0.50: 0.674,
    0.80: 1.282,
}


@dataclass
class ConfidenceInterval:
    """A confidence interval for a population parameter."""
    mean: float
    lower: float
    upper: float
    confidence_level: float
    standard_error: float
    margin_of_error: float
    sample_size: int


@dataclass
class HypothesisTest:
    """Result of a statistical hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    effect_interpretation: str
    sample_size: int
    confidence_level: float
    detail: str = ""


@dataclass
class CorrelationResult:
    """Result of a correlation analysis."""
    r: float
    r_squared: float
    p_value: float
    n: int
    interpretation: str
    direction: str
    strength: str


@dataclass
class DistributionStats:
    """Comprehensive statistics for a data distribution."""
    count: int
    mean: float
    median: float
    mode: float | None
    variance: float
    std_dev: float
    std_error: float
    skewness: float
    kurtosis: float
    min_val: float
    max_val: float
    range_val: float
    iqr: float
    cv: float  # coefficient of variation


class StatisticsEngine:
    """
    Comprehensive statistical analysis engine.

    Provides all the statistical tools needed for Sweep's
    evidence evaluation and grading system.
    """

    def __init__(self) -> None:
        self._analyses: list[dict[str, Any]] = []
        logger.info("StatisticsEngine initialized")

    # ════════════════════════════════════════════════════════════════
    # BASIC STATISTICS
    # ════════════════════════════════════════════════════════════════

    def describe(self, data: list[float]) -> DistributionStats:
        """Compute comprehensive statistics for a data distribution."""
        if not data:
            return DistributionStats(0, 0, 0, None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        n = len(data)
        mean = sum(data) / n
        sorted_data = sorted(data)

        # Median
        if n % 2 == 1:
            median = sorted_data[n // 2]
        else:
            median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2.0

        # Mode (most frequent value)
        counts: dict[float, int] = {}
        for v in data:
            counts[v] = counts.get(v, 0) + 1
        max_count = max(counts.values())
        mode_candidates = [k for k, v in counts.items() if v == max_count]
        mode = mode_candidates[0] if len(mode_candidates) < len(counts) else None

        # Variance and standard deviation (sample)
        variance = sum((x - mean) ** 2 for x in data) / max(1, n - 1)
        std_dev = math.sqrt(variance)
        std_error = std_dev / math.sqrt(n)

        # Skewness (Fisher's)
        if std_dev > 0:
            skewness = sum(((x - mean) / std_dev) ** 3 for x in data) * n / max(1, (n - 1) * (n - 2))
        else:
            skewness = 0.0

        # Kurtosis (excess)
        if std_dev > 0:
            kurt = sum(((x - mean) / std_dev) ** 4 for x in data) / n - 3.0
        else:
            kurt = 0.0

        # Range and IQR
        min_val = sorted_data[0]
        max_val = sorted_data[-1]
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        iqr = sorted_data[q3_idx] - sorted_data[q1_idx]

        # Coefficient of variation
        cv = std_dev / abs(mean) if mean != 0 else 0.0

        stats = DistributionStats(
            count=n, mean=mean, median=median, mode=mode,
            variance=variance, std_dev=std_dev, std_error=std_error,
            skewness=skewness, kurtosis=kurt,
            min_val=min_val, max_val=max_val, range_val=max_val - min_val,
            iqr=iqr, cv=cv,
        )

        logger.info(
            f"describe(n={n}): mean={mean:.4f} std={std_dev:.4f} "
            f"skew={skewness:.4f} kurt={kurt:.4f}"
        )
        return stats

    def z_score(self, value: float, mean: float, std_dev: float) -> float:
        """Compute Z-score: z = (x - μ) / σ"""
        if std_dev == 0:
            return 0.0
        return (value - mean) / std_dev

    def percentile(self, data: list[float], p: float) -> float:
        """Compute the p-th percentile of data."""
        if not data:
            return 0.0
        sorted_d = sorted(data)
        k = (len(sorted_d) - 1) * (p / 100.0)
        f = int(math.floor(k))
        c = min(f + 1, len(sorted_d) - 1)
        d = k - f
        return sorted_d[f] + d * (sorted_d[c] - sorted_d[f])

    # ════════════════════════════════════════════════════════════════
    # CONFIDENCE INTERVALS
    # ════════════════════════════════════════════════════════════════

    def confidence_interval(
        self,
        data: list[float],
        confidence_level: float = 0.95,
    ) -> ConfidenceInterval:
        """
        Compute confidence interval for the mean.

        CI = mean ± z * (σ / √n)
        """
        n = len(data)
        if n == 0:
            return ConfidenceInterval(0, 0, 0, confidence_level, 0, 0, 0)

        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / max(1, n - 1)
        std_dev = math.sqrt(variance)
        std_error = std_dev / math.sqrt(n)

        z = _Z_TABLE.get(confidence_level, 1.960)
        margin = z * std_error

        ci = ConfidenceInterval(
            mean=mean, lower=mean - margin, upper=mean + margin,
            confidence_level=confidence_level, standard_error=std_error,
            margin_of_error=margin, sample_size=n,
        )
        logger.info(
            f"CI({confidence_level*100:.0f}%): [{ci.lower:.4f}, {ci.upper:.4f}] "
            f"mean={mean:.4f} n={n}"
        )
        return ci

    def proportion_ci(
        self,
        successes: int,
        total: int,
        confidence_level: float = 0.95,
    ) -> ConfidenceInterval:
        """Confidence interval for a proportion (Wald interval)."""
        if total == 0:
            return ConfidenceInterval(0, 0, 0, confidence_level, 0, 0, 0)

        p = successes / total
        se = math.sqrt(p * (1.0 - p) / total)
        z = _Z_TABLE.get(confidence_level, 1.960)
        margin = z * se

        return ConfidenceInterval(
            mean=p, lower=max(0.0, p - margin), upper=min(1.0, p + margin),
            confidence_level=confidence_level, standard_error=se,
            margin_of_error=margin, sample_size=total,
        )

    # ════════════════════════════════════════════════════════════════
    # HYPOTHESIS TESTING
    # ════════════════════════════════════════════════════════════════

    def one_sample_t_test(
        self,
        data: list[float],
        population_mean: float,
        alpha: float = 0.05,
    ) -> HypothesisTest:
        """One-sample t-test: is the sample mean different from a known value?"""
        n = len(data)
        if n < 2:
            return HypothesisTest("one_sample_t", 0.0, 1.0, False, 0.0, "none", n, 1-alpha)

        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / (n - 1)
        se = math.sqrt(variance / n)

        t_stat = (mean - population_mean) / max(1e-10, se)

        # Approximate p-value using t-distribution (df = n-1)
        df = n - 1
        p = self._t_to_p(abs(t_stat), df) * 2  # two-tailed

        # Effect size (Cohen's d)
        d = abs(mean - population_mean) / max(1e-10, math.sqrt(variance))

        test = HypothesisTest(
            test_name="one_sample_t",
            statistic=t_stat, p_value=p,
            significant=p < alpha, effect_size=d,
            effect_interpretation=self._interpret_effect(d),
            sample_size=n, confidence_level=1.0 - alpha,
            detail=f"t({df})={t_stat:.3f}, mean={mean:.4f}, μ₀={population_mean}",
        )
        self._log_test(test)
        return test

    def two_sample_t_test(
        self,
        sample1: list[float],
        sample2: list[float],
        alpha: float = 0.05,
    ) -> HypothesisTest:
        """Two-sample t-test: are the two sample means different?"""
        n1, n2 = len(sample1), len(sample2)
        if n1 < 2 or n2 < 2:
            return HypothesisTest("two_sample_t", 0.0, 1.0, False, 0.0, "none", n1+n2, 1-alpha)

        m1 = sum(sample1) / n1
        m2 = sum(sample2) / n2
        v1 = sum((x - m1) ** 2 for x in sample1) / (n1 - 1)
        v2 = sum((x - m2) ** 2 for x in sample2) / (n2 - 1)

        # Welch's t-test (unequal variances)
        se = math.sqrt(v1/n1 + v2/n2)
        t_stat = (m1 - m2) / max(1e-10, se)

        # Welch-Satterthwaite degrees of freedom
        num = (v1/n1 + v2/n2) ** 2
        denom = (v1/n1)**2 / (n1-1) + (v2/n2)**2 / (n2-1)
        df = num / max(1.0, denom)

        p = self._t_to_p(abs(t_stat), df) * 2

        # Pooled standard deviation for effect size
        s_pooled = math.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
        d = abs(m1 - m2) / max(1e-10, s_pooled)

        test = HypothesisTest(
            test_name="two_sample_t",
            statistic=t_stat, p_value=p,
            significant=p < alpha, effect_size=d,
            effect_interpretation=self._interpret_effect(d),
            sample_size=n1+n2, confidence_level=1.0 - alpha,
            detail=f"t({df:.1f})={t_stat:.3f}, m1={m1:.4f}, m2={m2:.4f}",
        )
        self._log_test(test)
        return test

    def one_proportion_test(
        self,
        successes: int,
        total: int,
        null_proportion: float = 0.5,
        alpha: float = 0.05,
    ) -> HypothesisTest:
        """One-proportion z-test: is the observed proportion different from expected?"""
        if total == 0:
            return HypothesisTest("one_proportion_z", 0.0, 1.0, False, 0.0, "none", 0, 1-alpha)

        p_hat = successes / total
        se = math.sqrt(null_proportion * (1 - null_proportion) / total)
        z = (p_hat - null_proportion) / max(1e-10, se)

        # Two-tailed p-value from normal approximation
        p = 2.0 * (1.0 - self._normal_cdf(abs(z)))

        d = abs(p_hat - null_proportion) / max(1e-10, se)

        test = HypothesisTest(
            test_name="one_proportion_z",
            statistic=z, p_value=p,
            significant=p < alpha, effect_size=d,
            effect_interpretation=self._interpret_effect(d),
            sample_size=total, confidence_level=1.0 - alpha,
            detail=f"z={z:.3f}, p_hat={p_hat:.4f}, p₀={null_proportion}",
        )
        self._log_test(test)
        return test

    # ════════════════════════════════════════════════════════════════
    # CORRELATION
    # ════════════════════════════════════════════════════════════════

    def pearson_correlation(
        self, x: list[float], y: list[float],
    ) -> CorrelationResult:
        """Pearson correlation coefficient."""
        n = min(len(x), len(y))
        if n < 3:
            return CorrelationResult(0.0, 0.0, 1.0, n, "none", "none", "none")

        mx = sum(x[:n]) / n
        my = sum(y[:n]) / n

        num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
        den_x = math.sqrt(sum((x[i] - mx) ** 2 for i in range(n)))
        den_y = math.sqrt(sum((y[i] - my) ** 2 for i in range(n)))

        r = num / max(1e-10, den_x * den_y)
        r2 = r * r

        # Approximate p-value using t-distribution
        if abs(r) >= 1.0:
            p = 0.0
        else:
            t = r * math.sqrt((n - 2) / max(1e-10, 1.0 - r * r))
            p = self._t_to_p(abs(t), n - 2) * 2

        result = CorrelationResult(
            r=r, r_squared=r2, p_value=p, n=n,
            interpretation=f"R²={r2:.4f} ({r2*100:.1f}% variance explained)",
            direction="positive" if r > 0 else "negative",
            strength=self._interpret_correlation(abs(r)),
        )
        logger.info(f"Pearson r={r:.4f} R²={r2:.4f} p={p:.4f} n={n}")
        return result

    def spearman_correlation(
        self, x: list[float], y: list[float],
    ) -> CorrelationResult:
        """Spearman rank correlation (non-parametric)."""
        n = min(len(x), len(y))
        if n < 3:
            return CorrelationResult(0.0, 0.0, 1.0, n, "none", "none", "none")

        # Rank the data
        rx = self._rank(x[:n])
        ry = self._rank(y[:n])

        return self.pearson_correlation(rx, ry)

    # ════════════════════════════════════════════════════════════════
    # INTERNAL HELPERS
    # ════════════════════════════════════════════════════════════════

    def _rank(self, data: list[float]) -> list[float]:
        """Assign ranks to data (average rank for ties)."""
        indexed = sorted(enumerate(data), key=lambda x: x[1])
        ranks = [0.0] * len(data)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1.0
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF using error function approximation."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _t_to_p(self, t: float, df: float) -> float:
        """Approximate two-tailed p-value from t-distribution using Hill's algorithm."""
        if df <= 0:
            return 1.0
        x = df / (df + t * t)
        # Incomplete beta function approximation
        if x >= 1.0:
            return 1.0
        if x <= 0.0:
            return 0.0

        # Simple approximation for large df
        if df > 30:
            return 2.0 * (1.0 - self._normal_cdf(t))

        # Beta function approximation for small df
        a = df / 2.0
        b = 0.5
        return self._incomplete_beta(x, a, b)

    def _incomplete_beta(self, x: float, a: float, b: float) -> float:
        """Incomplete beta function approximation (regularized)."""
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0

        # Continued fraction expansion (Lentz's method)
        max_iter = 200
        eps = 1e-10

        # Use series expansion for small x
        if x < (a + 1) / (a + b + 2):
            return self._beta_series(x, a, b, max_iter, eps)
        else:
            return 1.0 - self._beta_series(1 - x, b, a, max_iter, eps)

    def _beta_series(self, x: float, a: float, b: float, max_iter: int, eps: float) -> float:
        """Series expansion for incomplete beta."""
        prefix = math.gamma(a + b) / (math.gamma(a) * math.gamma(b))
        prefix *= (x ** a) / a

        total = 1.0
        term = 1.0
        for n in range(1, max_iter):
            term *= x * (a + b + n - 1) / ((a + n) * (a + b + n))
            total += term
            if abs(term) < eps:
                break

        return prefix * total

    def _interpret_effect(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation strength."""
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "weak"
        elif r < 0.5:
            return "moderate"
        elif r < 0.7:
            return "strong"
        else:
            return "very strong"

    def _log_test(self, test: HypothesisTest) -> None:
        """Log a hypothesis test result."""
        sig = "***" if test.p_value < 0.001 else "**" if test.p_value < 0.01 else "*" if test.p_value < 0.05 else "ns"
        logger.info(
            f"{test.test_name}: stat={test.statistic:.3f} p={test.p_value:.4f} "
            f"{sig} effect={test.effect_interpretation} n={test.sample_size}"
        )
