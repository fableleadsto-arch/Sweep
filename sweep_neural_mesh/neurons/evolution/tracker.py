"""
PerformanceTracker — monitors metrics and calibrates confidence.

Responsibilities:
  - Track accuracy, latency, and other metrics per core.
  - Compute Expected Calibration Error (ECE).
  - Identify performance degradation.
  - Suggest configuration changes.
"""
from __future__ import annotations

from typing import Any


class PerformanceTracker:
    """Tracks performance metrics and optimises configurations.

    Maintains running averages for accuracy/latency per core,
    records calibration data points, and suggests optimisations
    when metrics degrade.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, list[float]] = {}
        self._core_performance: dict[str, dict[str, float]] = {}
        self._calibration_data: list[tuple[float, bool]] = []

    def record_metric(self, name: str, value: float) -> None:
        self._metrics.setdefault(name, []).append(value)
        if len(self._metrics[name]) > 1000:
            self._metrics[name] = self._metrics[name][-1000:]

    def record_core_performance(self, core_id: str, accuracy: float, latency: float) -> None:
        if core_id not in self._core_performance:
            self._core_performance[core_id] = {"accuracy": 0.0, "latency": 0.0, "count": 0}
        perf = self._core_performance[core_id]
        perf["count"] += 1
        n = perf["count"]
        perf["accuracy"] = perf["accuracy"] * (n - 1) / n + accuracy / n
        perf["latency"] = perf["latency"] * (n - 1) / n + latency / n

    def record_calibration(self, confidence: float, correct: bool) -> None:
        self._calibration_data.append((confidence, correct))
        if len(self._calibration_data) > 1000:
            self._calibration_data = self._calibration_data[-1000:]

    def get_calibration_error(self) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        if not self._calibration_data:
            return 0.0
        bins, bin_size = 10, 0.1
        ece = 0.0
        for i in range(bins):
            lo, hi = i * bin_size, (i + 1) * bin_size
            bin_data = [(c, ok) for c, ok in self._calibration_data if lo <= c < hi]
            if bin_data:
                avg_conf = sum(c for c, _ in bin_data) / len(bin_data)
                avg_ok = sum(1 for _, ok in bin_data if ok) / len(bin_data)
                ece += abs(avg_conf - avg_ok) * len(bin_data) / len(self._calibration_data)
        return ece

    def suggest_optimizations(self) -> list[dict[str, Any]]:
        suggestions: list[dict[str, Any]] = []
        for core_id, perf in self._core_performance.items():
            if perf["accuracy"] < 0.7:
                suggestions.append({
                    "type": "low_accuracy",
                    "core": core_id,
                    "accuracy": perf["accuracy"],
                    "suggestion": f"Core {core_id} has low accuracy ({perf['accuracy']:.1%}).",
                })
            if perf["latency"] > 5.0:
                suggestions.append({
                    "type": "high_latency",
                    "core": core_id,
                    "latency": perf["latency"],
                    "suggestion": f"Core {core_id} has high latency ({perf['latency']:.1f}ms).",
                })
        ece = self.get_calibration_error()
        if ece > 0.1:
            suggestions.append({
                "type": "miscalibration",
                "ece": ece,
                "suggestion": f"Model is miscalibrated (ECE={ece:.3f}).",
            })
        return suggestions

    def get_stats(self) -> dict[str, Any]:
        return {
            "metrics": {
                name: {
                    "mean": sum(v) / len(v) if v else 0,
                    "min": min(v) if v else 0,
                    "max": max(v) if v else 0,
                }
                for name, v in self._metrics.items()
            },
            "cores": self._core_performance,
            "calibration_error": self.get_calibration_error(),
        }
