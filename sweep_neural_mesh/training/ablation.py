"""
Ablation Testing Framework — §20

For every major component, measure:
    FULL SYSTEM
    vs:
    - without retrieval
    - without memory
    - without vision
    - without audio
    - without tool use
    - without reasoning layer
    - without embeddings
    - without verification

Determine which components actually improve performance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AblationConfig:
    """Configuration for an ablation run."""
    component_name: str
    enabled: bool
    description: str


@dataclass
class AblationResult:
    """Result of testing with a component ablated."""
    component_name: str
    enabled: bool
    accuracy: float
    latency_ms: float
    task_count: int
    details: dict[str, Any] = field(default_factory=dict)


class AblationStudy:
    """
    §20: Measures the contribution of each component by disabling
    them one at a time and measuring performance degradation.

    Component → benchmark impact analysis.
    """

    # Default components to ablate
    DEFAULT_COMPONENTS = [
        ("reasoning", True, "Core reasoning engine (cortex + task handlers)"),
        ("multi_core", True, "Multi-core parallel processing (5 cores)"),
        ("fast_path", True, "Fast-path early exit for simple queries"),
        ("evidence_pipeline", True, "Evidence cross-referencing and corroboration"),
        ("human_reasoning", True, "Human-like reasoning modules (common sense, ToM, etc.)"),
        ("intelligence", True, "Intelligence gathering pipeline"),
        ("web_scraper", True, "Web scraping and live knowledge retrieval"),
        ("self_evolution", True, "Self-evolution and learning system"),
        ("task_handlers", True, "Typed task handlers (logic, math, evidence, etc.)"),
        ("knowledge_base", True, "Factual and temporal knowledge bases"),
    ]

    def __init__(self) -> None:
        self._components: list[AblationConfig] = []
        self._results: list[AblationResult] = []
        self._baseline_result: AblationResult | None = None

    def set_components(self, components: list[tuple[str, bool, str]] | None = None) -> None:
        """Set the components to ablate."""
        components = components or self.DEFAULT_COMPONENTS
        self._components = [
            AblationConfig(component_name=name, enabled=enabled, description=desc)
            for name, enabled, desc in components
        ]

    def run_full_system(
        self,
        test_fn: Callable[[dict[str, bool]], tuple[float, float, int]],
    ) -> AblationResult:
        """
        Run the full system baseline.

        Args:
            test_fn: Function that takes {component: enabled} dict
                     and returns (accuracy, latency_ms, task_count).

        Returns:
            Baseline AblationResult.
        """
        if not self._components:
            self.set_components()

        enabled_map = {c.component_name: c.enabled for c in self._components}
        accuracy, latency, count = test_fn(enabled_map)
        self._baseline_result = AblationResult(
            component_name="FULL_SYSTEM",
            enabled=True,
            accuracy=accuracy,
            latency_ms=latency,
            task_count=count,
        )
        self._results.append(self._baseline_result)
        return self._baseline_result

    def run_ablation(
        self,
        component_name: str,
        test_fn: Callable[[dict[str, bool]], tuple[float, float, int]],
    ) -> AblationResult:
        """
        Run with one component disabled.

        Args:
            component_name: Name of the component to disable.
            test_fn: Function that takes {component: enabled} dict.

        Returns:
            AblationResult with the component disabled.
        """
        enabled_map = {c.component_name: c.enabled for c in self._components}
        enabled_map[component_name] = False

        accuracy, latency, count = test_fn(enabled_map)
        result = AblationResult(
            component_name=component_name,
            enabled=False,
            accuracy=accuracy,
            latency_ms=latency,
            task_count=count,
        )
        self._results.append(result)
        return result

    def run_all_ablations(
        self,
        test_fn: Callable[[dict[str, bool]], tuple[float, float, int]],
    ) -> list[AblationResult]:
        """
        Run ablation for every component.

        Args:
            test_fn: Function that takes {component: enabled} dict.

        Returns:
            List of AblationResults (one per component ablated).
        """
        if self._baseline_result is None:
            self.run_full_system(test_fn)

        results = []
        for comp in self._components:
            if comp.enabled:
                result = self.run_ablation(comp.component_name, test_fn)
                results.append(result)

        return results

    def compute_impact(self) -> list[dict[str, Any]]:
        """
        Compute the impact of each component.

        Returns a sorted list of component impacts, from most to least impactful.
        """
        if self._baseline_result is None:
            return []

        baseline_acc = self._baseline_result.accuracy
        baseline_lat = self._baseline_result.latency_ms

        impacts = []
        for result in self._results:
            if result.component_name == "FULL_SYSTEM":
                continue

            acc_delta = baseline_acc - result.accuracy
            lat_delta = result.latency_ms - baseline_lat

            impacts.append({
                "component": result.component_name,
                "accuracy_without": result.accuracy,
                "accuracy_drop": round(acc_delta, 4),
                "accuracy_drop_pct": round(acc_delta / max(baseline_acc, 0.001) * 100, 1),
                "latency_without": result.latency_ms,
                "latency_change_ms": round(lat_delta, 1),
                "importance": "critical" if acc_delta > 0.1 else "high" if acc_delta > 0.05 else "moderate" if acc_delta > 0.01 else "low",
            })

        impacts.sort(key=lambda x: x["accuracy_drop"], reverse=True)
        return impacts

    def summary(self) -> dict[str, Any]:
        """Generate a summary of the ablation study."""
        impacts = self.compute_impact()
        baseline = self._baseline_result

        critical = [i for i in impacts if i["importance"] == "critical"]
        high = [i for i in impacts if i["importance"] == "high"]
        moderate = [i for i in impacts if i["importance"] == "moderate"]
        low = [i for i in impacts if i["importance"] == "low"]

        return {
            "baseline_accuracy": baseline.accuracy if baseline else 0,
            "baseline_latency_ms": baseline.latency_ms if baseline else 0,
            "components_tested": len(impacts),
            "critical_components": [i["component"] for i in critical],
            "high_impact_components": [i["component"] for i in high],
            "moderate_impact_components": [i["component"] for i in moderate],
            "low_impact_components": [i["component"] for i in low],
            "impacts": impacts,
        }
