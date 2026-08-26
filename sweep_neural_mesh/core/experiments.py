"""
ExperimentalGraphRunner — runs configurable mesh experiments.

Researchers can define experimental mesh pipelines as JSON configs
without changing core source code. Each experiment records:
  - which nodes participate
  - parameters
  - input data
  - metrics
  - results

Structure:
    experiments/
        experiment_001/
            config.json
            results.json
            notes.md
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.engine import ExecutionEngine, ExecutionResult
from ..core.graph import MeshGraph
from ..core.node import NeuralNode
from ..core.packet import NeuralPacket
from ..core.router import ModelRouter
from ..fusion import FusionEngine
from ..fusion.confidence import ConfidenceEngine
from ..registry import ModelRegistry


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str = "unnamed_experiment"
    description: str = ""
    nodes: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    input_data: Any = None
    expected_output: Any = None
    metrics: list[str] = field(default_factory=lambda: ["accuracy", "latency", "confidence"])


@dataclass
class ExperimentResult:
    """Results from running an experiment."""
    config_name: str = ""
    success: bool = True
    output: Any = None
    metrics: dict[str, float] = field(default_factory=dict)
    execution_result: ExecutionResult | None = None
    duration_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class ExperimentalGraphRunner:
    """
    Runs configurable mesh experiments.

    Allows researchers to:
    1. Define experiments as configs
    2. Run them against registered nodes
    3. Compare results across experiments
    4. Store results for analysis
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.engine = ExecutionEngine()
        self.fusion = FusionEngine()
        self.confidence = ConfidenceEngine()
        self._results: list[ExperimentResult] = []
        self._experiments_dir = Path("experiments")

    def run_experiment(
        self,
        config: ExperimentConfig,
        node_map: dict[str, NeuralNode] | None = None,
    ) -> ExperimentResult:
        """Run a single experiment."""
        t0 = time.perf_counter()
        result = ExperimentResult(config_name=config.name)

        # Build graph from config
        graph = MeshGraph(name=f"experiment:{config.name}")
        nodes_used: dict[str, NeuralNode] = {}

        if node_map:
            # Use provided node map
            for node_id in config.nodes:
                if node_id in node_map:
                    node = node_map[node_id]
                    graph.add_node(node)
                    nodes_used[node_id] = node
        else:
            # Look up by capability in registry
            for cap in config.nodes:
                candidates = self.registry.find_capability(cap)
                if candidates:
                    node = candidates[0]
                    graph.add_node(node)
                    nodes_used[cap] = node
                else:
                    result.warnings.append(f"no node found for capability: {cap}")

        if not nodes_used:
            result.success = False
            result.warnings.append("no nodes available for experiment")
            result.duration_ms = (time.perf_counter() - t0) * 1000
            self._results.append(result)
            return result

        # Add edges (linear chain)
        node_list = list(nodes_used.values())
        for i in range(len(node_list) - 1):
            graph.add_edge(node_list[i], node_list[i + 1])

        # Execute
        initial_inputs = {}
        if config.input_data is not None and node_list:
            packet = NeuralPacket(data=config.input_data)
            initial_inputs[node_list[0].node_id] = packet

        exec_result = self.engine.execute(graph, initial_inputs)
        result.execution_result = exec_result

        # Compute metrics
        result.metrics["total_latency_ms"] = exec_result.total_latency_ms
        result.metrics["nodes_executed"] = exec_result.total_nodes_executed
        result.metrics["nodes_failed"] = exec_result.total_nodes_failed

        # Collect output
        if exec_result.output_packets:
            result.output = exec_result.output_packets[0].data
            result.metrics["confidence"] = exec_result.output_packets[0].confidence
        elif exec_result.node_results:
            first_result = next(iter(exec_result.node_results.values()))
            result.output = first_result.output

        # Check expected output
        if config.expected_output is not None:
            if result.output == config.expected_output:
                result.metrics["accuracy"] = 1.0
            else:
                result.metrics["accuracy"] = 0.0

        result.success = exec_result.success
        result.warnings.extend(exec_result.warnings)
        result.duration_ms = (time.perf_counter() - t0) * 1000

        self._results.append(result)
        return result

    def compare_experiments(
        self, results: list[ExperimentResult]
    ) -> dict[str, Any]:
        """Compare metrics across multiple experiment results."""
        if not results:
            return {}
        all_metrics: dict[str, list[float]] = {}
        for r in results:
            for k, v in r.metrics.items():
                if isinstance(v, (int, float)):
                    all_metrics.setdefault(k, []).append(v)

        comparison = {}
        for metric, values in all_metrics.items():
            comparison[metric] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "values": values,
            }
        return comparison

    def save_results(
        self, result: ExperimentResult, directory: str | None = None
    ) -> Path:
        """Save experiment results to disk."""
        exp_dir = Path(directory or self._experiments_dir) / result.config_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        results_file = exp_dir / "results.json"
        results_data = {
            "config_name": result.config_name,
            "success": result.success,
            "metrics": result.metrics,
            "duration_ms": result.duration_ms,
            "warnings": result.warnings,
            "output_type": type(result.output).__name__,
            "timestamp": result.timestamp,
        }
        results_file.write_text(json.dumps(results_data, indent=2))
        return results_file

    @property
    def results(self) -> list[ExperimentResult]:
        return list(self._results)

    def __repr__(self) -> str:
        return f"ExperimentalGraphRunner(results={len(self._results)})"
