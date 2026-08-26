"""
Benchmarking — automated performance evaluation for Mesh nodes and pipelines.

Measures:
  - Per-node: latency, throughput, memory, confidence calibration
  - Per-pipeline: end-to-end latency, accuracy, fusion quality
  - Cross-model: single vs ensemble vs fusion comparison
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.node import NeuralNode
from ..core.packet import NeuralPacket


@dataclass
class NodeBenchmark:
    """Benchmark result for a single node."""
    node_name: str
    node_id: str
    iterations: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    throughput_per_sec: float = 0.0
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    memory_estimate_mb: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class PipelineBenchmark:
    """Benchmark result for a full pipeline."""
    pipeline_name: str
    nodes_benchmarked: int = 0
    total_iterations: int = 0
    avg_total_latency_ms: float = 0.0
    avg_per_node_latency_ms: float = 0.0
    throughput_per_sec: float = 0.0
    success_rate: float = 0.0
    fusion_quality: float = 0.0
    node_benchmarks: list[NodeBenchmark] = field(default_factory=list)


class BenchmarkSuite:
    """
    Automated benchmark suite for the Neural Mesh.

    Runs repeated inferences on nodes and pipelines, collecting
    latency distributions, success rates, and confidence metrics.
    """

    def __init__(self, warmup_iterations: int = 3) -> None:
        self.warmup_iterations = warmup_iterations
        self._node_benchmarks: list[NodeBenchmark] = []
        self._pipeline_benchmarks: list[PipelineBenchmark] = []

    def benchmark_node(
        self,
        node: NeuralNode,
        test_inputs: list[Any],
        iterations: int = 50,
    ) -> NodeBenchmark:
        """Benchmark a single node with repeated inference."""
        bench = NodeBenchmark(
            node_name=node.name,
            node_id=node.node_id,
        )

        # Warmup
        for inp in test_inputs[:self.warmup_iterations]:
            node.execute(inp)

        # Benchmark
        latencies: list[float] = []
        successes = 0
        confidences: list[float] = []
        errors: list[str] = []

        for _ in range(iterations):
            for inp in test_inputs:
                result = node.execute(inp)
                latencies.append(result.latency_ms)
                if result.success:
                    successes += 1
                    if result.confidence > 0:
                        confidences.append(result.confidence)
                else:
                    if result.error:
                        errors.append(result.error)

        total_runs = iterations * len(test_inputs)
        bench.iterations = total_runs
        bench.success_rate = successes / total_runs if total_runs > 0 else 0
        bench.errors = list(set(errors))

        if latencies:
            sorted_lat = sorted(latencies)
            bench.avg_latency_ms = statistics.mean(latencies)
            bench.min_latency_ms = min(latencies)
            bench.max_latency_ms = max(latencies)
            bench.p50_latency_ms = sorted_lat[len(sorted_lat) // 2]
            bench.p95_latency_ms = sorted_lat[int(len(sorted_lat) * 0.95)]
            bench.p99_latency_ms = sorted_lat[int(len(sorted_lat) * 0.99)]
            if bench.avg_latency_ms > 0:
                bench.throughput_per_sec = 1000.0 / bench.avg_latency_ms

        if confidences:
            bench.avg_confidence = statistics.mean(confidences)

        self._node_benchmarks.append(bench)
        return bench

    def benchmark_pipeline(
        self,
        nodes: list[NeuralNode],
        test_inputs: list[Any],
        iterations: int = 20,
        name: str = "unnamed_pipeline",
    ) -> PipelineBenchmark:
        """Benchmark a pipeline of nodes in sequence."""
        pipe_bench = PipelineBenchmark(pipeline_name=name)

        # Benchmark each node individually
        for node in nodes:
            node_bench = self.benchmark_node(node, test_inputs, iterations)
            pipe_bench.node_benchmarks.append(node_bench)

        pipe_bench.nodes_benchmarked = len(nodes)
        pipe_bench.total_iterations = iterations * len(test_inputs)

        # Aggregate
        all_latencies = []
        total_successes = 0
        for nb in pipe_bench.node_benchmarks:
            all_latencies.append(nb.avg_latency_ms)
            total_successes += int(nb.success_rate * nb.iterations)

        total_runs = sum(nb.iterations for nb in pipe_bench.node_benchmarks)
        pipe_bench.avg_total_latency_ms = sum(all_latencies)
        pipe_bench.avg_per_node_latency_ms = (
            statistics.mean(all_latencies) if all_latencies else 0
        )
        pipe_bench.success_rate = total_successes / total_runs if total_runs > 0 else 0

        if pipe_bench.avg_total_latency_ms > 0:
            pipe_bench.throughput_per_sec = 1000.0 / pipe_bench.avg_total_latency_ms

        self._pipeline_benchmarks.append(pipe_bench)
        return pipe_bench

    def compare_single_vs_ensemble(
        self,
        single_node: NeuralNode,
        ensemble_nodes: list[NeuralNode],
        test_inputs: list[Any],
        iterations: int = 20,
    ) -> dict[str, Any]:
        """Compare single model vs ensemble performance."""
        single_bench = self.benchmark_node(single_node, test_inputs, iterations)

        ensemble_latencies = []
        for node in ensemble_nodes:
            bench = self.benchmark_node(node, test_inputs, iterations)
            ensemble_latencies.append(bench.avg_latency_ms)

        avg_ensemble_latency = (
            statistics.mean(ensemble_latencies) if ensemble_latencies else 0
        )

        return {
            "single": {
                "node": single_node.name,
                "avg_latency_ms": single_bench.avg_latency_ms,
                "success_rate": single_bench.success_rate,
                "throughput": single_bench.throughput_per_sec,
            },
            "ensemble": {
                "nodes": [n.name for n in ensemble_nodes],
                "avg_latency_ms": avg_ensemble_latency,
                "total_latency_ms": sum(ensemble_latencies),
                "node_count": len(ensemble_nodes),
            },
            "comparison": {
                "latency_ratio": avg_ensemble_latency / single_bench.avg_latency_ms
                if single_bench.avg_latency_ms > 0 else 0,
                "ensemble_is_slower": avg_ensemble_latency > single_bench.avg_latency_ms,
            },
        }

    @property
    def node_benchmarks(self) -> list[NodeBenchmark]:
        return list(self._node_benchmarks)

    @property
    def pipeline_benchmarks(self) -> list[PipelineBenchmark]:
        return list(self._pipeline_benchmarks)

    def summary(self) -> dict[str, Any]:
        return {
            "node_benchmarks": len(self._node_benchmarks),
            "pipeline_benchmarks": len(self._pipeline_benchmarks),
            "avg_node_latency": (
                statistics.mean([b.avg_latency_ms for b in self._node_benchmarks])
                if self._node_benchmarks else 0
            ),
        }

    def __repr__(self) -> str:
        return (
            f"BenchmarkSuite(nodes={len(self._node_benchmarks)}, "
            f"pipelines={len(self._pipeline_benchmarks)})"
        )
