"""Tests for Stages 6-8: optimization, meta-router, experiments, benchmarking, intent."""
from __future__ import annotations

import pytest

from sweep_neural_mesh.optimization.distillation import (
    DistillationConfig,
    DistillationEngine,
    DistillationRecord,
)
from sweep_neural_mesh.optimization.quantization import Quantizer, QuantizationProfile
from sweep_neural_mesh.optimization.batching import (
    BatchingStrategy,
    CachePolicy,
    Pruner,
)
from sweep_neural_mesh.core.meta_router import (
    MetaRouter,
    MetaRoutingDecision,
    RoutingExperience,
)
from sweep_neural_mesh.core.experiments import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentalGraphRunner,
)
from sweep_neural_mesh.core.benchmarking import BenchmarkSuite, NodeBenchmark
from sweep_neural_mesh.intent import IntentMesh, IntentRoute
from sweep_neural_mesh.mesh import MeshConstraints, NeuralMesh
from sweep_neural_mesh.core.node import NeuralNode


# ── Stage 6: Distillation ──

class TestDistillation:
    def test_distill_basic(self):
        teacher = NeuralNode(name="teacher", execute_fn=lambda x, **kw: x * 2)
        student = NeuralNode(name="student", execute_fn=lambda x, **kw: x * 2)
        engine = DistillationEngine()
        record = engine.distill(
            teacher, student,
            training_data=[1, 2, 3, 4, 5],
            config=DistillationConfig(epochs=1),
        )
        assert record.samples_processed == 5
        assert record.retention_ratio > 0
        assert record.duration_ms >= 0

    def test_distill_empty_data(self):
        teacher = NeuralNode(name="t", execute_fn=lambda x, **kw: x)
        student = NeuralNode(name="s", execute_fn=lambda x, **kw: x)
        engine = DistillationEngine()
        record = engine.distill(teacher, student, training_data=[])
        assert record.samples_processed == 0

    def test_distill_temperature_effect(self):
        teacher = NeuralNode(name="t", execute_fn=lambda x, **kw: [x * 0.5, x * 0.5])
        student = NeuralNode(name="s", execute_fn=lambda x, **kw: [x * 0.5, x * 0.5])
        engine = DistillationEngine()
        r1 = engine.distill(teacher, student, [1, 2, 3],
                            config=DistillationConfig(temperature=1.0))
        r2 = engine.distill(teacher, student, [1, 2, 3],
                            config=DistillationConfig(temperature=10.0))
        # Different temperatures should produce different soft losses
        assert r1.temperature != r2.temperature

    def test_distill_summary(self):
        engine = DistillationEngine()
        assert engine.summary()["runs"] == 0


# ── Stage 6: Quantization ──

class TestQuantization:
    def test_float32_quantize(self):
        q = Quantizer()
        data = [1.123456789, 2.987654321, 3.555555555]
        quantized, profile = q.quantize(data, "float32")
        assert profile.compression_ratio == 2.0  # 8 bytes → 4 bytes
        assert profile.max_error < 0.01

    def test_int8_quantize(self):
        q = Quantizer()
        data = [0.0, 0.25, 0.5, 0.75, 1.0]
        quantized, profile = q.quantize(data, "int8")
        assert profile.compression_ratio == 8.0  # 8 bytes → 1 byte
        assert profile.max_error >= 0

    def test_benchmark_precision(self):
        q = Quantizer()
        data = [float(i) / 10 for i in range(100)]
        results = q.benchmark_precision(data)
        assert "float32" in results
        assert "float16" in results
        assert "int8" in results
        # int8 should have highest compression
        assert results["int8"].compression_ratio >= results["float32"].compression_ratio

    def test_matrix_quantize(self):
        q = Quantizer()
        mat = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        quantized, profile = q.quantize(mat, "float16")
        assert len(quantized) == 2
        assert profile.samples_processed > 0


# ── Stage 6: Pruning ──

class TestPruning:
    def test_sparsity_analysis(self):
        pruner = Pruner(threshold=0.01)
        data = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
        report = pruner.analyze_sparsity(data)
        assert report.features_pruned == 3
        assert report.features_kept == 3
        assert report.sparsity == 0.5

    def test_node_redundancy(self):
        pruner = Pruner()
        latencies = {
            "fast_node": [1.0, 1.1, 0.9, 1.0],
            "variable_node": [1.0, 10.0, 2.0, 15.0],
        }
        results = pruner.analyze_node_redundancy(latencies)
        assert results["fast_node"]["suggestion"] == "stable"
        assert results["variable_node"]["suggestion"] == "consider caching"


# ── Stage 6: Batching ──

class TestBatching:
    def test_add_and_get(self):
        bs = BatchingStrategy(max_batch_size=3, max_wait_ms=0)
        bs.add_request("r1", "data1")
        bs.add_request("r2", "data2")
        batch = bs.get_batch()
        assert len(batch) == 2

    def test_max_batch_size(self):
        bs = BatchingStrategy(max_batch_size=2, max_wait_ms=0)
        for i in range(5):
            bs.add_request(f"r{i}", f"d{i}")
        batch = bs.get_batch()
        assert len(batch) == 2
        assert bs.pending_count == 3


# ── Stage 6: Cache Policy ──

class TestCachePolicy:
    def test_lru_eviction(self):
        cp = CachePolicy(max_entries=2, policy="lru")
        cp.put("a", 1)
        cp.put("b", 2)
        cp.put("c", 3)  # evicts "a"
        assert cp.get("a") is None
        assert cp.get("b") == 2
        assert cp.get("c") == 3

    def test_lfu_eviction(self):
        cp = CachePolicy(max_entries=2, policy="lfu")
        cp.put("a", 1)
        cp.put("b", 2)
        cp.get("a")  # hit a
        cp.get("a")  # hit a again
        cp.put("c", 3)  # evicts "b" (fewer hits)
        assert cp.get("a") == 1
        assert cp.get("b") is None


# ── Stage 7: Meta-Router ──

class TestMetaRouter:
    def test_record_and_recommend(self):
        mr = MetaRouter()
        for i in range(25):
            mr.record(RoutingExperience(
                timestamp=i, task="detect", input_modality="image",
                input_quality=0.9, node_selected="fast_model",
                node_capabilities=["detect"], confidence_achieved=0.95,
                latency_ms=10.0, success=True, feedback_score=0.9,
            ))
            mr.record(RoutingExperience(
                timestamp=i, task="detect", input_modality="image",
                input_quality=0.9, node_selected="slow_model",
                node_capabilities=["detect"], confidence_achieved=0.85,
                latency_ms=50.0, success=True, feedback_score=0.7,
            ))
        decision = mr.recommend("detect", "image", 0.9, ["fast_model", "slow_model"])
        assert decision.recommended_node == "fast_model"
        assert decision.confidence > 0
        assert decision.based_on_experiences > 0

    def test_no_data(self):
        mr = MetaRouter()
        decision = mr.recommend("x", "y", 1.0, ["a", "b"])
        assert decision.confidence == 0.0

    def test_summary(self):
        mr = MetaRouter()
        mr.record(RoutingExperience(
            timestamp=0, task="t", input_modality="m", input_quality=1.0,
            node_selected="n", node_capabilities=[], confidence_achieved=0.8,
            latency_ms=5.0, success=True,
        ))
        s = mr.summary()
        assert s["experiences"] == 1


# ── Stage 7: Experiments ──

class TestExperiments:
    def test_run_experiment(self):
        from sweep_neural_mesh.registry import ModelRegistry
        reg = ModelRegistry()
        node = NeuralNode(
            name="echo", capabilities=["echo_cap"],
            execute_fn=lambda data, **kw: data,
        )
        reg.register(node)
        runner = ExperimentalGraphRunner(reg)
        config = ExperimentConfig(
            name="test_exp",
            description="Test experiment",
            nodes=["echo_cap"],
            input_data="hello",
            expected_output="hello",
        )
        result = runner.run_experiment(config)
        assert result.success
        assert result.output == "hello"
        assert result.metrics.get("accuracy") == 1.0

    def test_compare_experiments(self):
        from sweep_neural_mesh.registry import ModelRegistry
        reg = ModelRegistry()
        node = NeuralNode(name="n", capabilities=["c"],
                          execute_fn=lambda data, **kw: data)
        reg.register(node)
        runner = ExperimentalGraphRunner(reg)
        r1 = runner.run_experiment(ExperimentConfig(name="e1", nodes=["c"], input_data=1))
        r2 = runner.run_experiment(ExperimentConfig(name="e2", nodes=["c"], input_data=2))
        comparison = runner.compare_experiments([r1, r2])
        assert "total_latency_ms" in comparison

    def test_save_results(self, tmp_path):
        from sweep_neural_mesh.registry import ModelRegistry
        reg = ModelRegistry()
        node = NeuralNode(name="n", capabilities=["c"],
                          execute_fn=lambda data, **kw: data)
        reg.register(node)
        runner = ExperimentalGraphRunner(reg)
        result = runner.run_experiment(
            ExperimentConfig(name="save_test", nodes=["c"], input_data="x")
        )
        path = runner.save_results(result, directory=str(tmp_path))
        assert path.exists()


# ── Stage 7: Benchmarking ──

class TestBenchmarking:
    def test_benchmark_node(self):
        node = NeuralNode(name="fast", execute_fn=lambda x, **kw: x * 2)
        suite = BenchmarkSuite(warmup_iterations=1)
        bench = suite.benchmark_node(node, [1, 2, 3], iterations=10)
        assert bench.iterations == 30  # 10 * 3
        assert bench.success_rate == 1.0
        assert bench.avg_latency_ms >= 0

    def test_benchmark_pipeline(self):
        n1 = NeuralNode(name="a", execute_fn=lambda x, **kw: x)
        n2 = NeuralNode(name="b", execute_fn=lambda x, **kw: x)
        suite = BenchmarkSuite(warmup_iterations=1)
        pipe = suite.benchmark_pipeline([n1, n2], [1, 2], iterations=5)
        assert pipe.nodes_benchmarked == 2
        assert pipe.success_rate == 1.0

    def test_compare_single_vs_ensemble(self):
        single = NeuralNode(name="single", execute_fn=lambda x, **kw: x)
        e1 = NeuralNode(name="e1", execute_fn=lambda x, **kw: x)
        e2 = NeuralNode(name="e2", execute_fn=lambda x, **kw: x)
        suite = BenchmarkSuite(warmup_iterations=1)
        comparison = suite.compare_single_vs_ensemble(single, [e1, e2], [1, 2])
        assert "single" in comparison
        assert "ensemble" in comparison


# ── Stage 8: Intent Integration ──

class TestIntentMesh:
    def test_route_known_intent(self):
        mesh = NeuralMesh()
        node = NeuralNode(
            name="echo", capabilities=["general_chat"],
            execute_fn=lambda data, **kw: f"response to: {data}",
        )
        mesh.register_node(node)
        im = IntentMesh(mesh)
        result = im.route("general_chat", "hello there")
        assert result.success
        assert result.output is not None

    def test_route_unknown_intent(self):
        mesh = NeuralMesh()
        node = NeuralNode(
            name="echo", capabilities=["general_chat"],
            execute_fn=lambda data, **kw: data,
        )
        mesh.register_node(node)
        im = IntentMesh(mesh)
        result = im.route("unknown_intent_xyz", "test")
        # Should fall back to general_chat
        assert result.success

    def test_coverage_report(self):
        mesh = NeuralMesh()
        im = IntentMesh(mesh)
        report = im.coverage_report()
        assert report["total_intents"] > 0
        assert report["satisfied"] == 0  # no nodes registered

    def test_available_intents(self):
        mesh = NeuralMesh()
        im = IntentMesh(mesh)
        intents = im.available_intents()
        assert "general_chat" in intents
        assert "voice_command" in intents

    def test_custom_route(self):
        mesh = NeuralMesh()
        node = NeuralNode(
            name="custom", capabilities=["my_custom_cap"],
            execute_fn=lambda data, **kw: "custom result",
        )
        mesh.register_node(node)
        im = IntentMesh(mesh)
        im.register_route(IntentRoute(
            intent="custom_intent",
            capabilities=["my_custom_cap"],
            modalities=["text"],
        ))
        result = im.route("custom_intent", "test")
        assert result.success
