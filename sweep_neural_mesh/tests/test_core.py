"""Tests for sweep_neural_mesh core abstractions (Stage 1)."""
from __future__ import annotations

import pytest

from sweep_neural_mesh.core.node import (
    Framework,
    Modality,
    NeuralNode,
    NodeCostProfile,
    NodeResult,
    NodeSchema,
    NodeStatus,
    NodeVersion,
)
from sweep_neural_mesh.core.packet import NeuralPacket, Provenance, SpatialContext
from sweep_neural_mesh.core.graph import Edge, MeshGraph
from sweep_neural_mesh.core.engine import ExecutionEngine, ExecutionResult
from sweep_neural_mesh.core.router import ModelRouter, RoutingContext, RoutingResult
from sweep_neural_mesh.registry import ModelRegistry
from sweep_neural_mesh.registry.capability_registry import CapabilityRegistry


# ── NeuralNode ──

class TestNeuralNode:
    def test_create_default(self):
        node = NeuralNode()
        assert node.node_id
        assert node.name == "unnamed"
        assert node.status == NodeStatus.IDLE

    def test_execute_success(self):
        node = NeuralNode(execute_fn=lambda x, **kw: x * 2)
        result = node.execute(5)
        assert result.success
        assert result.output == 10
        assert result.latency_ms >= 0

    def test_execute_failure(self):
        def bad_fn(*a, **kw):
            raise ValueError("boom")
        node = NeuralNode(execute_fn=bad_fn)
        result = node.execute()
        assert not result.success
        assert "boom" in result.error
        assert node.status == NodeStatus.FAILED

    def test_execute_no_fn(self):
        node = NeuralNode()
        result = node.execute()
        assert not result.success
        assert "no execute function" in result.error

    def test_history_tracking(self):
        node = NeuralNode(execute_fn=lambda x, **kw: x)
        node.execute(1)
        node.execute(2)
        assert len(node.history) == 2
        assert node.avg_latency_ms >= 0

    def test_to_dict(self):
        node = NeuralNode(name="test_node")
        d = node.to_dict()
        assert d["name"] == "test_node"
        assert "node_id" in d

    def test_model_lifecycle(self):
        node = NeuralNode()
        assert node.model is None
        node.load_model("fake_model")
        assert node.model == "fake_model"
        assert node.is_ready
        node.unload_model()
        assert node.model is None


# ── NeuralPacket ──

class TestNeuralPacket:
    def test_create_default(self):
        pkt = NeuralPacket(data=[1, 2, 3])
        assert pkt.packet_id
        assert pkt.modality == Modality.TENSOR
        assert pkt.data == [1, 2, 3]

    def test_provenance_chain(self):
        p1 = NeuralPacket(data="x", source_node_id="a", source_node_name="model_a")
        p2 = p1.extend_provenance("b", "model_b", "transform")
        assert p2.parent_packet_id == p1.packet_id
        assert len(p2.provenance.chain) == 1
        assert p2.provenance.source_node_name == "model_b"

    def test_to_dict(self):
        pkt = NeuralPacket(data=[1], embedding=[0.1, 0.2])
        d = pkt.to_dict()
        assert d["has_data"] is True
        assert d["has_embedding"] is True
        assert d["embedding_dim"] == 2


# ── MeshGraph ──

class TestMeshGraph:
    def test_linear_graph(self):
        n1 = NeuralNode(name="input")
        n2 = NeuralNode(name="process")
        n3 = NeuralNode(name="output")
        g = MeshGraph()
        g.add_node(n1)
        g.add_node(n2)
        g.add_node(n3)
        g.add_edge(n1, n2)
        g.add_edge(n2, n3)
        order = g.execution_order
        assert order.index(n1.node_id) < order.index(n2.node_id)
        assert order.index(n2.node_id) < order.index(n3.node_id)

    def test_diamond_graph(self):
        n1 = NeuralNode(name="input")
        n2 = NeuralNode(name="branch_a")
        n3 = NeuralNode(name="branch_b")
        n4 = NeuralNode(name="merge")
        g = MeshGraph()
        for n in [n1, n2, n3, n4]:
            g.add_node(n)
        g.add_edge(n1, n2)
        g.add_edge(n1, n3)
        g.add_edge(n2, n4)
        g.add_edge(n3, n4)
        order = g.execution_order
        assert order.index(n1.node_id) < order.index(n2.node_id)
        assert order.index(n1.node_id) < order.index(n3.node_id)
        assert order.index(n2.node_id) < order.index(n4.node_id)

    def test_cycle_detection(self):
        n1 = NeuralNode(name="a")
        n2 = NeuralNode(name="b")
        g = MeshGraph()
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(n1, n2)
        g.add_edge(n2, n1)
        with pytest.raises(ValueError, match="Cycle detected"):
            g.execution_order

    def test_roots_and_leaves(self):
        n1 = NeuralNode(name="input")
        n2 = NeuralNode(name="middle")
        n3 = NeuralNode(name="output")
        g = MeshGraph()
        for n in [n1, n2, n3]:
            g.add_node(n)
        g.add_edge(n1, n2)
        g.add_edge(n2, n3)
        assert len(g.roots()) == 1
        assert g.roots()[0].name == "input"
        assert len(g.leaves()) == 1
        assert g.leaves()[0].name == "output"


# ── ExecutionEngine ──

class TestExecutionEngine:
    def test_linear_execution(self):
        n1 = NeuralNode(name="doubler", execute_fn=lambda x, **kw: x * 2)
        n2 = NeuralNode(name="adder", execute_fn=lambda x, **kw: x + 10)
        g = MeshGraph()
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(n1, n2)
        engine = ExecutionEngine()
        from sweep_neural_mesh.core.packet import NeuralPacket
        inp = NeuralPacket(data=5)
        result = engine.execute(g, initial_inputs={n1.node_id: inp})
        assert result.success
        assert result.total_nodes_executed == 2

    def test_failure_handling(self):
        n1 = NeuralNode(name="good")
        n2 = NeuralNode(name="bad", execute_fn=lambda **kw: (_ for _ in ()).throw(ValueError("fail")))
        def good_fn(data, **kw):
            return data
        n1._execute_fn = good_fn
        g = MeshGraph()
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(n1, n2)
        engine = ExecutionEngine()
        result = engine.execute(g)
        assert result.total_nodes_failed >= 1


# ── ModelRegistry ──

class TestModelRegistry:
    def test_register_and_find(self):
        reg = ModelRegistry()
        node = NeuralNode(name="vision", capabilities=["face_detection"])
        reg.register(node)
        assert reg.size == 1
        found = reg.find_capability("face_detection")
        assert len(found) == 1

    def test_unregister(self):
        reg = ModelRegistry()
        node = NeuralNode(name="x", capabilities=["cap_a"])
        reg.register(node)
        removed = reg.unregister(node.node_id)
        assert removed is not None
        assert reg.size == 0

    def test_find_capabilities(self):
        reg = ModelRegistry()
        n1 = NeuralNode(name="a", capabilities=["cap_x", "cap_y"])
        n2 = NeuralNode(name="b", capabilities=["cap_x", "cap_z"])
        n3 = NeuralNode(name="c", capabilities=["cap_y", "cap_z"])
        reg.register(n1)
        reg.register(n2)
        reg.register(n3)
        both = reg.find_capabilities(["cap_x", "cap_y"], require_all=True)
        assert len(both) == 1
        any_match = reg.find_capabilities(["cap_x", "cap_y"], require_all=False)
        assert len(any_match) == 3  # a has cap_x+cap_y, b has cap_x, c has cap_y

    def test_summary(self):
        reg = ModelRegistry()
        reg.register(NeuralNode(name="a", capabilities=["c1"]))
        s = reg.summary()
        assert s["total_nodes"] == 1


# ── CapabilityRegistry ──

class TestCapabilityRegistry:
    def test_register_and_find(self):
        cr = CapabilityRegistry()
        cr.register("face_detection", Modality.IMAGE, description="detect faces")
        cr.register("speech_to_text", Modality.AUDIO)
        assert len(cr.find_by_modality(Modality.IMAGE)) == 1

    def test_path_finding(self):
        cr = CapabilityRegistry()
        cr.register("image_encode", Modality.IMAGE, outputs=[Modality.EMBEDDING])
        cr.register("embed_to_class", Modality.EMBEDDING, outputs=[Modality.STRUCTURED])
        path = cr.find_path(Modality.IMAGE, Modality.STRUCTURED)
        assert len(path) == 2


# ── Routing ──

class TestModelRouter:
    def test_rank_selects_best(self):
        reg = ModelRegistry()
        n1 = NeuralNode(
            name="slow", capabilities=["detect"],
            cost=NodeCostProfile(avg_latency_ms=100, memory_mb=100),
        )
        n2 = NeuralNode(
            name="fast", capabilities=["detect"],
            cost=NodeCostProfile(avg_latency_ms=10, memory_mb=50),
        )
        reg.register(n1)
        reg.register(n2)
        router = ModelRouter(reg)
        ctx = RoutingContext(latency_budget_ms=50, memory_budget_mb=200)
        result = router.rank_multi("detect", ctx)
        assert result.selected_node is not None
        assert result.selected_node.name == "fast"

    def test_rank_empty(self):
        reg = ModelRegistry()
        router = ModelRouter(reg)
        result = router.rank([], RoutingContext())
        assert result.selected_node is None
