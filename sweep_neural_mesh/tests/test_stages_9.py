"""
Tests for Stage 9: Cascade Router, Plugin System, Dynamic Graph Rewriter.
"""
import pytest
from pathlib import Path

from sweep_neural_mesh.routing.cascade import (
    CascadeRouter,
    CascadeTier,
    CascadeResult,
)
from sweep_neural_mesh.plugins.loader import (
    PluginLoader,
    PluginManifest,
    LoadedPlugin,
)
from sweep_neural_mesh.core.rewriting import (
    DynamicGraphRewriter,
    RewriteAction,
    RewriteProposal,
)
from sweep_neural_mesh.core.graph import MeshGraph
from sweep_neural_mesh.core.node import NeuralNode, NodeStatus


# ---------------------------------------------------------------------------
# Cascade Router
# ---------------------------------------------------------------------------


class TestCascadeRouter:
    def test_empty_tiers_raises(self):
        router = CascadeRouter()
        with pytest.raises(ValueError, match="no tiers"):
            router.route({"query": "test"})

    def test_single_tier_meets_threshold(self):
        def handler(q):
            return {"confidence": 0.95, "output": "fast result"}

        tier = CascadeTier(name="fast", cost_weight=1.0, latency_weight=1.0,
                           confidence_threshold=0.8, handler=handler)
        router = CascadeRouter([tier])
        result = router.route({"query": "hello"})

        assert result.tier_used == "fast"
        assert result.confidence == 0.95
        assert result.output == "fast result"
        assert not result.escalated
        assert result.tiers_attempted == ["fast"]

    def test_escalation_when_threshold_not_met(self):
        calls: list[str] = []

        def fast_handler(q):
            calls.append("fast")
            return {"confidence": 0.3, "output": "fast"}

        def slow_handler(q):
            calls.append("slow")
            return {"confidence": 0.92, "output": "slow but good"}

        tiers = [
            CascadeTier(name="fast", cost_weight=1.0, latency_weight=1.0,
                        confidence_threshold=0.8, handler=fast_handler),
            CascadeTier(name="slow", cost_weight=5.0, latency_weight=3.0,
                        confidence_threshold=0.8, handler=slow_handler),
        ]
        router = CascadeRouter(tiers)
        result = router.route({"query": "complex"})

        assert result.tier_used == "slow"
        assert result.escalated
        assert result.confidence == 0.92
        assert calls == ["fast", "slow"]
        assert len(result.tiers_attempted) == 2

    def test_falls_through_to_last_tier(self):
        def bad_handler(q):
            return {"confidence": 0.1, "output": "bad"}

        tiers = [
            CascadeTier(name="a", cost_weight=1.0, latency_weight=1.0,
                        confidence_threshold=0.9, handler=bad_handler),
            CascadeTier(name="b", cost_weight=2.0, latency_weight=2.0,
                        confidence_threshold=0.9, handler=bad_handler),
        ]
        router = CascadeRouter(tiers)
        result = router.route({"query": "x"})

        assert result.tier_used == "b"
        assert result.confidence == 0.1
        assert result.cost_saved == 0.0

    def test_cost_saved_calculation(self):
        def handler(q):
            return {"confidence": 1.0, "output": "ok"}

        tiers = [
            CascadeTier(name="cheap", cost_weight=1.0, latency_weight=1.0,
                        confidence_threshold=0.5, handler=handler),
            CascadeTier(name="expensive", cost_weight=10.0, latency_weight=10.0,
                        confidence_threshold=0.5, handler=handler),
        ]
        router = CascadeRouter(tiers)
        result = router.route({"query": "x"})

        assert result.tier_used == "cheap"
        assert result.cost_saved == pytest.approx(0.9, abs=0.01)

    def test_history_and_stats(self):
        def handler(q):
            return {"confidence": 0.95, "output": "ok"}

        tier = CascadeTier(name="t", cost_weight=1.0, latency_weight=1.0,
                           confidence_threshold=0.8, handler=handler)
        router = CascadeRouter([tier])

        for _ in range(5):
            router.route({"query": "x"})

        assert len(router.history) == 5
        stats = router.stats()
        assert stats["routes"] == 5
        assert stats["avg_confidence"] == pytest.approx(0.95)

    def test_handler_none_returns_zero_confidence(self):
        tier = CascadeTier(name="nohandler", cost_weight=1.0, latency_weight=1.0,
                           confidence_threshold=0.0, handler=None)
        router = CascadeRouter([tier])
        result = router.route({"query": "x"})
        assert result.confidence == 0.0

    def test_result_to_dict(self):
        def handler(q):
            return {"confidence": 0.9, "output": "ok"}
        tier = CascadeTier(name="t", cost_weight=1.0, latency_weight=1.0,
                           confidence_threshold=0.5, handler=handler)
        router = CascadeRouter([tier])
        result = router.route({"query": "x"})
        d = result.to_dict()
        assert d["tier_used"] == "t"
        assert "total_latency_ms" in d

    def test_repr(self):
        t = CascadeTier(name="x", cost_weight=1.0, latency_weight=1.0,
                        confidence_threshold=0.5)
        router = CascadeRouter([t])
        assert "CascadeRouter" in repr(router)


# ---------------------------------------------------------------------------
# Plugin System
# ---------------------------------------------------------------------------


class TestPluginManifest:
    def test_to_dict(self):
        m = PluginManifest(
            name="test-plugin",
            version="1.2.3",
            capabilities=["intent", "search"],
            entry_point="some.module",
            priority=10,
        )
        d = m.to_dict()
        assert d["name"] == "test-plugin"
        assert d["capabilities"] == ["intent", "search"]
        assert d["enabled"] is True


class TestPluginLoader:
    def test_register_and_count(self):
        loader = PluginLoader()
        m = PluginManifest(name="a", version="1.0.0")
        loader.register_manifest(m)
        assert loader.manifest_count == 1

    def test_load_disabled_plugin(self):
        loader = PluginLoader()
        m = PluginManifest(name="disabled", version="0.1", enabled=False)
        loader.register_manifest(m)
        lp = loader.load_plugin("disabled")
        assert not lp.healthy
        assert lp.error == "disabled"

    def test_load_nonexistent_plugin_raises(self):
        loader = PluginLoader()
        with pytest.raises(KeyError, match="unknown plugin"):
            loader.load_plugin("does_not_exist")

    def test_load_existing_module(self):
        loader = PluginLoader()
        m = PluginManifest(
            name="json_mod",
            version="1.0.0",
            entry_point="json",
        )
        loader.register_manifest(m)
        lp = loader.load_plugin("json_mod")
        assert lp.healthy
        assert lp.module is not None

    def test_load_broken_module_sets_error(self):
        loader = PluginLoader()
        m = PluginManifest(
            name="broken",
            version="1.0.0",
            entry_point="this_module_does_not_exist_xyz_123",
        )
        loader.register_manifest(m)
        lp = loader.load_plugin("broken")
        assert not lp.healthy
        assert lp.error is not None

    def test_load_all_skips_disabled(self):
        loader = PluginLoader()
        loader.register_manifest(PluginManifest(name="on", version="1.0", entry_point="json"))
        loader.register_manifest(PluginManifest(name="off", version="1.0", enabled=False))
        results = loader.load_all()
        assert "on" in results
        assert "off" not in results

    def test_plugins_for_capability(self):
        loader = PluginLoader()
        loader.register_manifest(PluginManifest(
            name="search_plugin", version="1.0", capabilities=["search", "intent"],
            entry_point="json",
        ))
        loader.register_manifest(PluginManifest(
            name="other_plugin", version="1.0", capabilities=["bluetooth"],
            entry_point="json",
        ))
        loader.load_all()
        search_plugins = loader.plugins_for_capability("search")
        assert len(search_plugins) == 1
        assert search_plugins[0].manifest.name == "search_plugin"

    def test_discover_directory_json_manifest(self, tmp_path):
        manifest_file = tmp_path / "sweep_plugin.json"
        manifest_file.write_text('{"name": "alpha", "version": "1.0", "capabilities": ["test"]}')

        loader = PluginLoader()
        found = loader.discover_directory(tmp_path)
        assert found == 1
        assert loader.manifest_count == 1

    def test_discover_directory_init_plugin(self, tmp_path):
        plugin_dir = tmp_path / "mypkg"
        plugin_dir.mkdir()
        init_file = plugin_dir / "__init__.py"
        init_file.write_text(
            'SWEEP_PLUGIN = {"name": "mypkg", "version": "2.0", "capabilities": ["cap1"]}'
        )

        loader = PluginLoader()
        found = loader.discover_directory(tmp_path)
        assert found == 1

    def test_discover_nonexistent_dir(self, tmp_path):
        loader = PluginLoader()
        found = loader.discover_directory(tmp_path / "nonexistent")
        assert found == 0

    def test_load_nonexistent_entry_point_sets_error(self):
        loader = PluginLoader()
        loader.register_manifest(PluginManifest(
            name="fake", version="1.0", entry_point="totally.fake.module.xyz",
        ))
        lp = loader.load_plugin("fake")
        assert not lp.healthy
        assert lp.error is not None

    def test_idempotent_load(self):
        loader = PluginLoader()
        loader.register_manifest(PluginManifest(name="j", version="1.0", entry_point="json"))
        lp1 = loader.load_plugin("j")
        lp2 = loader.load_plugin("j")
        assert lp1 is lp2

    def test_summary(self):
        loader = PluginLoader()
        loader.register_manifest(PluginManifest(name="x", version="1.0", capabilities=["a"]))
        s = loader.summary()
        assert s["manifests_registered"] == 1
        assert "a" in s["all_capabilities"]

    def test_repr(self):
        loader = PluginLoader()
        assert "PluginLoader" in repr(loader)

    def test_loaded_plugin_healthy_flag(self):
        lp = LoadedPlugin(manifest=PluginManifest(name="ok", version="1.0"), module=object())
        assert lp.healthy

    def test_loaded_plugin_error_flag(self):
        lp = LoadedPlugin(manifest=PluginManifest(name="err", version="1.0"), module=None, error="boom")
        assert not lp.healthy


# ---------------------------------------------------------------------------
# Dynamic Graph Rewriter
# ---------------------------------------------------------------------------


class TestDynamicGraphRewriter:
    def _make_node(self, nid: str, name: str | None = None) -> NeuralNode:
        return NeuralNode(node_id=nid, name=name or nid)

    def _make_linear_graph(self, n_ids: list[str]) -> MeshGraph:
        """Create a linear chain graph: n0 -> n1 -> n2 -> ..."""
        g = MeshGraph(graph_id="rewrite-test")
        nodes = []
        for nid in n_ids:
            node = self._make_node(nid)
            g.add_node(node)
            nodes.append(node)
        for i in range(len(nodes) - 1):
            g.add_edge(nodes[i], nodes[i + 1])
        return g

    def test_record_execution(self):
        rewriter = DynamicGraphRewriter()
        rewriter.record_execution("n1", latency_ms=50.0, confidence=0.9, success=True)
        rewriter.record_execution("n1", latency_ms=60.0, confidence=0.85, success=True)
        assert rewriter.metrics_tracked == 1

    def test_propose_bypass_low_confidence(self):
        rewriter = DynamicGraphRewriter()
        rewriter._min_samples = 3
        for _ in range(5):
            rewriter.record_execution("n1", latency_ms=10.0, confidence=0.01, success=False)

        g = self._make_linear_graph(["n0", "n1", "n2"])
        proposals = rewriter.propose_rewrites(g)
        bypass = [p for p in proposals if p.action == RewriteAction.BYPASS]
        assert len(bypass) == 1
        assert bypass[0].target_node_id == "n1"

    def test_propose_cache_high_latency(self):
        rewriter = DynamicGraphRewriter()
        rewriter._min_samples = 1
        for _ in range(5):
            rewriter.record_execution("n1", latency_ms=200.0, confidence=0.9, success=True)

        g = self._make_linear_graph(["n0", "n1", "n2"])
        proposals = rewriter.propose_rewrites(g)
        cache = [p for p in proposals if p.action == RewriteAction.ADD_CACHE]
        assert len(cache) == 1

    def test_validate_proposal_node_not_found(self):
        rewriter = DynamicGraphRewriter()
        g = self._make_linear_graph(["n0", "n1"])
        proposal = RewriteProposal(
            action=RewriteAction.BYPASS,
            target_node_id="nonexistent",
            description="test",
        )
        assert not rewriter.validate_proposal(proposal, g)

    def test_validate_bypass_needs_single_predecessor_successor(self):
        rewriter = DynamicGraphRewriter()
        g = MeshGraph(graph_id="test")
        n1 = self._make_node("n1", "a")
        n2 = self._make_node("n2", "b")
        n3 = self._make_node("n3", "c")
        n4 = self._make_node("n4", "d")
        g.add_node(n1)
        g.add_node(n2)
        g.add_node(n3)
        g.add_node(n4)
        g.add_edge(n1, n2)
        g.add_edge(n2, n3)
        g.add_edge(n2, n4)  # n2 has two successors

        proposal = RewriteProposal(
            action=RewriteAction.BYPASS,
            target_node_id="n2",
            description="test",
        )
        assert not rewriter.validate_proposal(proposal, g)

    def test_apply_bypass(self):
        rewriter = DynamicGraphRewriter()
        g = MeshGraph(graph_id="test")
        n1 = self._make_node("n1", "a")
        n2 = self._make_node("n2", "b")
        n3 = self._make_node("n3", "c")
        g.add_node(n1)
        g.add_node(n2)
        g.add_node(n3)
        g.add_edge(n1, n2)
        g.add_edge(n2, n3)

        proposal = RewriteProposal(
            action=RewriteAction.BYPASS,
            target_node_id="n2",
            description="bypass b",
        )
        record = rewriter.apply_proposal(proposal, g)
        assert record is not None
        assert rewriter.rewrite_count == 1
        # n2 should be gone
        node_ids = {n.node_id for n in g.nodes}
        assert "n2" not in node_ids
        # n1 should have edge to n3
        succs = g.successors("n1")
        assert any(s.node_id == "n3" for s in succs)

    def test_apply_cache_insertion(self):
        rewriter = DynamicGraphRewriter()
        g = self._make_linear_graph(["n0", "n1"])

        proposal = RewriteProposal(
            action=RewriteAction.ADD_CACHE,
            target_node_id="n0",
            description="cache n0",
            params={"avg_latency_ms": 150.0},
        )
        record = rewriter.apply_proposal(proposal, g)
        assert record is not None
        for node in g.nodes:
            if node.node_id == "n0":
                assert node.tags.get("cache_enabled") == "true"

    def test_stats(self):
        rewriter = DynamicGraphRewriter()
        rewriter.record_execution("a", 10.0, 0.9, True)
        s = rewriter.stats()
        assert s["nodes_tracked"] == 1
        assert s["rewrites_applied"] == 0

    def test_no_proposals_when_metrics_insufficient(self):
        rewriter = DynamicGraphRewriter()
        rewriter._min_samples = 10
        rewriter.record_execution("n1", 10.0, 0.5, True)
        g = self._make_linear_graph(["n0", "n1", "n2"])
        proposals = rewriter.propose_rewrites(g)
        assert len(proposals) == 0

    def test_repr(self):
        r = DynamicGraphRewriter()
        assert "DynamicGraphRewriter" in repr(r)

    def test_graph_hash_changes_after_removal(self):
        rewriter = DynamicGraphRewriter()
        g = self._make_linear_graph(["n0", "n1", "n2"])
        h1 = rewriter._graph_hash(g)
        g.remove_node("n1")
        h2 = rewriter._graph_hash(g)
        assert h1 != h2

    def test_proposals_sorted_by_confidence(self):
        rewriter = DynamicGraphRewriter()
        rewriter._min_samples = 1
        rewriter.record_execution("n1", 300.0, 0.9, True)  # cache proposal
        rewriter.record_execution("n2", 10.0, 0.01, False)  # bypass proposal (low confidence)
        g = MeshGraph(graph_id="test")
        nodes = []
        for nid in ["n0", "n1", "n2", "n3"]:
            node = self._make_node(nid)
            g.add_node(node)
            nodes.append(node)
        g.add_edge(nodes[0], nodes[1])
        g.add_edge(nodes[1], nodes[2])
        g.add_edge(nodes[2], nodes[3])

        proposals = rewriter.propose_rewrites(g)
        if len(proposals) >= 2:
            assert proposals[0].confidence >= proposals[1].confidence
