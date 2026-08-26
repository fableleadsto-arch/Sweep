"""Tests for advanced math/logic modules: information theory, fuzzy logic, graph algorithms."""
from __future__ import annotations

import math
import unittest

from sweep_neural_mesh.neurons.information import (
    InformationTheory, EntropyResult, MutualInfoResult, InformationGainResult,
)
from sweep_neural_mesh.neurons.fuzzy_logic import (
    FuzzyReasoner, FuzzyEvidenceGrader, FuzzySet, FuzzyRule, FuzzyResult,
    triangular_mf, trapezoidal_mf, gaussian_mf, sigmoid_mf,
    fuzzy_and, fuzzy_or, fuzzy_not, fuzzy_implies,
    probabilistic_and, probabilistic_or, bounded_and, bounded_or,
)
from sweep_neural_mesh.neurons.graph_algorithms import (
    ReasoningGraph, GraphNode, GraphEdge,
    PageRankResult, ShortestPathResult, CentralityResult, CommunityResult,
)


# ════════════════════════════════════════════════════════════════
# INFORMATION THEORY TESTS
# ════════════════════════════════════════════════════════════════

class TestInformationTheory(unittest.TestCase):
    def setUp(self):
        self.it = InformationTheory()

    def test_uniform_distribution_max_entropy(self):
        dist = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
        result = self.it.shannon_entropy(dist)
        self.assertAlmostEqual(result.entropy, 2.0, places=3)  # log2(4) = 2.0
        self.assertAlmostEqual(result.normalized, 1.0, places=3)

    def test_deterministic_distribution_zero_entropy(self):
        dist = {"a": 1.0, "b": 0.0, "c": 0.0}
        result = self.it.shannon_entropy(dist)
        self.assertAlmostEqual(result.entropy, 0.0, places=3)

    def test_binary_entropy(self):
        dist = {"yes": 0.5, "no": 0.5}
        result = self.it.shannon_entropy(dist)
        self.assertAlmostEqual(result.entropy, 1.0, places=3)  # log2(2) = 1.0

    def test_entropy_is_positive(self):
        dist = {"a": 0.3, "b": 0.7}
        result = self.it.shannon_entropy(dist)
        self.assertGreater(result.entropy, 0.0)

    def test_entropy_nats_conversion(self):
        dist = {"a": 1.0, "b": 1.0}
        result = self.it.shannon_entropy(dist)
        self.assertAlmostEqual(result.entropy_nats, math.log(2), places=3)

    def test_empty_distribution(self):
        result = self.it.shannon_entropy({})
        self.assertAlmostEqual(result.entropy, 0.0, places=3)

    def test_mutual_information_identical(self):
        joint = {("a", "a"): 0.5, ("b", "b"): 0.5}
        mx = {"a": 0.5, "b": 0.5}
        my = {"a": 0.5, "b": 0.5}
        result = self.it.mutual_information(joint, mx, my)
        self.assertAlmostEqual(result.mi_xy, 1.0, places=3)
        self.assertEqual(result.interpretation, "very strong")

    def test_mutual_information_independent(self):
        joint = {("a", "x"): 0.25, ("a", "y"): 0.25, ("b", "x"): 0.25, ("b", "y"): 0.25}
        mx = {"a": 0.5, "b": 0.5}
        my = {"x": 0.5, "y": 0.5}
        result = self.it.mutual_information(joint, mx, my)
        self.assertAlmostEqual(result.mi_xy, 0.0, places=3)

    def test_cross_entropy(self):
        actual = {"a": 1.0, "b": 0.0}
        predicted = {"a": 1.0, "b": 0.0}
        ce = self.it.cross_entropy(actual, predicted)
        self.assertAlmostEqual(ce, 0.0, places=3)

    def test_cross_entropy_worse_when_predicted_is_different(self):
        actual = {"a": 1.0, "b": 0.0}
        good_pred = {"a": 0.9, "b": 0.1}
        bad_pred = {"a": 0.1, "b": 0.9}
        ce_good = self.it.cross_entropy(actual, good_pred)
        ce_bad = self.it.cross_entropy(actual, bad_pred)
        self.assertLess(ce_good, ce_bad)

    def test_perfect_perplexity(self):
        actual = {"a": 1.0}
        predicted = {"a": 1.0}
        pp = self.it.perplexity(actual, predicted)
        self.assertAlmostEqual(pp, 1.0, places=3)

    def test_surprise_high_for_rare_event(self):
        s = self.it.surprise(0.01)
        self.assertGreater(s, 6.0)  # -log2(0.01) ≈ 6.64

    def test_surprise_zero_for_certain_event(self):
        s = self.it.surprise(1.0)
        self.assertAlmostEqual(s, 0.0, places=3)

    def test_is_surprising(self):
        self.assertTrue(self.it.is_surprising(0.01))
        self.assertFalse(self.it.is_surprising(0.9))

    def test_information_gain(self):
        parent = ["yes", "yes", "no", "no", "yes"]
        children = {
            "group1": ["yes", "yes"],
            "group2": ["no", "no", "yes"],
        }
        result = self.it.information_gain(parent, children, "test_attribute")
        self.assertGreater(result.information_gain, 0.0)
        self.assertEqual(result.split_attribute, "test_attribute")

    def test_information_gain_perfect_split(self):
        parent = ["yes", "yes", "no", "no"]
        children = {"group1": ["yes", "yes"], "group2": ["no", "no"]}
        result = self.it.information_gain(parent, children, "perfect")
        self.assertGreater(result.information_gain, 0.5)

    def test_get_stats(self):
        self.it.shannon_entropy({"a": 1.0})
        stats = self.it.get_stats()
        self.assertEqual(stats["total_computations"], 1)


# ════════════════════════════════════════════════════════════════
# FUZZY LOGIC TESTS
# ════════════════════════════════════════════════════════════════

class TestFuzzyLogic(unittest.TestCase):
    def test_triangular_mf_peak(self):
        self.assertAlmostEqual(triangular_mf(5.0, 0.0, 5.0, 10.0), 1.0)

    def test_triangular_mf_zero_at_foot(self):
        self.assertAlmostEqual(triangular_mf(0.0, 0.0, 5.0, 10.0), 0.0)
        self.assertAlmostEqual(triangular_mf(10.0, 0.0, 5.0, 10.0), 0.0)

    def test_triangular_mf_midpoint(self):
        self.assertAlmostEqual(triangular_mf(2.5, 0.0, 5.0, 10.0), 0.5)

    def test_trapezoidal_mf_plateau(self):
        self.assertAlmostEqual(trapezoidal_mf(5.0, 0.0, 3.0, 7.0, 10.0), 1.0)
        self.assertAlmostEqual(trapezoidal_mf(6.0, 0.0, 3.0, 7.0, 10.0), 1.0)

    def test_trapezoidal_mf_zero_outside(self):
        self.assertAlmostEqual(trapezoidal_mf(0.0, 0.0, 3.0, 7.0, 10.0), 0.0)
        self.assertAlmostEqual(trapezoidal_mf(10.0, 0.0, 3.0, 7.0, 10.0), 0.0)

    def test_gaussian_mf_peak_at_mean(self):
        self.assertAlmostEqual(gaussian_mf(5.0, 5.0, 1.0), 1.0)

    def test_gaussian_mf_decreases_away(self):
        self.assertLess(gaussian_mf(10.0, 5.0, 1.0), gaussian_mf(6.0, 5.0, 1.0))

    def test_sigmoid_mf_at_center(self):
        self.assertAlmostEqual(sigmoid_mf(5.0, 5.0, 1.0), 0.5)

    def test_sigmoid_mf_above_center(self):
        self.assertGreater(sigmoid_mf(10.0, 5.0, 1.0), 0.5)

    def test_fuzzy_and_min(self):
        self.assertAlmostEqual(fuzzy_and(0.3, 0.7), 0.3)
        self.assertAlmostEqual(fuzzy_and(0.8, 0.9), 0.8)

    def test_fuzzy_or_max(self):
        self.assertAlmostEqual(fuzzy_or(0.3, 0.7), 0.7)
        self.assertAlmostEqual(fuzzy_or(0.1, 0.2), 0.2)

    def test_fuzzy_not_complement(self):
        self.assertAlmostEqual(fuzzy_not(0.0), 1.0)
        self.assertAlmostEqual(fuzzy_not(1.0), 0.0)
        self.assertAlmostEqual(fuzzy_not(0.5), 0.5)

    def test_fuzzy_implies(self):
        self.assertAlmostEqual(fuzzy_implies(0.0, 1.0), 1.0)  # F → T = T
        self.assertAlmostEqual(fuzzy_implies(1.0, 0.0), 0.0)  # T → F = F
        self.assertAlmostEqual(fuzzy_implies(1.0, 1.0), 1.0)  # T → T = T

    def test_probabilistic_and(self):
        self.assertAlmostEqual(probabilistic_and(0.5, 0.5), 0.25)

    def test_probabilistic_or(self):
        self.assertAlmostEqual(probabilistic_or(0.5, 0.5), 0.75)

    def test_bounded_and(self):
        self.assertAlmostEqual(bounded_and(0.5, 0.5), 0.0)
        self.assertAlmostEqual(bounded_and(0.8, 0.8), 0.6)

    def test_bounded_or(self):
        self.assertAlmostEqual(bounded_or(0.5, 0.5), 1.0)

    def test_fuzzy_reasoner_basic(self):
        reasoner = FuzzyReasoner()
        reasoner.add_fuzzy_set("temperature", "hot", {"20": 0.0, "30": 0.5, "40": 1.0})
        reasoner.add_fuzzy_set("temperature", "cold", {"20": 1.0, "30": 0.5, "40": 0.0})
        reasoner.add_rule(FuzzyRule(
            name="hot_rule",
            antecedents=[("temperature", "and", "hot")],
            consequent=("comfort", "uncomfortable"),
        ))
        result = reasoner.infer({"temperature": {"hot": 0.5, "cold": 0.5}})
        self.assertIsInstance(result, FuzzyResult)
        self.assertGreaterEqual(result.defuzzified, 0.0)

    def test_fuzzy_evidence_grader(self):
        grader = FuzzyEvidenceGrader()
        result = grader.grade(strength=0.8, reliability=0.9, coherence=0.7)
        self.assertIsInstance(result, FuzzyResult)
        self.assertGreater(result.defuzzified, 0.0)

    def test_fuzzy_evidence_grader_weak(self):
        grader = FuzzyEvidenceGrader()
        strong = grader.grade(strength=0.9, reliability=0.9)
        weak = grader.grade(strength=0.1, reliability=0.1)
        self.assertGreater(strong.defuzzified, weak.defuzzified)


# ════════════════════════════════════════════════════════════════
# GRAPH ALGORITHMS TESTS
# ════════════════════════════════════════════════════════════════

class TestGraphAlgorithms(unittest.TestCase):
    def setUp(self):
        self.graph = ReasoningGraph()

    def test_add_node(self):
        self.graph.add_node("A")
        stats = self.graph.get_stats()
        self.assertEqual(stats["node_count"], 1)

    def test_add_edge(self):
        self.graph.add_edge("A", "B")
        stats = self.graph.get_stats()
        self.assertEqual(stats["node_count"], 2)
        self.assertEqual(stats["edge_count"], 1)

    def test_pagerank_simple(self):
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        self.graph.add_edge("C", "A")
        result = self.graph.pagerank(max_iter=50)
        self.assertIsInstance(result, PageRankResult)
        self.assertTrue(result.converged)
        # All nodes should have roughly equal PageRank in a cycle
        for node, rank in result.rankings.items():
            self.assertAlmostEqual(rank, 1 / 3, delta=0.1)

    def test_pagerank_linear_chain(self):
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        result = self.graph.pagerank(max_iter=50)
        # C should have highest rank (most incoming links)
        self.assertGreater(result.rankings["C"], result.rankings["A"])

    def test_shortest_path_direct(self):
        self.graph.add_edge("A", "B", weight=1.0)
        self.graph.add_edge("B", "C", weight=1.0)
        result = self.graph.shortest_path("A", "C")
        self.assertTrue(result.found)
        self.assertEqual(result.path, ["A", "B", "C"])
        self.assertAlmostEqual(result.total_weight, 2.0)

    def test_shortest_path_direct_vs_indirect(self):
        self.graph.add_edge("A", "B", weight=1.0)
        self.graph.add_edge("B", "C", weight=1.0)
        self.graph.add_edge("A", "C", weight=5.0)
        result = self.graph.shortest_path("A", "C")
        self.assertEqual(result.path, ["A", "B", "C"])

    def test_shortest_path_not_found(self):
        self.graph.add_node("A")
        self.graph.add_node("B")
        result = self.graph.shortest_path("A", "B")
        self.assertFalse(result.found)

    def test_betweenness_centrality(self):
        # Bridge graph: A → B → C
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        result = self.graph.betweenness_centrality()
        self.assertIsInstance(result, CentralityResult)
        # B should have highest betweenness (it's the bridge)
        self.assertGreater(result.centrality["B"], result.centrality["A"])

    def test_label_propagation(self):
        # Two cliques connected by a bridge
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "A")
        self.graph.add_edge("A", "C")
        self.graph.add_edge("C", "A")
        self.graph.add_edge("B", "D")
        self.graph.add_edge("D", "B")
        self.graph.add_edge("C", "D", weight=0.1)  # weak bridge
        result = self.graph.label_propagation(max_iter=20)
        self.assertIsInstance(result, CommunityResult)
        self.assertGreaterEqual(result.num_communities, 1)

    def test_topological_sort(self):
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        order = self.graph.topological_sort()
        self.assertEqual(order.index("A"), 0)
        self.assertEqual(order.index("B"), 1)
        self.assertEqual(order.index("C"), 2)

    def test_density_complete(self):
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "A")
        self.graph.add_edge("A", "C")
        self.graph.add_edge("C", "A")
        self.graph.add_edge("B", "C")
        self.graph.add_edge("C", "B")
        self.assertAlmostEqual(self.graph.density(), 1.0)

    def test_density_empty(self):
        self.assertAlmostEqual(self.graph.density(), 0.0)

    def test_neighbors(self):
        self.graph.add_edge("A", "B")
        self.graph.add_edge("C", "A")
        neighbors = self.graph.neighbors("A")
        self.assertIn("B", neighbors)
        self.assertIn("C", neighbors)

    def test_reachable_from(self):
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        self.graph.add_node("D")  # isolated
        reachable = self.graph.reachable_from("A")
        self.assertIn("A", reachable)
        self.assertIn("B", reachable)
        self.assertIn("C", reachable)
        self.assertNotIn("D", reachable)

    def test_in_out_degree(self):
        self.graph.add_edge("A", "B")
        self.graph.add_edge("C", "B")
        self.assertEqual(self.graph.in_degree("B"), 2)
        self.assertEqual(self.graph.out_degree("A"), 1)


if __name__ == "__main__":
    unittest.main()
