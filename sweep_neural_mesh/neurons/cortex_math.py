"""
Cortex Math — information theory, graph algorithms, midbrain metrics.

Extracted from cortex.py to reduce its size.
"""
from __future__ import annotations

from typing import Any

from .signal import Signal


def compute_math_modules(
    evidence_signals: list[Signal],
    forebrain: Any,
) -> tuple[float, dict[str, float]]:
    """Compute information theory and graph metrics on evidence.

    Returns:
        (entropy_bits, pagerank_dict)
    """
    entropy = 0.0
    pagerank: dict[str, float] = {}

    if evidence_signals:
        buckets = {f"ev_{i}": s.confidence for i, s in enumerate(evidence_signals)}
        try:
            er = forebrain.information_theory.shannon_entropy(buckets)
            entropy = er.entropy
        except Exception:
            pass

    if len(evidence_signals) >= 2:
        graph = forebrain.reasoning_graph
        for i, s in enumerate(evidence_signals):
            graph.add_node(f"ev_{i}", weight=s.confidence)
        for i in range(len(evidence_signals)):
            for j in range(i + 1, min(i + 4, len(evidence_signals))):
                graph.add_edge(f"ev_{i}", f"ev_{j}", weight=0.5)
        try:
            pr = graph.pagerank(max_iter=20)
            pagerank = pr.rankings
        except Exception:
            pass

    return entropy, pagerank


def midbrain_metrics(midbrain_result: Any) -> tuple[float, float, float]:
    """Extract average midbrain metrics.

    Returns:
        (avg_value_prediction, avg_salience_modulation, avg_inhibition)
    """
    avg_val = 0.0
    if midbrain_result.value_predictions:
        avg_val = sum(v["value"] for v in midbrain_result.value_predictions) / len(midbrain_result.value_predictions)

    avg_sal = 0.0
    if midbrain_result.salience_modulations:
        avg_sal = sum(m["modulated"] for m in midbrain_result.salience_modulations) / len(midbrain_result.salience_modulations)

    avg_inh = 0.0
    if midbrain_result.inhibition_decisions:
        avg_inh = sum(d["gated"] for d in midbrain_result.inhibition_decisions) / len(midbrain_result.inhibition_decisions)

    return avg_val, avg_sal, avg_inh
