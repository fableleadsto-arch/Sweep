"""
Verifier — Deterministic verification for task outputs.

§15: Every candidate must be verified using the strongest available
deterministic mechanism.
"""
from __future__ import annotations

import re
from typing import Any

from sweep_neural_mesh.training.task_generator import Task


class Verifier:
    """
    Verifies task outputs against ground truth.

    §15: Avoid using another LLM as the only verifier.
    Uses the task generator's ground truth and symbolic checks.
    """

    def verify(self, task: Task, prediction: str) -> tuple[bool, str]:
        """
        Verify a prediction against the task's expected output.

        Returns (correct: bool, method: str).
        """
        method = task.verification_method
        expected = task.expected_output.strip().upper()
        pred = prediction.strip().upper()

        if method in ("symbolic_chain", "chain_verification"):
            return self._verify_chain(task, pred, expected)
        elif method == "exact_value":
            return self._verify_exact(pred, expected)
        elif method == "temporal_chain":
            return self._verify_temporal(pred, expected)
        elif method == "evidence_count":
            return self._verify_evidence(pred, expected)
        elif method == "transitivity_check":
            return self._verify_transitivity(pred, expected)
        elif method == "graph_algorithm":
            return self._verify_graph(task, prediction)
        elif method in ("count_descendants", "count_connections", "count_steps"):
            return self._verify_count(pred, expected)
        elif method in ("exact_computation",):
            return self._verify_exact(pred, expected)
        elif method in ("cycle_check", "path_counting", "pattern_match",
                        "spatial_chain", "causal_chain", "ambiguity_check",
                        "knowledge_completeness"):
            return self._verify_exact(pred, expected)
        else:
            return self._verify_exact(pred, expected)

    def _verify_chain(self, task: Task, pred: str, expected: str) -> tuple[bool, str]:
        pred_clean = pred.replace(" ", "")
        expected_clean = expected.replace(" ", "")
        return pred_clean == expected_clean, "chain_match"

    def _verify_exact(self, pred: str, expected: str) -> tuple[bool, str]:
        return pred.strip() == expected.strip(), "exact_match"

    def _verify_temporal(self, pred: str, expected: str) -> tuple[bool, str]:
        return pred.strip() == expected.strip(), "temporal_match"

    def _verify_evidence(self, pred: str, expected: str) -> tuple[bool, str]:
        pred_upper = pred.upper().strip()
        expected_upper = expected.upper().strip()
        return pred_upper == expected_upper, "evidence_match"

    def _verify_transitivity(self, pred: str, expected: str) -> tuple[bool, str]:
        return pred.strip() == expected.strip(), "transitivity_match"

    def _verify_graph(self, task: Task, prediction: str) -> tuple[bool, str]:
        expected = task.expected_output.strip()
        pred = prediction.strip()

        if expected == "NONE" and pred.upper() == "NONE":
            return True, "graph_none_match"

        expected_set = set(e.strip() for e in expected.split(",") if e.strip())
        pred_set = set(p.strip() for p in pred.split(",") if p.strip())
        return expected_set == pred_set, "graph_set_match"

    def _verify_count(self, pred: str, expected: str) -> tuple[bool, str]:
        pred_num = re.findall(r'\d+', pred)
        expected_num = re.findall(r'\d+', expected)
        if pred_num and expected_num:
            return pred_num[0] == expected_num[0], "count_match"
        return pred.strip() == expected.strip(), "count_fallback"
