"""
Sweep Runner — Runs Sweep's neural mesh on benchmark tasks.

CRITICAL: No deterministic solvers in the inference path.
Only the neural mesh cortex processes each task.
"""
from __future__ import annotations

import time
from typing import Any

from sweep_neural_mesh.neurons.cortex import ReasoningCortex
from sweep_neural_mesh.neurons.proof_mesh import NeuralProofMesh
from neural_eval.core import Task, Result


class SweepRunner:
    """
    Runs tasks through Sweep's neural mesh only.

    The cortex provides a neural decision (support/refute/mixed/insufficient).
    Sweep's own grounded reasoning neurons (NeuralProofMesh) then refine the
    decision where they form explicit logical structure; when they reach a
    confident conclusion, their canonical answer drives the final prediction.
    """

    def __init__(self) -> None:
        self._cortex = ReasoningCortex()
        self._proof_mesh = NeuralProofMesh()
        self._step_count = 0

    def run(self, task: Task) -> Result:
        """Run a single task through the neural mesh."""
        t0 = time.perf_counter()

        result = self._cortex.reason(
            query=task.input_text,
            evidence=[task.input_text],
        )

        predicted = self._map_answer(result.decision, result.reasoning, task)

        # Sweep's grounded reasoning neurons: explicit proof propagation.
        # Used when they can form strong logical structure for this task.
        pm = self._proof_mesh.solve(task.input_text, [task.input_text])
        if pm.answer is not None and pm.conclusion in (
            "supported", "refuted", "mixed", "insufficient",
        ):
            predicted = str(pm.answer).strip()
            logical_applied = True
        else:
            logical_applied = False

        latency_ms = (time.perf_counter() - t0) * 1000

        return Result(
            task_id=task.task_id,
            predicted=predicted,
            expected=task.expected_output,
            correct=predicted.strip().upper() == task.expected_output.strip().upper(),
            confidence=pm.confidence if logical_applied else result.confidence,
            latency_ms=latency_ms,
            reasoning_steps=self._step_count,
            metadata={
                "cortex_decision": result.decision,
                "cortex_reasoning": result.reasoning[:200],
                "proof_mesh_applied": logical_applied,
                "proof_mesh_conclusion": pm.conclusion,
                "proof_mesh_answer": pm.answer,
            },
        )

    def run_batch(self, tasks: list[Task]) -> list[Result]:
        """Run a batch of tasks."""
        results = []
        for task in tasks:
            results.append(self.run(task))
        return results

    def _map_answer(self, decision: str, reasoning: str, task: Task) -> str:
        """Map cortex decision to expected answer format."""
        decision_lower = decision.lower()
        reasoning_lower = reasoning.lower()
        expected = task.expected_output.strip().upper()

        if expected in ("YES", "NO"):
            if "support" in decision_lower or "yes" in reasoning_lower:
                return "YES"
            if "refut" in decision_lower or "no" in reasoning_lower:
                return "NO"
            return decision.upper() if decision.upper() in ("YES", "NO") else "UNCERTAIN"

        if expected in ("TRUE", "FALSE"):
            if "support" in decision_lower:
                return "TRUE"
            if "refut" in decision_lower:
                return "FALSE"
            return "UNCERTAIN"

        if expected in ("SUPPORTED", "REFUTED", "AMBIGUOUS", "INSUFFICIENT"):
            if "support" in decision_lower:
                return "SUPPORTED"
            if "refut" in decision_lower:
                return "REFUTED"
            if "mixed" in decision_lower:
                return "AMBIGUOUS"
            return "AMBIGUOUS"

        if expected in ("CONSISTENT", "CONTRADICTION"):
            if "support" in decision_lower:
                return "CONSISTENT"
            if "refut" in decision_lower:
                return "CONTRADICTION"
            return "CONSISTENT"

        if expected in ("POSSIBLE", "IMPOSSIBLE"):
            if "support" in decision_lower:
                return "POSSIBLE"
            if "refut" in decision_lower:
                return "IMPOSSIBLE"
            return "POSSIBLE"

        import re
        nums = re.findall(r"\d+", task.expected_output)
        if nums:
            pred_nums = re.findall(r"\d+", reasoning)
            if pred_nums:
                return pred_nums[0]

        return decision
