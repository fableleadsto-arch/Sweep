"""
LogicNode — a NeuralMesh node with capability `logic_reasoning`.

Wraps the FOLIO first-order-logic prover as a Mesh node so that reasoning
tasks can be dispatched through `mesh.analyze(data=task, task="logic_reasoning")`.

The node's execute_fn receives the task dict:
    {
      "id": int,
      "premises_fol": [str, ...],   # formal FOL premises (FOLIO ships these)
      "conclusion_fol": str,        # formal FOL conclusion to check
      "expected": "TRUE"|"FALSE"|"UNKNOWN",
    }
and returns:
    {
      "id", "verdict", "expected", "covered", "correct", "arity", "error"
    }
Honest by construction: verdicts are TRUE / FALSE / UNKNOWN, plus the
possibility of PARSE_FAIL (uncovered) — never a fake success.
"""
from __future__ import annotations

from typing import Any, Callable

from sweep_neural_mesh.core.node import (
    Framework,
    Modality,
    NeuralNode,
    NodeCostProfile,
    NodeSchema,
)
from sweep_neural_mesh.training.logic_data.folio_eval import prover_verdict


def _run_logic_prover(data: Any, **kwargs: Any) -> dict[str, Any]:
    """execute_fn for the logic node."""
    premises_fol: list[str] = data.get("premises_fol") or []
    conclusion_fol: str = data.get("conclusion_fol") or ""
    expected: str = data.get("expected") or "UNKNOWN"
    try:
        verdict = prover_verdict(premises_fol, conclusion_fol)
        covered = verdict != "PARSE_FAIL"
        correct = bool(covered and verdict == expected)
        return {
            "id": data.get("id"),
            "verdict": verdict,
            "expected": expected,
            "covered": covered,
            "correct": correct,
            "arity": len(premises_fol) + 1,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - surface as a failed task, not crash
        return {
            "id": data.get("id"),
            "verdict": "ERROR",
            "expected": expected,
            "covered": False,
            "correct": False,
            "arity": 0,
            "error": str(exc),
        }


def build_logic_node(
    cost_ms: float = 5.0,
    memory_mb: float = 64.0,
) -> NeuralNode:
    """Construct and return a registered-ready LogicNode."""
    return NeuralNode(
        name="folio_logic_node",
        framework=Framework.PURE_PYTHON,
        schema=NodeSchema(
            input_modalities=[Modality.TEXT],
            output_modalities=[Modality.TEXT],
        ),
        cost=NodeCostProfile(
            avg_latency_ms=cost_ms,
            memory_mb=memory_mb,
            flop_estimate=0,
        ),
        execute_fn=_run_logic_prover,
        capabilities=["logic_reasoning"],
        tags={"domain": "logic", "engine": "fol-resolver"},
    )


def make_execute_fn() -> Callable[[Any], dict[str, Any]]:
    """Return the node's execute_fn (handy for tests)."""
    return _run_logic_prover
