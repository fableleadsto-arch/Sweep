"""
Knowledge Mode — chunk-feed Sweep's reasoning neurons with real open datasets.

The mesh's grounder was built on the synthetic benchmark's task idioms. Feeding
real corpora (RuleTaker, ProofWriter, FOLIO, LogiQA) exposes the honest gap:
natural-language theories use idioms the grounder does not yet parse.

This module runs each task through Sweep's reasoning layer (NeuralProofMesh for
entailment-style questions, ReasoningCortex as a fallback), reports grounding
coverage + accuracy per source, and stores unmodified raw tasks so RSI can
selectively extend the grounder (never touching the source data).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from sweep_neural_mesh.neurons.proof_mesh import NeuralProofMesh, ProofMeshResult
from sweep_neural_mesh.training.logic_data.loader import TRUE, FALSE, UNKNOWN


@dataclass
class FeedRecord:
    """Result of feeding one task through Sweep's reasoning layer."""
    task_id: str
    domain: str
    expected: str
    predicted: str | None
    correct: bool | None
    grounded: bool
    conclusion: str
    confidence: float
    source: str = ""


@dataclass
class SourceStats:
    source: str
    tasks: int = 0
    grounded: int = 0
    correct: int = 0
    coverage_pct: float = 0.0
    accuracy_pct: float = 0.0

    def export(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "tasks": self.tasks,
            "grounded": self.grounded,
            "correct": self.correct,
            "coverage_pct": round(self.coverage_pct, 1),
            "accuracy_pct": round(self.accuracy_pct, 1),
        }


def _verdict_of(raw: str) -> str:
    """Map a mesh answer to TRUE/FALSE/UNKNOWN."""
    r = (raw or "").strip().upper()
    if r in ("TRUE", "YES", "SUPPORTED", "CONSISTENT", "ENTAILED", "1"):
        return TRUE
    if r in ("FALSE", "NO", "REFUTED", "CONTRADICTION", "0"):
        return FALSE
    return UNKNOWN


class KnowledgeFeed:
    """Feeds tasks into Sweep in bounded chunks, collecting records + stats."""

    def __init__(self, chunk_size: int = 200, use_fol: bool = True) -> None:
        self._mesh = NeuralProofMesh()
        self._chunk_size = chunk_size
        self._use_fol = use_fol
        self._eng = None  # lazy FOLEngine
        self.records: list[FeedRecord] = []

    def feed(self, tasks: Iterable[Any], chunk_size: int | None = None) -> list[FeedRecord]:
        """Process tasks in chunks (preserving raw tasks untouched)."""
        cs = chunk_size or self._chunk_size
        chunk: list[Any] = []
        batch_records: list[FeedRecord] = []
        seen = 0
        for task in tasks:
            chunk.append(task)
            if len(chunk) >= cs:
                batch_records.extend(self._process_chunk(chunk))
                chunk = []
            seen += 1
        if chunk:
            batch_records.extend(self._process_chunk(chunk))
        self.records.extend(batch_records)
        return batch_records

    def _process_chunk(self, chunk: list[Any]) -> list[FeedRecord]:
        out: list[FeedRecord] = []
        for task in chunk:
            out.append(self._feed_one(task))
        return out

    def _feed_one(self, task: Any) -> FeedRecord:
        source = (task.metadata or {}).get("source", task.domain)
        expected = str(task.expected_output).strip().upper()
        meta = task.metadata or {}
        grounded = False
        predicted = None
        correct = None
        conclusion = "unknown"
        confidence = 0.0

        # Entailment-style tasks (TRUE/FALSE/UNKNOWN): run through the FOL
        # reasoning neuron using the raw context + statement preserved in meta.
        if expected in (TRUE, FALSE, UNKNOWN):
            context = meta.get("context", "")
            statement = meta.get("statement", "")
            if self._use_fol and context and statement:
                if self._eng is None:
                    from sweep_neural_mesh.training.logic_data.fol_grounder import FOLEngine
                    self._eng = FOLEngine()
                self._eng.parse(context)
                q = self._eng.parse_question(statement)
                if q:
                    e, a, neg = q
                    v = self._eng.query(e, a, neg)
                    predicted = v
                    # grounded = we derived a definite verdict (not default-UNKNOWN)
                    grounded = (v != UNKNOWN)
                    correct = (v == expected)
                    conclusion = "fol_derived" if grounded else "fol_unknown"
                    confidence = 0.9 if grounded else 0.1
                    return FeedRecord(task.task_id, task.domain, expected,
                                      predicted, correct, grounded, conclusion,
                                      confidence, source)
                # FOL couldn't parse the statement -> fall through to mesh
            try:
                result = self._mesh.solve(task.input, [task.input])
            except Exception:
                return FeedRecord(task.task_id, task.domain, expected, None, False,
                                  False, "error", 0.0, source)
            raw = result.answer
            if raw is not None:
                predicted = _verdict_of(str(raw))
                correct = (predicted == expected)
            grounded = bool(result.bonds)
            conclusion = result.conclusion
            confidence = result.confidence
            return FeedRecord(task.task_id, task.domain, expected, predicted,
                              correct, grounded, conclusion, confidence, source)

        # Non-entailment tasks (e.g. MCQ) — run through the mesh, no FOL.
        try:
            result = self._mesh.solve(task.input, [task.input])
        except Exception:
            return FeedRecord(task.task_id, task.domain, expected, None, False,
                              False, "error", 0.0, source)
        predicted = str(result.answer) if result.answer is not None else None
        correct = (predicted is not None and predicted == expected)
        return FeedRecord(task.task_id, task.domain, expected, predicted,
                          correct, bool(result.bonds), result.conclusion,
                          result.confidence, source)

    def stats(self) -> dict[str, SourceStats]:
        by: dict[str, SourceStats] = {}
        for r in self.records:
            st = by.setdefault(r.source, SourceStats(source=r.source))
            st.tasks += 1
            if r.grounded:
                st.grounded += 1
            if r.correct is True:
                st.correct += 1
        for st in by.values():
            st.coverage_pct = 100.0 * st.grounded / max(1, st.tasks)
            graded = sum(1 for r in self.records if r.source == st.source and r.correct is not None)
            st.accuracy_pct = 100.0 * st.correct / max(1, graded) if graded else 0.0
        return by

    def export_stats(self) -> list[dict[str, Any]]:
        return [st.export() for st in self.stats().values()]
