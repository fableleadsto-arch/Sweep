"""
FOLIO prover evaluator — honest, non-overfit measurement of the FOL reasoner.

Streams the FOLIO dev split and runs :mod:`folio_prover` on the *formal* FOL
that FOLIO ships (premises-FOL / conclusion-FOL). Reports:

    * coverage  : fraction of rows whose premises+conclusion all parse
    * accuracy  : agreement with gold label on the covered subset
    * confusions: which gold classes the prover answers correctly / misses

This is deliberately separate from the NL-parser pipeline: it measures the
reasoner on the dataset's own formal structure, with no tuning to samples.
"""
from __future__ import annotations

import sys
from typing import Iterator

from sweep_neural_mesh.training.logic_data import folio_prover as F
from sweep_neural_mesh.training.logic_data.loader import (
    FOLIO_HF, TRUE, FALSE, UNKNOWN, fetch_hf_streaming, _label_to_verdict,
)


def prover_verdict(premises_fol: list[str], conclusion_fol: str) -> str:
    """Return TRUE / FALSE / UNKNOWN from the prover, or 'PARSE_FAIL'."""
    try:
        kb = set()
        for p in premises_fol:
            for cl in F.cnf(F.parse_fol(p)):
                kb.add(cl)
        neg_conc = F._cnf_clauses(F._skolem(F.nnf(F._neg(F.parse_fol(conclusion_fol)))))
        if F.saturate(set(kb) | set(neg_conc)):
            return TRUE
        pos_conc = F._cnf_clauses(F._skolem(F.nnf(F.parse_fol(conclusion_fol))))
        if F.saturate(set(kb) | set(pos_conc)):
            return FALSE
        return UNKNOWN
    except Exception:
        return "PARSE_FAIL"


def evaluate(max_rows: int = 250) -> dict:
    rows = fetch_hf_streaming(FOLIO_HF, "validation", max_rows)
    total = covered = 0
    correct = 0
    by_gold: dict[str, list] = {}
    uncertainties = 0
    samples_hit = 0
    samples_miss = 0
    per_class_correct: dict[str, int] = {}
    per_class_total: dict[str, int] = {}
    for row in rows:
        premises_fol = [str(p).strip() for p in
                        str(row.get("premises-FOL") or "").splitlines() if str(p).strip()]
        conclusion_fol = str(row.get("conclusion-FOL") or "").strip()
        if not premises_fol or not conclusion_fol:
            continue
        gold = _label_to_verdict(str(row.get("label") or ""))
        verdict = prover_verdict(premises_fol, conclusion_fol)
        if verdict == "PARSE_FAIL":
            continue
        covered += 1
        by_gold.setdefault(gold, []).append(verdict)
        per_class_total[gold] = per_class_total.get(gold, 0) + 1
        if verdict == gold:
            correct += 1
            per_class_correct[gold] = per_class_correct.get(gold, 0) + 1
            samples_hit += 1
        else:
            samples_miss += 1
        if verdict == UNKNOWN:
            uncertainties += 1

    acc = (correct / covered) if covered else 0.0
    return {
        "rows_checked": len(rows), "covered": covered,
        "coverage": (covered / len(rows)) if rows else 0.0,
        "correct": correct, "accuracy": acc, "uncertain": uncertainties,
        "per_class_total": per_class_total, "per_class_correct": per_class_correct,
    }


def _fmt(d: dict) -> str:
    lines = [
        "FOLIO prover eval (formal FOL) — honest, no sample tuning",
        f"  rows checked          : {d['rows_checked']}",
        f"  covered (parsed)      : {d['covered']}  ({d['coverage']*100:.1f}%)",
        f"  correct               : {d['correct']}",
        f"  accuracy (on covered) : {d['accuracy']*100:.1f}%",
        f"  UNKNOWN verdicts      : {d['uncertain']}",
        "  per class (correct/total):",
    ]
    for g in (TRUE, FALSE, UNKNOWN):
        tot = d["per_class_total"].get(g, 0)
        ok = d["per_class_correct"].get(g, 0)
        lines.append(f"      {g:8s}: {ok}/{tot}")
    return "\n".join(lines)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    print(_fmt(evaluate(n)))
