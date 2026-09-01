"""
Self-terminating, honesty-constrained 1-hour RSI driver for the logic layer.

Runs TWO self-improvement phases sequentially, each with its own wall-clock
sub-budget, all anchored to a HARD global deadline (default 1 hour). At the
deadline the process exits by itself and writes a measured report.

  Phase A — FOL-prover grammar generalization:
      Mine the TRAIN split's formal FOL for formula strings the parser cannot
      parse. Cluster failures by cause. A *principled* grammar gap is one that
      is general (repeats across several distinct examples) AND is not a data
      malformation (e.g. unbalanced parentheses = annotation error, not a
      missing parser capability). Each iteration: sample a fresh train slice,
      find new general gaps, and re-measure HONEST held-out eval coverage /
      accuracy. We do NOT tune the parser to individual sample sentences and
      we do NOT mutate live source mid-run; discovered gaps are recorded with
      evidence as improvement candidates, and headroom is measured honestly.

  Phase B — FOLEngine lexicon self-improvement:
      The existing lexicon-mining RSI (rsi.run_rsi internals) extended so it
      keeps improving until its sub-deadline or an honest plateau.

Honesty contract (unchanged):
  * train and eval splits are always disjoint
  * every number is measured on the held-out eval split, never invented
  * we report plateau truthfully (small growth near grammar saturation is the
    real result), and we never re-optimize a frozen benchmark.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
import time as _time
from collections import Counter
from pathlib import Path
from typing import Any

from sweep_neural_mesh.training.logic_data import folio_prover as F
from sweep_neural_mesh.training.logic_data.loader import (
    FOLIO_HF, TRUE, FALSE, UNKNOWN, fetch_hf_streaming, _label_to_verdict,
)
from sweep_neural_mesh.training.logic_data.rsi import (
    run_rsi, dump_summary, RSIConfig, evaluate, _coverage, _accuracy,
)
from sweep_neural_mesh.training.logic_data.loader import (
    ruletaker_rows as _ruletaker_rows,
)
import sweep_neural_mesh.training.logic_data.loader as L

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
LOG = RESULTS / "rsi_hour.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def hard_deadline(start: float, budget: float) -> bool:
    return (time.time() - start) >= budget


# ─────────────────────────────────────────────────────────────────────
# Phase A: FOL-prover grammar coverage audit + measured headroom
# ─────────────────────────────────────────────────────────────────────

def fol_premises(row) -> list[str]:
    return [str(p).strip() for p in
            str(row.get("premises-FOL") or "").splitlines() if str(p).strip()]


def fol_conclusion(row) -> str:
    return str(row.get("conclusion-FOL") or "").strip()


def load_folio_train() -> list[dict]:
    return fetch_hf_streaming(FOLIO_HF, "train", 20000)


def load_folio_eval() -> list[dict]:
    return fetch_hf_streaming(FOLIO_HF, "validation", 250)


def is_balanced_parens(s: str) -> bool:
    d = 0
    for ch in s:
        if ch in ("(", "[", "{"):
            d += 1
        elif ch in (")", "]", "}"):
            d -= 1
        if d < 0:
            return False
    return d == 0


EVAL_SET_SIZE = 50
EVAL_SEED = 7
MAX_KB_CLAUSES = 20       # per-row premise budget (documented eval config)
ROW_DEADLINE = 1.0        # seconds per row across both refutation checks
SAT_MAX_CLAUSES = 1200


def fixed_eval_set(rows, size: int = EVAL_SET_SIZE, seed: int = EVAL_SEED) -> list[dict]:
    """Deterministic, fixed held-out eval benchmark (never touched by training)."""
    rng = random.Random(seed)
    pool = [r for r in rows if fol_premises(r) and fol_conclusion(r)]
    rng.shuffle(pool)
    return pool[:size]


def eval_folio(rows, parse_only: bool = False, row_timeout: float | None = None) -> dict:
    """Measure the FOL prover on a held-out eval benchmark (bounded cost).

    row_timeout defaults to ROW_DEADLINE; per-row premise clauses are capped at
    MAX_KB_CLAUSES so that very large stories do not dominate a single pass.
    """
    if row_timeout is None:
        row_timeout = ROW_DEADLINE
    covered = correct = total = derive = 0
    by_g = {}
    for row in rows:
        pf = fol_premises(row)
        cf = fol_conclusion(row)
        if not pf or not cf:
            continue
        gold = _label_to_verdict(str(row.get("label") or ""))
        total += 1
        row_deadline = _time.monotonic() + row_timeout
        try:
            kb = set()
            for p in pf:
                for cl in F.cnf(F.parse_fol(p)):
                    kb.add(cl)
                    if len(kb) >= MAX_KB_CLAUSES:
                        break
                if len(kb) >= MAX_KB_CLAUSES:
                    break
            ng = F._cnf_clauses(F._skolem(F.nnf(F._neg(F.parse_fol(cf)))))
            if F.saturate(set(kb) | set(ng), deadline=row_deadline,
                          max_clauses=SAT_MAX_CLAUSES):
                v = TRUE
            else:
                pg = F._cnf_clauses(F._skolem(F.nnf(F.parse_fol(cf))))
                if F.saturate(set(kb) | set(pg), deadline=row_deadline,
                              max_clauses=SAT_MAX_CLAUSES):
                    v = FALSE
                else:
                    v = UNKNOWN
        except Exception:
            continue
        covered += 1
        by_g[gold] = by_g.get(gold, 0) + 1
        if v == gold:
            correct += 1
        if v != UNKNOWN:
            derive += 1
    return {
        "total": total, "covered": covered,
        "coverage": 100.0 * covered / max(1, total),
        "correct": correct, "accuracy": 100.0 * correct / max(1, covered),
        "derived": derive,
    }


def grammar_gaps_from_train(train_rows, seed: int, limit: int = 400,
                            seen: Counter | None = None,
                            examples: dict | None = None) -> Counter:
    """Sample a train slice, cluster unparseable formulas by cause."""
    if seen is None:
        seen = Counter()
    if examples is None:
        examples = {}
    rng = random.Random(seed)
    sample = rng.sample(train_rows, min(limit, len(train_rows)))
    for row in sample:
        for s in fol_premises(row) + [fol_conclusion(row)]:
            if not s:
                continue
            try:
                F.parse_fol(s)
                continue
            except Exception as e:
                msg = str(e).splitlines()[0]
            key = msg
            # distinguish data malformation from grammar gap
            if not is_balanced_parens(s) and "RPAREN" in msg:
                key = "DATA_MALFORMED(unbalanced parens)"
            seen[key] += 1
            examples.setdefault(key, s)
    return seen


def phase_a(budget: float, eval_rows, train_rows, out: dict) -> None:
    a0 = time.time()
    log(f"PHASE A start (budget {budget:.0f}s): FOL-prover grammar audit")
    seen = Counter()
    examples = {}
    results = []
    iter_i = 0
    eval_set = fixed_eval_set(eval_rows)
    base = eval_folio(eval_set)
    log(f"  A eval-set={len(eval_set)} base cov={base['coverage']:.1f}% "
        f"acc={base['accuracy']:.1f}% covered={base['covered']}/{base['total']}")
    out["phase_a"] = {"base": base, "eval_set_size": len(eval_set),
                      "iterations": results}
    prev_n_gaps = None
    stable = 0
    ev = base
    while not hard_deadline(a0, budget):
        iter_i += 1
        t_i = time.time()
        before_len = sum(seen.values())
        grammar_gaps_from_train(train_rows, seed=1000 + iter_i,
                                seen=seen, examples=examples)
        new_this = sum(seen.values()) - before_len
        n_gaps = len([k for k in seen
                      if k != "DATA_MALFORMED(unbalanced parens)"])
        ev = eval_folio(eval_set)
        results.append({
            "iteration": iter_i, "new_parse_failures_this_iter": new_this,
            "distinct_grammar_gap_classes": n_gaps,
            "eval_coverage_pct": round(ev["coverage"], 1),
            "eval_accuracy_pct": round(ev["accuracy"], 1),
            "eval_correct": ev["correct"], "eval_covered": ev["covered"],
            "elapsed_iter": round(time.time() - t_i, 1),
        })
        log(f"  A iter {iter_i}: +{new_this} new pf; n_gaps={n_gaps}; "
            f"eval cov={ev['coverage']:.1f}% acc={ev['accuracy']:.1f}% "
            f"({round(time.time()-a0,0):.0f}s)")
        if hard_deadline(a0, budget):
            break
        # honest plateau: grammar gap classes unchanged across fresh train slices
        if n_gaps == prev_n_gaps:
            stable += 1
        else:
            stable = 0
        prev_n_gaps = n_gaps
        if stable >= 2:
            log("  A plateau (no new general grammar gaps on fresh train) — stop")
            break
        if len(results) >= 8:
            break
    final = ev
    gaps = {k: {"count": c, "example": examples.get(k, ""),
                "data_malformed": k == "DATA_MALFORMED(unbalanced parens)"}
            for k, c in seen.most_common()}
    out["phase_a"]["final"] = final
    out["phase_a"]["grammar_gap_classes"] = gaps
    out["phase_a"]["gaps_non_malformed"] = [
        {"cause": k, "count": c, "example": examples.get(k, "")
         } for k, c in seen.items()
         if k != "DATA_MALFORMED(unbalanced parens)"]
    log(f"PHASE A done in {round(time.time()-a0,0):.0f}s. "
        f"final cov={final['coverage']:.1f}% acc={final['accuracy']:.1f}%")


# ─────────────────────────────────────────────────────────────────────
# Phase B: FOLEngine lexicon RSI until sub-deadline or plateau
# ─────────────────────────────────────────────────────────────────────

def phase_b(budget: float, out: dict) -> None:
    b0 = time.time()
    log(f"PHASE B start (budget {budget:.0f}s): FOLEngine lexicon RSI")
    iterations = []
    it = 0
    pool = None
    while not hard_deadline(b0, budget):
        it += 1
        cfg = RSIConfig(source="ruletaker", train_rows=120, eval_rows=80,
                        max_iterations=4, reward_floor=0.002, seed=42)
        cfg.output_dir = str(RESULTS)
        try:
            s = run_rsi(cfg)
        except Exception as e:
            log(f"  B iter {it} error: {e}")
            break
        entry = {
            "iteration": it,
            "start_acc": s.start_accuracy, "final_acc": s.final_accuracy,
            "start_cov": s.start_coverage, "final_cov": s.final_coverage,
            "d_acc": s.accuracy_growth, "d_cov": s.coverage_growth,
            "iters_run": len(s.iterations),
            "elapsed": s.total_seconds,
        }
        iterations.append(entry)
        log(f"  B iter {it}: acc {s.start_accuracy}% -> {s.final_accuracy}% "
            f"(d={s.accuracy_growth:+}%) cov {s.start_coverage}%->{s.final_coverage}% "
            f"iters_run={len(s.iterations)} ({s.total_seconds:.0f}s)")
        # honest plateau: no further reward on a fresh pass
        if s.accuracy_growth <= 0 and len(iterations) >= 2:
            log("  B plateau (no further held-out reward) — stopping B early")
            break
    out["phase_b"] = {"iterations": iterations,
                      "final": iterations[-1] if iterations else None}


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────

def main():
    total_budget = float(sys.argv[1]) if len(sys.argv) > 1 else 3600.0
    a_budget = float(sys.argv[2]) if len(sys.argv) > 2 else total_budget * 0.5
    b_budget = float(sys.argv[3]) if len(sys.argv) > 3 else max(0.0, total_budget - a_budget)
    t0 = time.time()
    out = {"started": time.time(),
           "budget_seconds": total_budget, "phase_a_budget": a_budget,
           "phase_b_budget": b_budget,
           "honesty": "train/eval disjoint; all numbers measured on held-out eval; "
                      "no sample tuning; no fake growth"}
    log(f"RSI-HOUR driver start. total budget={total_budget:.0f}s "
        f"(A={a_budget:.0f}s B={b_budget:.0f}s)")

    train_rows = load_folio_train()
    eval_rows = load_folio_eval()
    log(f"train={len(train_rows)} eval={len(eval_rows)} loaded")

    phase_a(a_budget, eval_rows, train_rows, out)
    out["phase_a_elapsed"] = round(time.time() - t0, 1)

    if not hard_deadline(t0, total_budget):
        phase_b(min(b_budget, t0 + total_budget - time.time()), out)
    out["total_elapsed"] = round(time.time() - t0, 1)

    report = RESULTS / "rsi_hour_report.json"
    report.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"DONE. total elapsed={out['total_elapsed']}s. "
        f"report={report}. self-terminating.")


if __name__ == "__main__":
    main()
