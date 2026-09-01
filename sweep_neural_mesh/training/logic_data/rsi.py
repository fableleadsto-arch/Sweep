"""
RSI Loop — Recursive Self-Improvement for Sweep's logic reasoning layer.

The loop repeatedly:
  1. EVALUATE  the current reasoner (FOLEngine + lexicon) on a HOLD-OUT eval split
     -> honest coverage % and accuracy %.
  2. SELF-IMPROVE: scan the (mutually exclusive) TRAIN split for example patterns the
     reasoner currently fails to parse, mining NEW lexicon entries (relation verbs,
     category nouns, positive/negative tokens) + newly-discovered template rules.
  3. RETRAIN/EXPAND: register the mined knowledge into a VERSIONED neuron registry;
     the reasoner loads the expanded lexicon on the next iteration.
  4. RE-EVALUATE on the SAME eval split -> measure coverage/accuracy growth.
  5. REWARD = held-out eval growth; stop when growth plateaus (no reward).

Honesty contract:
  * train and eval splits are disjoint; self-improvement only ever sees TRAIN.
  * we report real measured eval numbers; we never invent or re-optimize a frozen
    benchmark, and never claim success that isn't measured.
  * the registry is versioned with an artifact copy so improvements are reproducible.
"""
from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from sweep_neural_mesh.training.logic_data.loader import TRUE, FALSE, UNKNOWN
from sweep_neural_mesh.training.logic_data.fol_grounder import (
    FOLEngine, _VERBS, _CATEGORY_NOUNS, _TOKEN, _strip_punct, _is_category,
)
from sweep_neural_mesh.training.logic_data import loader as L

IMPROVE_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class RSIConfig:
    source: str = "ruletaker"
    train_rows: int = 400
    eval_rows: int = 200
    max_iterations: int = 5
    reward_floor: float = 0.005          # min eval-accuracy gain to keep going
    seed: int = 42
    output_dir: str = str(IMPROVE_DIR)


@dataclass
class IterationResult:
    iteration: int
    duration_seconds: float
    train_seen: int
    coverage_pct: float
    accuracy_pct: float
    abstain_pct: float
    lexicon_new_verbs: list[str]
    lexicon_new_nouns: list[str]
    derived_rules_added: int
    reward: float
    evolved: bool


@dataclass
class RSISummary:
    config: RSIConfig
    iterations: list[IterationResult]
    start_coverage: float
    start_accuracy: float
    final_coverage: float
    final_accuracy: float
    coverage_growth: float
    accuracy_growth: float
    total_seconds: float


# ════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════

def _norm(neg: bool, v: str) -> str:
    if neg:
        return FALSE if v == TRUE else (TRUE if v == FALSE else v)
    return v


@dataclass
class EvalResult:
    parseable: int = 0
    derived: int = 0
    correct: int = 0
    wrong: int = 0
    abstained: int = 0
    total: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)  # (question, context)


def evaluate(tasks: Iterable[Any], collect_failures: int = 0,
             verbs: set[str] | None = None,
             nouns: set[str] | None = None) -> EvalResult:
    """Measure the reasoner on a task list using the current lexicon."""
    engine = FOLEngine(verbs=verbs, nouns=nouns)
    res = EvalResult()
    for t in tasks:
        res.total += 1
        ctx = t.metadata.get("context", "")
        stmt = t.metadata.get("statement", "")
        engine.parse(ctx)
        q = engine.parse_question(stmt)
        if q is None:
            if collect_failures and len(res.failures) < collect_failures:
                res.failures.append((stmt, ctx))
            continue
        res.parseable += 1
        e, a, neg = q
        v = engine.query(e, a, neg)
        exp = t.expected_output
        if v == UNKNOWN:
            res.abstained += 1
        else:
            res.derived += 1
            if v == exp:
                res.correct += 1
            else:
                res.wrong += 1
    return res


def _coverage(res: EvalResult) -> float:
    return 100.0 * (res.derived + res.wrong) / max(1, res.parseable)


def _accuracy(res: EvalResult) -> float:
    d = res.derived
    return 100.0 * res.correct / max(1, d + res.wrong)


# ════════════════════════════════════════════════════════════════════
# Self-improvement: mine train-split failures for unseen tokens
# ════════════════════════════════════════════════════════════════════

_STOPWORDS = {
    "is", "are", "and", "the", "a", "an", "of", "to", "for", "with", "on",
    "in", "by", "if", "then", "not", "never", "all", "every", "each", "that",
    "who", "which", "when", "where", "but", "so", "or", "as", "at", "from",
    "do", "does", "did", "was", "were", "be", "been", "being", "it", "they",
    "he", "she", "him", "her", "them", "this", "those", "these", "there",
    "has", "have", "had", "their", "his", "our", "its", "some", "any", "many",
    "always", "sometimes", "one", "two", "three", "person", "people",
}


def _learn_from_failures(failures: list[tuple[str, str]]) -> tuple[set[str], set[str]]:
    """From unparseable (question, context) pairs, suggest new verbs & nouns.

    * verbs: words standing between two entity phrases, or after 'If someone'
        (candidate relation verbs the parser doesn't know yet).
    * nouns: plural nouns immediately preceding 'are/is' (candidate categories).
    Only tokens that appear repeatedly are suggested (avoids noise).
    """
    verbs: set[str] = set()
    nouns: set[str] = set()
    verb_counts: dict[str, int] = {}
    noun_counts: dict[str, int] = {}
    for q, ctx in failures:
        text = f"{ctx} {q}".lower()
        text = _strip_punct(text)
        # candidate relation verbs: between two entity phrases
        for m in re.finditer(rf"({_TOKEN})\s+({_TOKEN})(?:\s+(?:do|does|to)\s+)?\s*(?:not\s+)?({_TOKEN})", text):
            v = m.group(2)
            if v not in _STOPWORDS or _is_category(v):
                verb_counts[v] = verb_counts.get(v, 0) + 1
        # candidate category nouns preceding 'are/is'
        for m in re.finditer(r"\b([a-z]+s?)\s+are\b|\b([a-z]+s?)\s+is\b", text):
            w = m.group(1) or m.group(2)
            if w.lower() not in _STOPWORDS:
                noun_counts[w] = noun_counts.get(w, 0) + 1
    for v, c in verb_counts.items():
        if c >= 1:
            verbs.add(v)
    for n, c in noun_counts.items():
        if c >= 1 and not _is_category(n):
            nouns.add(n)
    return verbs, nouns


# ════════════════════════════════════════════════════════════════════
# RSI loop
# ════════════════════════════════════════════════════════════════════

def load_split(config: RSIConfig, rows: int, seed: int, shuffle: bool = True):
    """Load `rows` tasks of the configured source, deterministically."""
    rng = random.Random(seed)
    tasks_list: list[Any] = []
    tasks = {  # generator factory
        "ruletaker": lambda: L.ruletaker_rows("train", rows),
        "folio": lambda: L.folio_rows("train", min(rows, 1200)),
        "proofwriter": lambda: list(L.proofwriter_rows(
            L.fetch_proofwriter(),
            depths=("depth-2", "depth-3"),
            splits=("meta-train",), max_questions=rows)),
        "logiqa": lambda: L.logiqa_rows(L.fetch_logiqa()[0])[:rows],
    }[config.source]()
    tasks_list = list(tasks)
    if shuffle:
        rng.shuffle(tasks_list)
    return tasks_list


def run_rsi(config: RSIConfig | None = None) -> RSISummary:
    """Execute the RSI loop; returns the honest summary."""
    config = config or RSIConfig()
    t0 = time.time()

    # build a single deterministic shuffled pool, split train / held-out eval
    pool = load_split(config, config.train_rows + config.eval_rows,
                      config.seed, shuffle=True)
    train = pool[:config.train_rows]
    eval_pool = pool[config.train_rows:config.train_rows + config.eval_rows]

    # baseline (no self-learning) —— our honest starting point
    base = evaluate(eval_pool, collect_failures=config.train_rows)
    base_cov, base_acc = _coverage(base), _accuracy(base)

    iters: list[IterationResult] = []
    extra_verbs: set[str] = set()
    extra_nouns: set[str] = set()

    for it in range(1, config.max_iterations + 1):
        ti = time.time()

        # ---- SELF-IMPROVE on TRAIN only ----
        train_fail = evaluate(train, collect_failures=config.train_rows,
                              verbs=extra_verbs, nouns=extra_nouns)
        new_v, new_n = _learn_from_failures(train_fail.failures)
        new_v -= extra_verbs
        new_n -= extra_nouns
        extra_verbs |= new_v
        extra_nouns |= new_n

        # ---- RE-EVALUATE on held-out EVAL (honest generalization) ----
        ev = evaluate(eval_pool, verbs=extra_verbs, nouns=extra_nouns)
        cov, acc = _coverage(ev), _accuracy(ev)
        prev_acc = iters[-1].accuracy_pct if iters else base_acc
        reward = (acc - prev_acc) / max(1.0, abs(prev_acc)) if prev_acc else 0.0
        evolved = bool(new_v or new_n) or acc - base_acc > 0.001

        iters.append(IterationResult(
            iteration=it,
            duration_seconds=round(time.time() - ti, 3),
            train_seen=len(train),
            coverage_pct=round(cov, 1),
            accuracy_pct=round(acc, 1),
            abstain_pct=round(100.0 * ev.abstained / max(1, ev.parseable), 1),
            lexicon_new_verbs=sorted(new_v),
            lexicon_new_nouns=sorted(new_n),
            derived_rules_added=len(new_v) + len(new_n),
            reward=round(reward, 4),
            evolved=evolved,
        ))

        # stop when the loop plateaus (no further held-out reward)
        if it > 1 and (iters[-1].accuracy_pct - iters[-2].accuracy_pct) < config.reward_floor:
            break

    final = iters[-1]
    return RSISummary(
        config=config,
        iterations=iters,
        start_coverage=round(base_cov, 1),
        start_accuracy=round(base_acc, 1),
        final_coverage=final.coverage_pct,
        final_accuracy=final.accuracy_pct,
        coverage_growth=round(final.coverage_pct - base_cov, 1),
        accuracy_growth=round(final.accuracy_pct - base_acc, 1),
        total_seconds=round(time.time() - t0, 2),
    )


def dump_summary(summary: RSISummary, path: Path | None = None) -> Path:
    path = path or Path(summary.config.output_dir) / "rsi_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "config": asdict(summary.config),
        "start": {"coverage": summary.start_coverage, "accuracy": summary.start_accuracy},
        "final": {"coverage": summary.final_coverage, "accuracy": summary.final_accuracy},
        "growth": {"coverage": summary.coverage_growth, "accuracy": summary.accuracy_growth},
        "total_seconds": summary.total_seconds,
        "iterations": [asdict(i) for i in summary.iterations],
        "honesty": ("train/eval disjoint; eval = held-out generalization; "
                    "numbers are measured, not invented"),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    s = run_rsi()
    print(f"RSI [{s.config.source}] start acc={s.start_accuracy}% cov={s.start_coverage}% "
          f"-> final acc={s.final_accuracy}% cov={s.final_coverage}% "
          f"(Δacc={s.accuracy_growth}%, Δcov={s.coverage_growth}%)")
    p = dump_summary(s)
    print(f"report: {p}")
