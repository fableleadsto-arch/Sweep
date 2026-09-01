"""
Logic Dataset Loader — pulls open reasoning datasets and converts them into
Sweep's trainer Task schema, keeping the RAW source data intact on disk.

Supported sources:
  * ProofWriter  (aristo-data-public S3 zip, meta-*.jsonl)  [open]
  * LogiQA       (plain TXT files, no dependencies)          [open]
  * RuleTaker    (HuggingFace hitachi-nlp/ruletaker)         [open]
  * FOLIO        (HuggingFace tasksource/folio)              [open]

Honesty note: the raw records are never rewritten. We only READ them and emit
Sweep Tasks (with a pointer back to the source row). Download/caching is
explicit and bounded by `max_rows` for streaming sources.
"""
from __future__ import annotations

import json
import os
import random
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterator

from sweep_neural_mesh.training.task_generator import Task

DATA_DIR = Path(__file__).resolve().parents[3] / "neural_eval" / "datasets"

PROOFWRITER_URL = (
    "https://aristo-data-public.s3.amazonaws.com/proofwriter/"
    "proofwriter-dataset-V2020.12.3.zip"
)
LOGIIQA_BASE = "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/"
RULE_TAKER_HF = "hitachi-nlp/ruletaker"
FOLIO_HF = "tasksource/folio"


# Truth-label vocabulary normalised across datasets.
TRUE = "TRUE"
FALSE = "FALSE"
UNKNOWN = "UNKNOWN"


class DatasetFetchError(RuntimeError):
    """Raised when a dataset cannot be fetched or parsed."""


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════
# FETCH
# ════════════════════════════════════════════════════════════════════

def fetch_proofwriter(url: str = PROOFWRITER_URL, force: bool = False) -> Path:
    """Download (once) and return the ProofWriter zip path."""
    zp = DATA_DIR / os.path.basename(url)
    _ensure_dir(DATA_DIR)
    if not zp.exists() or force:
        print(f"[loader] downloading ProofWriter -> {zp.name}")
        urllib.request.urlretrieve(url, zp)
    return zp


def fetch_logiqa(splits=("Train", "Eval", "Test"), force: bool = False) -> list[Path]:
    """Download LogiQA plain-text files and return their paths."""
    _ensure_dir(DATA_DIR / "logiqa")
    paths = []
    for split in splits:
        name = f"{split}.txt"
        p = DATA_DIR / "logiqa" / name
        if not p.exists() or force:
            print(f"[loader] downloading LogiQA {name}")
            urllib.request.urlretrieve(LOGIIQA_BASE + name, p)
        paths.append(p)
    return paths


def fetch_hf_streaming(repo: str, split: str, max_rows: int) -> list[dict[str, Any]]:
    """Stream `max_rows` rows from a HuggingFace dataset (bounded download)."""
    from datasets import load_dataset
    ds = load_dataset(repo, split=split, streaming=True)
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        rows.append({k: v for k, v in row.items()})
    return rows


# ════════════════════════════════════════════════════════════════════
# PARSING -> (raw preserved, normalised task)
# ════════════════════════════════════════════════════════════════════

def _label_to_verdict(label: str) -> str:
    lab = (label or "").strip().lower()
    if lab in ("entailment", "true", "yes", "1", "entailed", "supported"):
        return TRUE
    if lab in ("contradiction", "false", "no", "0", "refuted"):
        return FALSE
    return UNKNOWN


def _split_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def proofwriter_rows(arch_in: str,
                     depths=("depth-2", "depth-3"),
                     splits=("meta-train", "meta-dev", "meta-test"),
                     max_questions: int | None = None) -> Iterator[Task]:
    """Yield Tasks from ProofWriter meta files. Ground-truth = each Q answer."""
    zp = Path(arch_in)
    base = "proofwriter-dataset-V2020.12.3/CWA"
    seen = set()
    emitted = 0
    with zipfile.ZipFile(zp) as z:
        for depth in depths:
            for split in splits:
                rel = f"{base}/{depth}/{split}.jsonl"
                if rel not in z.namelist():
                    continue
                for line in z.open(rel).read().decode("utf-8").splitlines():
                    obj = json.loads(line)
                    questions = obj.get("questions") or {}
                    for qkey in sorted(questions.keys()):
                        if max_questions is not None and emitted >= max_questions:
                            return
                        q = questions[qkey]
                        text = str(q.get("question", "")).strip()
                        if not text or text in seen:
                            continue
                        seen.add(text)
                        emitted += 1
                        answer = bool(q.get("answer"))
                        inp = (f"{obj.get('theory','')} \n\n"
                               f"Statement: {text}\n\n"
                               f"Does the theory entail this statement?\n"
                               f"Answer TRUE or FALSE.")
                        yield Task(
                            task_id=f"PRW-{obj.get('id')}-{qkey}",
                            domain="proofwriter",
                            difficulty=int(q.get("QDep", 1) or 1),
                            input=inp,
                            expected_output=TRUE if answer else FALSE,
                            reasoning_type="proof",
                            generation_seed=0,
                            verification_method="meta-answer",
                            metadata={
                                "source": "proofwriter",
                                "split": f"{depth}/{split}",
                                "depth": q.get("QDep"),
                                "strategy": q.get("strategy"),
                                "raw_id": obj.get("id"),
                                "theory": obj.get("theory", ""),
                                "statement": text,
                                "ground_truth": TRUE if answer else FALSE,
                            },
                        )


def logiqa_rows(path: Path, seed: int = 42) -> list[Task]:
    """Parse a LogiQA TXT file (blocks separated by blank lines)."""
    rng = random.Random(seed)
    text = path.read_text(encoding="utf-8")
    blocks = [b for b in text.strip().split("\n\n") if b.strip()]
    tasks: list[Task] = []
    for idx, block in enumerate(blocks):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        answer_key = lines[0].strip().upper()          # e.g. "C"
        body = lines[1:]
        passage = body[0] if body else ""
        question = ""
        options: dict[str, str] = {}
        for ln in body[1:]:
            m = ln[0:2]
            if len(ln) > 1 and ln[0].isalpha() and ln[1] == ".":
                options[ln[0].upper()] = ln[2:].strip()
            elif "?" in ln:
                question += (" " if question else "") + ln
        expected = answer_key if answer_key in options else answer_key
        inp = (f"{passage}\n\n{question}\n"
               + "\n".join(f"{k}. {v}" for k, v in options.items())
               + "\n\nAnswer with a single option letter (A-D).")
        tasks.append(Task(
            task_id=f"LQA-{path.stem}-{idx:05d}",
            domain="logiqa",
            difficulty=2,
            input=inp,
            expected_output=expected,
            reasoning_type="reading_comprehension",
            generation_seed=rng.randint(0, 2 ** 31),
            verification_method="gold-option",
            metadata={"source": "logiqa", "file": path.name, "options": options},
        ))
    return tasks


def ruletaker_rows(hf_split: str, max_rows: int, seed: int = 42) -> list[Task]:
    """Convert RuleTaker rows (context, question, label) into Tasks."""
    rows = fetch_hf_streaming(RULE_TAKER_HF, hf_split, max_rows)
    rng = random.Random(seed)
    tasks: list[Task] = []
    for i, row in enumerate(rows):
        context = str(row.get("context", "")).strip()
        question = str(row.get("question", "")).strip()
        label = str(row.get("label", "")).strip()
        if not context or not question:
            continue
        verdict = _label_to_verdict(label)
        cfg = str(row.get("config", "1"))
        digits = "".join(ch for ch in cfg if ch.isdigit())
        try:
            diff = int(digits)
        except ValueError:
            diff = 1
        if not diff:
            diff = 1
        inp = (f"{context}\n\nStatement: {question}\n\n"
               f"Does the context entail this statement?\n"
               f"Answer TRUE, FALSE or UNKNOWN.")
        tasks.append(Task(
            task_id=f"RT-{hf_split}-{i:06d}",
            domain="ruletaker",
            difficulty=diff,
            input=inp,
            expected_output=verdict,
            reasoning_type="theory_question",
            generation_seed=rng.randint(0, 2 ** 31),
            verification_method="gold-label",
            metadata={"source": "ruletaker", "raw_label": label, "config": row.get("config"),
                      "context": context, "statement": question,
                      "ground_truth": verdict},
        ))
    return tasks


def folio_rows(hf_split: str, max_rows: int, seed: int = 42) -> list[Task]:
    """Convert FOLIO rows (premises, conclusion, label) into Tasks."""
    rows = fetch_hf_streaming(FOLIO_HF, hf_split, max_rows)
    rng = random.Random(seed)
    tasks: list[Task] = []
    # FOLIO answers live in a 'label' column: Entailment/Contradiction/Neutral
    for i, row in enumerate(rows):
        premises = _split_list(row.get("premises"))
        conclusion = str(row.get("conclusion", "") or row.get("conclusion-FOL", "")).strip()
        label = str(row.get("label", "")).strip()
        # Formal FOL structure (FOLIO ships real first-order logic here).
        # premises-FOL is a single multi-line string; one FOL formula per line.
        premises_fol = [str(p).strip() for p in
                        str(row.get("premises-FOL") or "").splitlines() if str(p).strip()]
        conclusion_fol = str(row.get("conclusion-FOL", "") or "").strip()
        if not premises or not conclusion:
            continue
        verdict = _label_to_verdict(label)
        inp = ("Premises:\n" + "\n".join(f"- {p}" for p in premises)
               + f"\n\nConclusion: {conclusion}\n\n"
               + f"Does the conclusion follow? Answer {TRUE}, {FALSE} or {UNKNOWN}.")
        tasks.append(Task(
            task_id=f"FOL-{hf_split}-{i:06d}",
            domain="folio",
            difficulty=3,
            input=inp,
            expected_output=verdict,
            reasoning_type="natural_deduction",
            generation_seed=rng.randint(0, 2 ** 31),
            verification_method="gold-label",
            metadata={"source": "folio", "raw_label": label, "story_id": row.get("story_id"),
                      "context": "\n".join(premises), "statement": conclusion,
                      "premises_fol": premises_fol, "conclusion_fol": conclusion_fol,
                      "ground_truth": verdict},
        ))
    return tasks


# ════════════════════════════════════════════════════════════════════
# ORCHESTRATION
# ════════════════════════════════════════════════════════════════════

def build_dataset(
    sources: tuple[str, ...] = ("proofwriter", "logiqa", "folio"),
    ruletaker_rows: int = 2000,
    folio_rows: int = 1400,
    proofwriter_depths=("depth-2", "depth-3"),
    proofwriter_questions: int = 400,
) -> dict[str, list[Task]]:
    """Fetch + convert selected sources into Task lists (raw kept intact)."""
    out: dict[str, list[Task]] = {}

    if "proofwriter" in sources:
        zp = fetch_proofwriter()
        print("[loader] parsing ProofWriter...")
        out["proofwriter"] = list(proofwriter_rows(
            zp, depths=proofwriter_depths, max_questions=proofwriter_questions))

    if "logiqa" in sources:
        paths = fetch_logiqa()
        print("[loader] parsing LogiQA (Train/Eval)...")
        out["logiqa"] = logiqa_rows(paths[0]) + (logiqa_rows(paths[1]) if len(paths) > 1 else [])

    if "ruletaker" in sources:
        print(f"[loader] streaming RuleTaker ({ruletaker_rows})...")
        out["ruletaker"] = ruletaker_rows("train", ruletaker_rows)

    if "folio" in sources:
        print(f"[loader] streaming FOLIO ({folio_rows})...")
        out["folio"] = folio_rows("train", folio_rows)

    total = sum(len(v) for v in out.values())
    print(f"[loader] built {total} tasks across {list(out.keys())}")
    return out
