"""Retrain the intent classifier with expanded dataset.

Combines the existing example queries with the generated dataset,
deduplicates, trains a TF-IDF + LogisticRegression pipeline,
and reports cross-validation + in-sample metrics.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ML = ROOT / "services" / "ml-service"
RAW = ML / "datasets" / "raw"
MODEL_DIR = ML / "models" / "intent" / "intent-baseline-001"
OUT = ML / "datasets" / "raw" / "intents.combined.jsonl"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def combine_and_dedup() -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []
    for source in ["intents.example.jsonl", "intents.generated.jsonl"]:
        p = RAW / source
        if not p.exists():
            print(f"  [skip] {source} not found")
            continue
        for row in load_jsonl(p):
            key = (row["text"].lower().strip(), row["intent"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged


def train(data: list[dict]) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    texts = [r["text"] for r in data]
    labels = [r["intent"] for r in data]
    label_counts = Counter(labels)
    min_count = min(label_counts.values())
    print(f"\n  dataset: {len(texts)} samples, {len(set(labels))} intents")
    print(f"  min class size: {min_count} (smallest: {min(label_counts, key=label_counts.get)})")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2 if min_count >= 5 else 1,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver="lbfgs",
            n_jobs=-1,
        )),
    ])

    n_splits = min(5, min_count)
    print(f"\n  cross-validation ({n_splits} folds) ...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_preds = cross_val_predict(pipe, texts, labels, cv=cv)
    print("\n  cross-validation report:")
    print(classification_report(labels, cv_preds, zero_division=0))

    print("  training full model ...")
    pipe.fit(texts, labels)
    train_preds = pipe.predict(texts)
    train_correct = sum(p == t for p, t in zip(train_preds, labels))
    print(f"  in-sample accuracy: {train_correct}/{len(texts)} = {train_correct / len(texts):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(pipe, MODEL_DIR / "model.joblib")

    metadata = {
        "version": "intent-baseline-002",
        "description": "Expanded 500+ query dataset, TF-IDF word 1-2gram + LogisticRegression",
        "n_samples": len(texts),
        "n_intents": len(set(labels)),
        "intents": sorted(set(labels)),
        "cv_accuracy": float(sum(p == t for p, t in zip(cv_preds, labels)) / len(labels)),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    labels_file = MODEL_DIR / "labels.json"
    labels_file.write_text(json.dumps(sorted(set(labels)), indent=2), encoding="utf-8")

    (OUT).write_text(
        "\n".join(json.dumps(r) for r in data) + "\n",
        encoding="utf-8",
    )
    print(f"\n  model saved -> {MODEL_DIR / 'model.joblib'}")
    print(f"  combined dataset -> {OUT}")


def main() -> int:
    print("=== INTENT RETRAINING ===")
    data = combine_and_dedup()
    if len(data) < 50:
        print(f"FAIL: only {len(data)} samples after merge")
        return 1
    train(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
