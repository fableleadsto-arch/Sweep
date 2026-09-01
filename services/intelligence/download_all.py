#!/usr/bin/env python3
"""One-shot download all models. Run this and wait."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import snapshot_download

MODELS = [
    # NLP
    ("BERT", "bert-base-uncased", "models/nlp/bert-base-uncased"),
    ("DeBERTa", "microsoft/deberta-v3-base", "models/nlp/deberta-v3-base"),
    ("RoBERTa", "FacebookAI/roberta-base", "models/nlp/roberta-base"),
    ("XLM-RoBERTa", "FacebookAI/xlm-roberta-base", "models/nlp/xlm-roberta-base"),
    ("GLiNER", "urchade/gliner_base-v2.1", "models/nlp/gliner_base-v2.1"),
    # Embeddings
    ("MiniLM", "sentence-transformers/all-MiniLM-L6-v2", "models/embeddings/all-MiniLM-L6-v2"),
    ("MPNet", "sentence-transformers/all-mpnet-base-v2", "models/embeddings/all-mpnet-base-v2"),
    # Documents
    ("LayoutLM", "microsoft/layoutlm-base-uncased", "models/documents/layoutlm"),
    ("LayoutLMv3", "microsoft/layoutlmv3-base", "models/documents/layoutlmv3-base"),
    # Audio
    ("Wav2Vec2", "facebook/wav2vec2-base", "models/audio/wav2vec2"),
]


def download_model(name, repo_id, local_path):
    dest = Path(local_path)
    if dest.exists() and len(list(dest.iterdir())) > 5:
        print(f"  [CACHED]  {name}")
        return True

    dest.mkdir(parents=True, exist_ok=True)
    try:
        t0 = time.time()
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        elapsed = time.time() - t0
        size_mb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
        print(f"  [OK]     {name:<18s} {size_mb:.0f}MB in {elapsed:.0f}s")
        return True
    except Exception as e:
        print(f"  [FAIL]   {name:<18s} {str(e)[:60]}")
        return False


def main():
    print("\n  Sweep Intelligence — Model Downloader\n")
    t0 = time.time()
    results = {}
    for i, (name, repo_id, local_path) in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] {name} ({repo_id})")
        results[name] = download_model(name, repo_id, local_path)

    elapsed = time.time() - t0
    ok = sum(1 for v in results.values() if v)
    fail = len(results) - ok
    print(f"\n  Done: {ok} OK, {fail} failed in {elapsed:.0f}s\n")


if __name__ == "__main__":
    main()
