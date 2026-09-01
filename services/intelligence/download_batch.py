#!/usr/bin/env python3
"""Download all models with retry and progress. Run from project root."""

import os
import sys
import time
import subprocess

sys.path.insert(0, os.getcwd())

from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path


def dl_snapshot(name, repo_id, dest):
    """Download a full snapshot with retry."""
    d = Path(dest)
    if d.exists() and any(d.glob("*.safetensors")) or any(d.glob("*.bin")) or any(d.glob("*.onnx")):
        return "cached"
    d.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            snapshot_download(repo_id=repo_id, local_dir=str(d), local_dir_use_symlinks=False)
            return "ok"
        except Exception as e:
            print(f"  attempt {attempt+1} failed: {str(e)[:80]}")
            time.sleep(2)
    return "failed"


def main():
    tasks = [
        # NLP — these complete tokenizer first, weights come from HF cache
        ("RoBERTa", "FacebookAI/roberta-base", "models/nlp/roberta-base"),
        ("XLM-RoBERTa", "FacebookAI/xlm-roberta-base", "models/nlp/xlm-roberta-base"),
        ("GLiNER", "urchade/gliner_base-v2.1", "models/nlp/gliner_base-v2.1"),
        # Embeddings — check for safetensors or onnx
        ("MiniLM", "sentence-transformers/all-MiniLM-L6-v2", "models/embeddings/all-MiniLM-L6-v2"),
        ("MPNet", "sentence-transformers/all-mpnet-base-v2", "models/embeddings/all-mpnet-base-v2"),
        # Documents
        ("LayoutLM", "microsoft/layoutlm-base-uncased", "models/documents/layoutlm"),
        ("LayoutLMv3", "microsoft/layoutlmv3-base", "models/documents/layoutlmv3-base"),
        # Audio
        ("Wav2Vec2", "facebook/wav2vec2-base", "models/audio/wav2vec2"),
        # CLIP
        ("CLIP-B32", "openai/clip-vit-base-patch32", "models/vision/clip-vit-b32"),
    ]

    for name, repo_id, dest in tasks:
        d = Path(dest)
        has_weights = (
            any(d.glob("*.safetensors")) or
            any(d.glob("*.bin")) or
            any(d.glob("*.onnx")) or
            any(d.glob("*.pt"))
        )
        file_count = len(list(d.rglob("*"))) if d.exists() else 0
        if has_weights and file_count > 5:
            print(f"[CACHED]  {name}")
            continue

        print(f"[START]   {name} ({repo_id})")
        t0 = time.time()
        result = dl_snapshot(name, repo_id, dest)
        elapsed = time.time() - t0
        if d.exists():
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
            print(f"[{result.upper():>7}]  {name:<18s} {size:.0f}MB in {elapsed:.0f}s")
        else:
            print(f"[{result.upper():>7}]  {name}")


if __name__ == "__main__":
    main()
