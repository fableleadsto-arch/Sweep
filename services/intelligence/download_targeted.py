#!/usr/bin/env python3
"""Targeted model downloader — fetches ONLY config/tokenizer/safetensors.
Avoids downloading TF/Flax/ONNX duplicates."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from huggingface_hub import snapshot_download

# Common patterns for every HF model
BASE_PATTERNS = [
    "*.json",           # config, tokenizer_config, etc.
    "vocab*",           # vocab files
    "merges*",
    "spm.model",
    "sentencepiece.bpe.model",
    "tokenizer.model",
    "*.md",
]
WEIGHT_PATTERNS = ["model.safetensors"]  # prefer safetensors
FALLBACK_WEIGHTS = ["pytorch_model.bin"]

# sentence-transformers need extra dirs
ST_PATTERNS = BASE_PATTERNS + WEIGHT_PATTERNS + FALLBACK_WEIGHTS + [
    "1_Pooling/*",
    "2_Dense/*",
    "modules.json",
    "onnx/model.onnx",
]

TASKS = [
    # (name, repo_id, dest, patterns)
    ("XLM-RoBERTa", "FacebookAI/xlm-roberta-base", "models/nlp/xlm-roberta-base",
     BASE_PATTERNS + WEIGHT_PATTERNS),
    ("MPNet", "sentence-transformers/all-mpnet-base-v2", "models/embeddings/all-mpnet-base-v2",
     ST_PATTERNS),
    ("GLiNER", "urchade/gliner_base-v2.1", "models/nlp/gliner_base-v2.1",
     BASE_PATTERNS + WEIGHT_PATTERNS + FALLBACK_WEIGHTS),
    ("LayoutLM", "microsoft/layoutlm-base-uncased", "models/documents/layoutlm",
     BASE_PATTERNS + WEIGHT_PATTERNS),
    ("LayoutLMv3", "microsoft/layoutlmv3-base", "models/documents/layoutlmv3-base",
     BASE_PATTERNS + WEIGHT_PATTERNS),
    ("Wav2Vec2", "facebook/wav2vec2-base", "models/audio/wav2vec2",
     BASE_PATTERNS + WEIGHT_PATTERNS),
    ("CLIP-B32", "openai/clip-vit-base-patch32", "models/vision/clip-vit-b32",
     BASE_PATTERNS + WEIGHT_PATTERNS + ["preprocessor_config.json"]),
]


def complete(d: Path) -> bool:
    if not d.exists():
        return False
    has_weights = any(d.glob("*.safetensors")) or any(
        f for f in d.iterdir() if f.is_file() and f.suffix in (".bin", ".pt"))
    return has_weights


def main():
    print("\n=== Sweep Intelligence — Targeted Model Downloader ===\n")
    results = {}
    for i, (name, repo_id, dest, patterns) in enumerate(TASKS, 1):
        d = Path(dest)
        if complete(d):
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
            print(f"[{i}/{len(TASKS)}] [CACHED] {name:<14s} {size:.0f}MB")
            results[name] = "cached"
            continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(d),
                allow_patterns=patterns,
                ignore_patterns=["*.h5", "*.ot", "*.msgpack", "flax*", "tf_*", "rust_model*"],
            )
            elapsed = time.time() - t0
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
            ok = complete(d)
            status = "OK" if ok else "PARTIAL"
            print(f"[{i}/{len(TASKS)}] [{status:>7}] {name:<14s} {size:.0f}MB in {elapsed:.0f}s")
            results[name] = "ok" if ok else "partial"
        except Exception as e:
            print(f"[{i}/{len(TASKS)}] [FAIL] {name:<14s} {str(e)[:70]}")
            results[name] = "failed"

    print(f"\nSummary: ", end="")
    for name, r in results.items():
        print(f"{name}={r}", end=" ")
    print()


if __name__ == "__main__":
    main()
