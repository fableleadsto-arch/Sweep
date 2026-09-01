"""Download models CLI — fetches pretrained weights to local models/ directory."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.intelligence.model_downloader import ModelDownloader
from services.intelligence.model_manager.device import detect_device
from services.intelligence.model_manager.registry import ModelRegistry

# Category groups
CORE_CATEGORIES = ["nlp", "embeddings"]
VISION_CATEGORIES = ["vision", "detection", "face"]
AUDIO_CATEGORIES = ["audio", "speech"]
DOCUMENT_CATEGORIES = ["documents"]
ALL_CATEGORIES = CORE_CATEGORIES + VISION_CATEGORIES + AUDIO_CATEGORIES + DOCUMENT_CATEGORIES


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained model weights for Sweep Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python services/intelligence/download_models.py --all
  python services/intelligence/download_models.py --core
  python services/intelligence/download_models.py --vision
  python services/intelligence/download_models.py --audio
  python services/intelligence/download_models.py --documents
  python services/intelligence/download_models.py --model deberta-v3-base
  python services/intelligence/download_models.py --list
        """,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--core", action="store_true", help="Download core NLP + embedding models")
    group.add_argument("--vision", action="store_true", help="Download vision + detection models")
    group.add_argument("--audio", action="store_true", help="Download audio + speech models")
    group.add_argument("--documents", action="store_true", help="Download document models")
    group.add_argument("--all", action="store_true", help="Download all models")
    group.add_argument("--model", help="Download a specific model by name")
    group.add_argument("--list", action="store_true", help="List all models and their status")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--include-optional", action="store_true", help="Include optional models")

    args = parser.parse_args()
    device = detect_device()
    registry = ModelRegistry()
    downloader = ModelDownloader(registry)

    # List mode
    if args.list:
        _print_model_list(registry)
        return

    # Determine which categories to download
    categories = []
    if args.all:
        categories = ALL_CATEGORIES
    elif args.core:
        categories = CORE_CATEGORIES
    elif args.vision:
        categories = VISION_CATEGORIES
    elif args.audio:
        categories = AUDIO_CATEGORIES
    elif args.documents:
        categories = DOCUMENT_CATEGORIES
    elif args.model:
        print(f"\n  Downloading model: {args.model}\n")
        result = downloader.download(args.model, force=args.force)
        _print_result(result)
        return
    else:
        parser.print_help()
        return

    print(f"\n{'='*60}")
    print(f"  Sweep Intelligence Model Downloader")
    print(f"  Device: {device.device} | RAM: {device.ram_total_gb:.1f}GB | Disk: {device.ram_total_gb:.0f}GB+")
    print(f"  Categories: {', '.join(categories)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    for cat in categories:
        entries = registry.by_category(cat)
        if not entries:
            continue
        print(f"\n  ── {cat.upper()} {'─'*(50-len(cat))}")
        for entry in entries:
            if entry.optional and not args.include_optional:
                print(f"  [skip]  {entry.name} (optional, use --include-optional)")
                continue
            if entry.local_path == "":
                print(f"  [skip]  {entry.name} (system/package — no download needed)")
                continue
            result = downloader.download(entry.name, force=args.force)
            _print_result(result)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"{'='*60}\n")


def _print_result(result: dict):
    status = result["status"]
    name = result["name"]
    if status == "downloaded":
        path = result.get("path", "")
        t = result.get("time_s", 0)
        print(f"  [OK]     {name:<35s} -> {path} ({t:.1f}s)")
    elif status == "exists":
        print(f"  [CACHED] {name:<35s} already downloaded")
    elif status == "skip":
        reason = result.get("reason", "")
        print(f"  [SKIP]   {name:<35s} {reason}")
    elif status == "error":
        error = result.get("error", "unknown")
        print(f"  [FAIL]   {name:<35s} {error[:60]}")
    else:
        print(f"  [{status:>7}] {name}")


def _print_model_list(registry: ModelRegistry):
    print(f"\n{'='*80}")
    print(f"  Sweep Intelligence — Model Registry")
    print(f"{'='*80}\n")

    device = detect_device()
    print(f"  Device: {device.device} | RAM: {device.ram_total_gb:.1f}GB")
    if device.cuda_available:
        print(f"  GPU: {device.gpu_name} ({device.gpu_memory_gb:.1f}GB)")
    print()

    for cat in ["nlp", "embeddings", "vision", "detection", "face", "audio", "speech", "documents"]:
        entries = registry.by_category(cat)
        if not entries:
            continue
        print(f"  {cat.upper()}")
        for e in entries:
            status_icon = {
                "not_downloaded": "○",
                "downloaded": "●",
                "loaded": "◉",
                "error": "✗",
                "unavailable": "—",
            }.get(e.status.value, "?")
            opt = " (optional)" if e.optional else ""
            gpu = " [GPU]" if e.requires_gpu else ""
            auth = " [AUTH]" if e.requires_auth else ""
            print(f"    {status_icon} {e.name:<35s} {e.task:<30s}{opt}{gpu}{auth}")
        print()


if __name__ == "__main__":
    main()
