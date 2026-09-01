"""Model downloader — downloads pretrained model weights into models/."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .model_manager.registry import ModelEntry, ModelRegistry, ModelStatus


class ModelDownloader:
    """Downloads model weights from HuggingFace and other sources to local paths."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def download(self, name: str, force: bool = False) -> dict:
        """Download a single model. Returns status dict."""
        entry = self.registry.get(name)
        if entry is None:
            return {"name": name, "status": "error", "error": f"Unknown model: {name}"}

        if entry.local_path == "":
            return {"name": name, "status": "skip", "reason": "No download needed (system/package)"}

        local = Path(entry.local_path)

        if local.exists() and not force:
            # Check if it has actual weights
            if self._is_complete(entry, local):
                entry.status = ModelStatus.DOWNLOADED
                self.registry.save_status()
                return {"name": name, "status": "exists", "path": str(local)}
            else:
                # Partial download — clean and re-download
                shutil.rmtree(local, ignore_errors=True)

        local.mkdir(parents=True, exist_ok=True)
        entry.status = ModelStatus.DOWNLOADING
        self.registry.save_status()

        try:
            t0 = time.time()
            downloader = self._get_downloader(entry)
            downloader(entry, local)
            elapsed = time.time() - t0

            if self._is_complete(entry, local):
                entry.status = ModelStatus.DOWNLOADED
                entry.verified = True
                self.registry.save_status()
                return {
                    "name": name, "status": "downloaded",
                    "path": str(local), "time_s": round(elapsed, 1),
                }
            else:
                entry.status = ModelStatus.ERROR
                entry.error = "Downloaded but verification failed"
                self.registry.save_status()
                return {"name": name, "status": "error", "error": "Verification failed"}

        except Exception as e:
            entry.status = ModelStatus.ERROR
            entry.error = str(e)
            self.registry.save_status()
            return {"name": name, "status": "error", "error": str(e)}

    def download_all(self, force: bool = False) -> list[dict]:
        """Download all non-optional models."""
        results = []
        for entry in self.registry.all():
            if entry.optional and not force:
                results.append({"name": entry.name, "status": "skip", "reason": "optional"})
                continue
            result = self.download(entry.name, force=force)
            results.append(result)
            print(f"  [{result['status']:>10}] {entry.name}")
        return results

    def download_category(self, category: str, force: bool = False) -> list[dict]:
        """Download all models in a category."""
        results = []
        for entry in self.registry.by_category(category):
            result = self.download(entry.name, force=force)
            results.append(result)
        return results

    def _is_complete(self, entry: ModelEntry, path: Path) -> bool:
        """Check if a model download is complete."""
        if not path.exists():
            return False

        if entry.provider == "ultralytics":
            return (path / "yolov8n.pt").exists()

        if entry.provider == "openai-whisper":
            return any(path.glob(f"{entry.checkpoint}.pt"))

        if entry.provider == "insightface":
            return any((path / "models" / "buffalo_l").glob("*.onnx"))

        if entry.provider == "opencv" or entry.provider == "tesseract":
            return True

        # For HuggingFace models, check for config.json or model files
        files = list(path.iterdir())
        if not files:
            return False

        # Must have at least a config or weights file
        has_config = (path / "config.json").exists() or (path / "tokenizer_config.json").exists()
        has_weights = any(
            f.suffix in (".bin", ".safetensors", ".pt", ".pth", ".onnx")
            for f in files
        )
        has_pytorch = (path / "pytorch_model.bin").exists() or any(
            f.suffix == ".safetensors" for f in files
        )
        return has_config or has_weights or has_pytorch

    def _get_downloader(self, entry: ModelEntry):
        """Get the appropriate download function for a model."""
        dispatch = {
            "huggingface": self._download_hf,
            "sentence-transformers": self._download_hf,
            "open_clip": self._download_clip,
            "ultralytics": self._download_ultralytics,
            "insightface": self._download_insightface,
            "openai-whisper": self._download_whisper,
            "pyannote": self._download_pyannote,
            "opencv": self._noop,
            "tesseract": self._noop,
        }
        return dispatch.get(entry.provider, self._download_hf)

    def _download_hf(self, entry: ModelEntry, dest: Path):
        """Download from HuggingFace Hub."""
        from huggingface_hub import snapshot_download
        print(f"    Downloading {entry.checkpoint} -> {dest}")
        snapshot_download(
            repo_id=entry.checkpoint,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )

    def _download_clip(self, entry: ModelEntry, dest: Path):
        """Download CLIP model via open_clip."""
        import open_clip
        parts = entry.checkpoint.split("::")
        model_name = parts[0]
        pretrained = parts[1] if len(parts) > 1 else "openai"
        # open_clip downloads to its own cache; we symlink/copy
        print(f"    Downloading CLIP {model_name} (pretrained={pretrained})")
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        # Copy from HF cache to our local path
        import open_clip._pretrained as _pt
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        # Find the model in cache
        model_id = f"models--{entry.checkpoint.split('::')[0]}" if "::" in entry.checkpoint else ""
        # Just download to local dir using snapshot_download
        from huggingface_hub import snapshot_download
        # open_clip ViT models are hosted on HF
        hf_map = {
            "ViT-B-32": "openai/clip-vit-base-patch32",
            "ViT-L-14": "openai/clip-vit-large-patch14",
        }
        hf_repo = hf_map.get(model_name)
        if hf_repo:
            snapshot_download(repo_id=hf_repo, local_dir=str(dest), local_dir_use_symlinks=False)

    def _download_ultralytics(self, entry: ModelEntry, dest: Path):
        """Download YOLO model."""
        from ultralytics import YOLO
        model = YOLO(entry.checkpoint)  # downloads automatically
        # Move to our local path
        src = Path(model.ckpt_path) if hasattr(model, "ckpt_path") else None
        if src and src.exists():
            shutil.copy2(str(src), str(dest / "yolov8n.pt"))
        else:
            # YOLO caches in ultralytics home; just note it
            print(f"    YOLO model cached in ultralytics dir")

    def _download_insightface(self, entry: ModelEntry, dest: Path):
        """Download InsightFace model pack."""
        from huggingface_hub import snapshot_download
        print(f"    Downloading InsightFace buffalo_l")
        # InsightFace models are on HF
        try:
            snapshot_download(
                repo_id="highkite/buffalo_l",
                local_dir=str(dest),
                local_dir_use_symlinks=False,
            )
        except Exception:
            # Fallback: insightface downloads to its own cache
            import insightface
            print(f"    InsightFace will download on first use to its cache")

    def _download_whisper(self, entry: ModelEntry, dest: Path):
        """Download Whisper model."""
        import whisper
        print(f"    Downloading Whisper {entry.checkpoint}")
        whisper._download(
            whisper._MODELS[entry.checkpoint],
            str(dest),
            check_sha256=False,
        )
        # Whisper downloads as a single .pt file; move into dir structure
        pt_file = dest / f"{entry.checkpoint}.pt"
        if not pt_file.exists():
            # Model may be directly in dest
            for f in dest.glob("*.pt"):
                if f.name != f"{entry.checkpoint}.pt":
                    shutil.move(str(f), str(pt_file))

    def _download_pyannote(self, entry: ModelEntry, dest: Path):
        """Download pyannote model (requires HF auth)."""
        from huggingface_hub import snapshot_download
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise RuntimeError(
                "pyannote requires HF authentication. Set HF_TOKEN env var."
            )
        snapshot_download(
            repo_id=entry.checkpoint,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            token=token,
        )

    def _noop(self, entry: ModelEntry, dest: Path):
        """No download needed."""
        pass
