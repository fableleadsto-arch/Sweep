"""Checkpoint persistence for Relay models.

Layout of a model directory in the registry:
- ``config.json``            — ModelConfig dict (authoritative), always written,
- ``model.safetensors``      — trained weights in safetensors (safe format),
- ``tokenizer.json``         — versioned BPE tokenizer,
- ``optimizer.safetensors``  — optimizer moments (safetensors, tensors only),
- ``training_state.json``    — step/epoch/loss/dataset metadata.

Weights and optimizer tensors use safetensors (no pickle); scalar metadata uses
JSON. ``torch`` is only imported lazily so the module can be probed cheaply.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
from safetensors.torch import load_file, save_file

from ..architecture import ModelConfig, RelayTransformer
from ..tokenizer import RelayBpeTokenizer


def _model_dir(checkpoint_dir: str) -> Path:
    p = Path(checkpoint_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(
    model: RelayTransformer,
    tokenizer: RelayBpeTokenizer,
    optimizer,
    step: int,
    best_loss: float,
    checkpoint_dir: str,
    dataset: str = "",
    eval_metrics: Optional[dict[str, Any]] = None,
) -> str:
    """Persist model weights + optimizer + metadata. Returns checkpoint dir."""
    d = _model_dir(checkpoint_dir)

    with (d / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(model.config.to_dict(), fh, indent=2, sort_keys=True)

    save_file(model.state_dict(), str(d / "model.safetensors"))

    if tokenizer is not None:
        tokenizer.save(str(d / "tokenizer.json"))

    if optimizer is not None:
        opt_state = optimizer.state_dict()
        tensor_dict: dict[str, torch.Tensor] = {}
        for i, (m, v) in enumerate(zip(opt_state.get("m", []), opt_state.get("v", []))):
            tensor_dict[f"m_{i}"] = m
            tensor_dict[f"v_{i}"] = v
        if tensor_dict:
            save_file(tensor_dict, str(d / "optimizer.safetensors"))
        scalar_opt = {k: v for k, v in opt_state.items() if k not in ("m", "v")}

    training = {
        "step": step,
        "best_loss": best_loss,
        "dataset": dataset,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "trained" if best_loss < float("inf") else "experimental",
        "evaluation": eval_metrics or {},
        "optimizer": scalar_opt if optimizer is not None else {},
    }
    with (d / "training_state.json").open("w", encoding="utf-8") as fh:
        json.dump(training, fh, indent=2, sort_keys=True)

    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": _hardware_snapshot(),
    }
    with (d / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    return str(d)


def load_model(checkpoint_dir: str) -> RelayTransformer:
    """Load weights into a fresh model built from the stored config."""
    d = Path(checkpoint_dir)
    with (d / "config.json").open("r", encoding="utf-8") as fh:
        cfg = ModelConfig.from_dict(json.load(fh))
    model = RelayTransformer(cfg)
    state = load_file(str(d / "model.safetensors"))
    model.load_state_dict(state, strict=True)
    return model


def load_tokenizer(checkpoint_dir: str) -> RelayBpeTokenizer:
    return RelayBpeTokenizer.load(str(Path(checkpoint_dir) / "tokenizer.json"))


def _hardware_snapshot() -> dict[str, Any]:
    from ...tools.common import module_available

    hw: dict[str, Any] = {
        "torch": module_available("torch"),
        "cpu_count": os.cpu_count() or 0,
    }
    try:
        if module_available("torch"):
            import torch as _torch

            hw["cuda"] = _torch.cuda.is_available()
            if hw["cuda"]:
                hw["gpu_name"] = _torch.cuda.get_device_name(0)
    except Exception:
        pass
    return hw


__all__ = ["save_checkpoint", "load_model", "load_tokenizer"]
