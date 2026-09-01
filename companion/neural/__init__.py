"""Relay Neural — native neural-network intelligence layer.

The companion brain service's own neural stack. Everything here is optional and
lazy-loaded: importing this package never imports torch, safetensors, or the HF
tokenizer library. Torch-dependent pieces (architecture, training, inference,
evaluation) are resolved lazily on first access (PEP 562), so the boot path and
the capability/registry probes stay dependency-free.

The stack follows the honest-scaling rule: a small, real, trained model beats
an untrained giant. Parameter counts are always computed from the actual model
(or its config), never claimed by hand. See ``companion/neural/README.md``.
"""

from __future__ import annotations

from .architecture.config import ModelConfig, merge_with_scale
from .models.scales import SCALES, scale_config
from .registry import DEFAULT_REGISTRY_DIR, ModelRecord, ModelRegistry, availability
from .router import NativeRouter, RouteDecision
from .selection import ModelFit, detect_hardware, estimate_fit, recommend_model

__all__ = [
    "ModelConfig",
    "merge_with_scale",
    "SCALES",
    "scale_config",
    "ModelRegistry",
    "ModelRecord",
    "DEFAULT_REGISTRY_DIR",
    "availability",
    "NativeRouter",
    "RouteDecision",
    "ModelFit",
    "detect_hardware",
    "estimate_fit",
    "recommend_model",
    "RelayTransformer",  # lazily resolved below
    "train",  # lazily resolved below
]


def __getattr__(name: str):
    if name == "RelayTransformer":
        from .architecture.transformer import RelayTransformer

        return RelayTransformer
    if name == "train":
        from .training.trainer import train

        return train
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
