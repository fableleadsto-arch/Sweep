"""Model manager — framework-independent model ↔ backend resolution.

Lets callers describe a model by what it *is* (format, task, precision,
estimated size) and get back which compute backend should run it — without
caring which framework serialized it. Also checks a model fits the target
device's memory (hardware-aware, so a 24GB model never lands on a 6GB GPU).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..tools.common import module_available
from .capability_detector import ComputeEnvironment, detect

# Serialized-model formats → (backend id, framework runtime).
MODEL_FORMATS: dict[str, tuple[str, str]] = {
    "pytorch": ("pytorch", "torch"),
    "torch": ("pytorch", "torch"),
    "onnx": ("onnx", "onnxruntime"),
    "tensorflow": ("tensorflow", "tensorflow"),
    "keras": ("tensorflow", "keras"),
    "saved_model": ("tensorflow", "tensorflow"),
    "gguf": ("cpu", "llama_cpp"),
    "ggml": ("cpu", "llama_cpp"),
    "safetensors": ("pytorch", "torch"),
}

# Common task kinds → preferred backend ordering (best first).
TASK_PREFERENCES: dict[str, tuple[str, ...]] = {
    "training": ("cuda", "mps", "rocm", "pytorch", "tensorflow", "cpu"),
    "inference": ("cuda", "onnx", "mps", "pytorch", "tensorflow", "cpu"),
    "embedding": ("onnx", "pytorch", "tensorflow", "cpu"),
    "generation": ("cuda", "pytorch", "tensorflow", "onnx", "cpu"),
    "transform": ("cpu", "pytorch", "onnx"),
}


@dataclass
class ModelSpec:
    """A framework-independent description of a model."""

    model_format: str = ""  # "pytorch" | "onnx" | "tensorflow" | "gguf" | ...
    task: str = "inference"  # training | inference | embedding | generation | transform
    precision: str = ""  # fp32 | fp16 | bf16 | int8
    estimated_memory_mb: Optional[float] = None
    name: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        return cls(
            model_format=str(data.get("model_format") or data.get("format") or "").lower(),
            task=str(data.get("task") or "inference").lower(),
            precision=str(data.get("precision") or "").lower(),
            estimated_memory_mb=data.get("estimated_memory_mb"),
            name=str(data.get("name") or ""),
        )

    def runtime(self) -> Optional[str]:
        if self.model_format in MODEL_FORMATS:
            return MODEL_FORMATS[self.model_format][1]
        return None

    def preferred_backend(self) -> str:
        fmt = self.model_format
        if fmt and fmt in MODEL_FORMATS:
            return MODEL_FORMATS[fmt][0]
        prefs = TASK_PREFERENCES.get(self.task, TASK_PREFERENCES["inference"])
        return prefs[0] if prefs else "cpu"


class ModelManager:
    """Resolves models to backends and validates memory fit."""

    def __init__(self, environment: Optional[ComputeEnvironment] = None) -> None:
        self.env = environment or detect()

    def resolve_backend(self, spec: ModelSpec) -> str:
        """The single best backend id for ``spec`` (framework-aware first)."""
        preferred = spec.preferred_backend()
        if preferred == "pytorch" and not self.env.has_framework("torch"):
            if self.env.has_framework("onnxruntime"):
                return "onnx"
            if self.env.has_framework("tensorflow"):
                return "tensorflow"
        if preferred == "tensorflow" and not self.env.has_framework("tensorflow"):
            if self.env.has_framework("torch"):
                return "pytorch"
        if preferred == "onnx" and not self.env.has_framework("onnxruntime"):
            if self.env.has_framework("torch"):
                return "pytorch"
        return preferred if self.env.has_framework(spec.runtime() or preferred) else "cpu"

    def preference_order(self, task: str) -> list[str]:
        """Backend ids best-first for a task kind, filtered to installed runtimes."""
        order: list[str] = []
        for backend in TASK_PREFERENCES.get(task, TASK_PREFERENCES["inference"]):
            if backend == "cpu":
                order.append(backend)
                continue
            runtime = {
                "pytorch": "torch",
                "tensorflow": "tensorflow",
                "onnx": "onnxruntime",
            }.get(backend)
            if runtime and self.env.has_framework(runtime):
                order.append(backend)
            elif backend in ("cuda", "mps", "rocm"):
                order.append(backend)  # device backends gated on HW + framework later
        return order

    def fits_memory(self, spec: ModelSpec, backend_id: str) -> tuple[bool, str]:
        """Whether ``spec`` fits the best device the backend exposes."""
        if spec.estimated_memory_mb is None:
            return True, "no size estimate provided"
        if spec.estimated_memory_mb <= 0:
            return True, "size estimate is zero/unknown"
        device_free: Optional[float] = None
        if backend_id == "cpu":
            device_free = float(self.env.memory.available_mb or self.env.memory.total_mb)
            pool = "system RAM"
        else:
            from .backend_manager import BackendManager

            manager = BackendManager()
            for gpu in self.env.gpus:
                if gpu.api in backend_id:
                    device_free = float(gpu.memory_free_mb or gpu.memory_total_mb)
                    pool = f"GPU VRAM ({gpu.name})"
                    break
            if device_free is None:
                return True, "no GPU VRAM data available"
        # Reserve ~15% headroom so the runtime has working memory too.
        if spec.estimated_memory_mb * 1.15 <= (device_free or 0):
            return True, f"fits in {pool}"
        return (
            False,
            f"model needs ~{spec.estimated_memory_mb:,.0f}MB but {pool} has ~{device_free:,.0f}MB free",
        )


__all__ = ["MODEL_FORMATS", "ModelManager", "ModelSpec", "TASK_PREFERENCES"]
