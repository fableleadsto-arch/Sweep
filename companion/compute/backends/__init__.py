"""Compute backend implementations.

Each framework/device provider is a `ComputeBackend`. Framework backends
(cpu/pytorch/tensorflow/onnx) execute tasks; device backends (cuda/rocm/mps)
are hardware providers that select a device for a framework to run on.

Probing (`available`/`reason`) never imports the framework — only ``run`` does,
and only when the scheduler actually picks the backend.
"""

from __future__ import annotations

from .cpu import CPUBackend
from .devices import CUDABackend, MPSBackend, ROCmBackend
from .onnx import ONNXBackend
from .pytorch import PyTorchBackend
from .tensorflow import TensorFlowBackend

# Canonical discovery order: CPU first (always there), then framework backends
# (lightest to heaviest), then device providers.
ALL_BACKENDS: list[type] = [
    CPUBackend,
    ONNXBackend,
    TensorFlowBackend,
    PyTorchBackend,
    CUDABackend,
    ROCmBackend,
    MPSBackend,
]

BACKEND_BY_ID: dict[str, type] = {cls.id: cls for cls in ALL_BACKENDS}


def instantiate_all() -> list:
    return [cls() for cls in ALL_BACKENDS]


def backend_for(id: str):
    cls = BACKEND_BY_ID.get(id)
    return cls() if cls is not None else None


__all__ = [
    "ALL_BACKENDS",
    "BACKEND_BY_ID",
    "CPUBackend",
    "CUDABackend",
    "MPSBackend",
    "ONNXBackend",
    "PyTorchBackend",
    "ROCmBackend",
    "TensorFlowBackend",
    "backend_for",
    "instantiate_all",
]
