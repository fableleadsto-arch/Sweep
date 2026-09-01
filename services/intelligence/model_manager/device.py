"""Device detection — CPU/GPU/VRAM/RAM."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field


@dataclass
class DeviceInfo:
    device: str = "cpu"
    cuda_available: bool = False
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    cpu_count: int = 0
    cpu_name: str = ""
    half_precision: bool = False  # fp16 supported on GPU
    onnx_provider: str = "CPUExecutionProvider"


def detect_device() -> DeviceInfo:
    """Detect the best available compute device."""
    info = DeviceInfo()
    info.cpu_count = os.cpu_count() or 1
    info.cpu_name = platform.processor() or "unknown"

    # RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        info.ram_total_gb = mem.total / 1e9
        info.ram_available_gb = mem.available / 1e9
    except ImportError:
        pass

    # CUDA via PyTorch
    try:
        import torch
        info.cuda_available = torch.cuda.is_available()
        if info.cuda_available:
            info.device = "cuda"
            info.gpu_name = torch.cuda.get_device_name(0)
            info.gpu_memory_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            info.cuda_version = torch.version.cuda or ""
            info.half_precision = True
            info.onnx_provider = "CUDAExecutionProvider"
        else:
            info.device = "cpu"
    except (ImportError, RuntimeError):
        info.device = "cpu"

    # Check for MPS (Apple Silicon)
    if info.device == "cpu":
        try:
            import torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                info.device = "mps"
        except (ImportError, RuntimeError):
            pass

    return info


def get_torch_device(info: DeviceInfo | None = None) -> str:
    """Return a torch device string."""
    if info is None:
        info = detect_device()
    return info.device


def get_device_map(info: DeviceInfo | None = None) -> str:
    """Return a HuggingFace device_map string."""
    if info is None:
        info = detect_device()
    if info.device == "cuda":
        return "auto"
    return "cpu"
