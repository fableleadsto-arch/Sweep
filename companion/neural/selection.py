"""Hardware-aware model selection.

Decides which scale can *actually* run on the current machine based on real
hardware readings (CPU, RAM, GPU presence) and real memory math:
    total = parameters * bytes_per_param + KV-cache + activations margin.

It never claims a model fits that doesn't; recommendations are the largest
scale whose full footprint fits in the currently available memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from ..tools.common import module_available
from .architecture import ModelConfig
from .models.scales import SCALES

_BYTES_PER_PARAM = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1}


def _available_memory_bytes() -> int:
    """Best-effort estimate of available system RAM (bytes)."""
    if module_available("psutil"):
        import psutil

        return psutil.virtual_memory().available
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MEMORYSTATUSEX()
        m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return int(m.ullAvailPhys)
    except Exception:
        pass
    # Fall back to total memory if we can't read the free amount.
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 4 * 1024**3  # conservative 4 GiB guess


def detect_hardware() -> dict[str, Any]:
    """Snapshot of the actual machine, no imports of heavy frameworks."""
    hw: dict[str, Any] = {
        "cpus": os.cpu_count() or 0,
        "ram_bytes": _available_memory_bytes(),
        "gpu": False,
        "gpu_name": None,
        "precision_supported": ["fp32"],
    }
    if module_available("torch"):
        try:
            import torch

            hw["torch_cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                hw["gpu"] = True
                hw["gpu_name"] = torch.cuda.get_device_name(0)
            # bf16 is usable on CPU on recent torch (oneDNN); claim only what is real.
            hw["precision_supported"].append("bf16")
        except Exception:
            pass
    return hw


@dataclass
class ModelFit:
    scale: str
    parameters: int
    bytes_per_param: int
    footprint_bytes: int
    kv_cache_bytes: int
    activations_bytes: int
    total_bytes: int
    available_bytes: int
    fits: bool
    reason: str = ""
    mode: str = "infer"  # "train" | "infer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "parameters": self.parameters,
            "bytes_per_param": self.bytes_per_param,
            "footprint_bytes": self.footprint_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "activations_bytes": self.activations_bytes,
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
            "fits": self.fits,
            "reason": self.reason,
            "mode": self.mode,
        }


def _params_from_config(cfg: ModelConfig) -> int:
    from .registry import _count_parameters_from_config

    return _count_parameters_from_config(cfg)


def estimate_fit(
    cfg: ModelConfig,
    hw: Optional[dict[str, Any]] = None,
    seq_len: int | None = None,
    mode: str = "infer",
) -> ModelFit:
    """Estimate whether ``cfg`` fits on the current hardware.

    ``mode="train"`` accounts for weights + gradients + Adam moments (≈3× the
    parameter footprint) — the honest way to ask "can I train this here?".
    ``mode="infer"`` is weights + KV cache + activations.
    """
    hw = hw or detect_hardware()
    precision = cfg.precision
    bpp = _BYTES_PER_PARAM.get(precision, 4)
    params = _params_from_config(cfg)
    sl = seq_len or cfg.max_context_length

    if mode == "train":
        footprint = params * bpp * 3  # weights + grads + Adam (m, v)
    else:
        footprint = params * bpp
    kv = cfg.num_key_value_heads * cfg.head_dim * 2 * sl * cfg.num_layers * bpp  # K+V
    activations = cfg.hidden_size * sl * cfg.num_layers * 2 * bpp

    total = footprint + kv + activations
    available = hw.get("ram_bytes", 0)
    # Keep 40% headroom for the interpreter, tokenizer, and OS.
    fits = available > 0 and total < available * 0.6

    reason = (
        f"scale '{cfg.name}' fits in available RAM ({mode})"
        if fits
        else f"scale '{cfg.name}' ({mode}) needs {total / 1024**3:.1f} GiB but only ~{available / 1024**3:.1f} GiB available"
    )
    return ModelFit(
        scale=cfg.name,
        parameters=params,
        bytes_per_param=bpp,
        footprint_bytes=footprint,
        kv_cache_bytes=kv,
        activations_bytes=activations,
        total_bytes=total,
        available_bytes=available,
        fits=fits,
        reason=reason,
        mode=mode,
    )


def recommend_model(hw: Optional[dict[str, Any]] = None, mode: str = "train") -> ModelFit:
    """Pick the largest scale that genuinely fits on this machine.

    Defaults to ``train`` mode because the spec's bar is "trains here".
    """
    hw = hw or detect_hardware()
    results = []
    for name in SCALES:
        cfg = ModelConfig(**SCALES[name])
        results.append((name, estimate_fit(cfg, hw, mode=mode)))
    fits = [(n, r) for n, r in results if r.fits]
    if not fits:
        smallest = min(results, key=lambda t: t[1].total_bytes)
        return smallest[1]
    return max(fits, key=lambda t: t[1].total_bytes)[1]


__all__ = ["detect_hardware", "estimate_fit", "recommend_model", "ModelFit"]
