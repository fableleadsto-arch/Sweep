"""
ResourceManager — monitors hardware and manages compute budgets.

The Mesh must understand the machine it runs on. The ResourceManager
tracks CPU, RAM, GPU (where available) and enforces workload states:
FULL, BALANCED, LIGHT, EMERGENCY.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from enum import Enum
from typing import Any


class WorkloadState(Enum):
    FULL = "full"
    BALANCED = "balanced"
    LIGHT = "light"
    EMERGENCY = "emergency"


@dataclass
class SystemProfile:
    """Snapshot of system resources."""
    cpu_count: int = 0
    cpu_usage_pct: float = 0.0
    ram_total_mb: float = 0.0
    ram_available_mb: float = 0.0
    ram_usage_pct: float = 0.0
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_vram_total_mb: float = 0.0
    gpu_vram_available_mb: float = 0.0
    platform: str = ""
    python_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_count": self.cpu_count,
            "cpu_usage_pct": self.cpu_usage_pct,
            "ram_total_mb": self.ram_total_mb,
            "ram_available_mb": self.ram_available_mb,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_vram_total_mb": self.gpu_vram_total_mb,
            "platform": self.platform,
        }


class ResourceManager:
    """
    Monitors system resources and manages compute budgets.

    The Mesh queries the ResourceManager before executing expensive
    nodes to decide whether to use a lighter model.
    """

    def __init__(self) -> None:
        self._state = WorkloadState.BALANCED
        self._profile = self._snapshot()

    def _snapshot(self) -> SystemProfile:
        """Take a snapshot of current system resources."""
        profile = SystemProfile(
            cpu_count=os.cpu_count() or 1,
            platform=platform.system(),
            python_version=platform.python_version(),
        )
        try:
            import psutil
            mem = psutil.virtual_memory()
            profile.ram_total_mb = mem.total / (1024 * 1024)
            profile.ram_available_mb = mem.available / (1024 * 1024)
            profile.ram_usage_pct = mem.percent
            profile.cpu_usage_pct = psutil.cpu_percent(interval=0.1)
        except ImportError:
            # Fallback: estimate from os
            profile.ram_total_mb = 8192  # conservative guess
            profile.ram_available_mb = 4096

        # GPU detection
        try:
            import torch
            if torch.cuda.is_available():
                profile.gpu_available = True
                profile.gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                profile.gpu_vram_total_mb = props.total_mem / (1024 * 1024)
        except (ImportError, RuntimeError):
            pass

        return profile

    def refresh(self) -> SystemProfile:
        self._profile = self._snapshot()
        self._update_state()
        return self._profile

    @property
    def profile(self) -> SystemProfile:
        return self._profile

    @property
    def state(self) -> WorkloadState:
        return self._state

    def _update_state(self) -> None:
        p = self._profile
        if p.ram_usage_pct > 90 or p.cpu_usage_pct > 95:
            self._state = WorkloadState.EMERGENCY
        elif p.ram_usage_pct > 75:
            self._state = WorkloadState.LIGHT
        elif p.ram_usage_pct > 50:
            self._state = WorkloadState.BALANCED
        else:
            self._state = WorkloadState.FULL

    def can_fit(self, memory_mb: float, needs_gpu: bool = False) -> bool:
        """Check whether a model's requirements fit on this machine."""
        if needs_gpu and not self._profile.gpu_available:
            return False
        if needs_gpu:
            return memory_mb <= self._profile.gpu_vram_total_mb
        return memory_mb <= self._profile.ram_available_mb

    def max_model_size_mb(self) -> float:
        """Suggest maximum model size based on current state."""
        if self._state == WorkloadState.EMERGENCY:
            return 128
        if self._state == WorkloadState.LIGHT:
            return 512
        if self._state == WorkloadState.BALANCED:
            return 2048
        return self._profile.ram_available_mb * 0.5

    def __repr__(self) -> str:
        return (
            f"ResourceManager(state={self._state.value}, "
            f"ram={self._profile.ram_available_mb:.0f}MB free)"
        )
