"""
Hardware Adaptation — §22

Sweep must automatically determine available hardware.
Detect: CPU, RAM, GPU, VRAM, disk space.
Then select an appropriate inference configuration:
    LOW_RESOURCE / BALANCED / HIGH_PERFORMANCE

Never assume CUDA is available.
The engine must continue functioning on CPU where possible.
"""
from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    cpu_count: int = 0
    cpu_name: str = ""
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    gpu_driver: str = ""
    cuda_available: bool = False
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    has_internet: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "os": f"{self.os_name} {self.os_version}",
            "python": self.python_version,
            "cpu": f"{self.cpu_name} ({self.cpu_count} cores)",
            "ram_total_gb": round(self.ram_total_gb, 1),
            "ram_available_gb": round(self.ram_available_gb, 1),
            "gpu": self.gpu_name if self.gpu_available else "None",
            "gpu_vram_gb": round(self.gpu_vram_gb, 1) if self.gpu_available else 0,
            "cuda": self.cuda_available,
            "disk_free_gb": round(self.disk_free_gb, 1),
            "internet": self.has_internet,
        }


@dataclass
class InferenceConfig:
    """Selected inference configuration based on hardware."""
    mode: str  # LOW_RESOURCE, BALANCED, HIGH_PERFORMANCE
    max_concurrent: int
    batch_size: int
    context_length: int
    use_gpu: bool
    quantization: str  # none, float16, int8
    cache_size: int
    web_fetch_enabled: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "max_concurrent": self.max_concurrent,
            "batch_size": self.batch_size,
            "context_length": self.context_length,
            "use_gpu": self.use_gpu,
            "quantization": self.quantization,
            "cache_size": self.cache_size,
            "web_fetch_enabled": self.web_fetch_enabled,
        }


class HardwareDetector:
    """
    §22: Auto-detects hardware capabilities and selects
    an appropriate inference configuration.
    """

    def __init__(self) -> None:
        self._profile: HardwareProfile | None = None

    def detect(self) -> HardwareProfile:
        """Detect hardware capabilities."""
        if self._profile is not None:
            return self._profile

        profile = HardwareProfile()

        # OS
        profile.os_name = platform.system()
        profile.os_version = platform.version()
        profile.python_version = platform.python_version()

        # CPU
        profile.cpu_count = os.cpu_count() or 1
        profile.cpu_name = platform.processor() or "Unknown CPU"

        # RAM
        try:
            import psutil
            mem = psutil.virtual_memory()
            profile.ram_total_gb = mem.total / (1024 ** 3)
            profile.ram_available_gb = mem.available / (1024 ** 3)
        except ImportError:
            # Fallback: estimate from OS
            if profile.os_name == "Windows":
                try:
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    c_ulonglong = ctypes.c_ulonglong
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", c_ulonglong),
                            ("ullAvailPhys", c_ulonglong),
                        ]
                    mem = MEMORYSTATUSEX()
                    mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                    kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
                    profile.ram_total_gb = mem.ullTotalPhys / (1024 ** 3)
                    profile.ram_available_gb = mem.ullAvailPhys / (1024 ** 3)
                except Exception:
                    profile.ram_total_gb = 8.0  # conservative default
                    profile.ram_available_gb = 4.0
            else:
                try:
                    with open("/proc/meminfo", "r") as f:
                        lines = f.readlines()
                    for line in lines:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            profile.ram_total_gb = kb / (1024 ** 2)
                        elif line.startswith("MemAvailable:"):
                            kb = int(line.split()[1])
                            profile.ram_available_gb = kb / (1024 ** 2)
                except Exception:
                    profile.ram_total_gb = 8.0
                    profile.ram_available_gb = 4.0

        # GPU
        profile.gpu_available, profile.gpu_name, profile.gpu_vram_gb = self._detect_gpu()
        profile.cuda_available = self._detect_cuda()

        # Disk
        try:
            disk = shutil.disk_usage("/")
            profile.disk_total_gb = disk.total / (1024 ** 3)
            profile.disk_free_gb = disk.free / (1024 ** 3)
        except Exception:
            profile.disk_total_gb = 100.0
            profile.disk_free_gb = 50.0

        # Internet
        profile.has_internet = self._check_internet()

        self._profile = profile
        return profile

    def _detect_gpu(self) -> tuple[bool, str, float]:
        """Detect GPU via nvidia-smi or platform info."""
        # Try nvidia-smi
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                name = parts[0].strip()
                vram = float(parts[1].strip()) / 1024 if len(parts) > 1 else 0
                return True, name, vram
        except Exception:
            pass

        # Try torch
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
                return True, name, vram
        except ImportError:
            pass

        return False, "", 0.0

    def _detect_cuda(self) -> bool:
        """Detect CUDA availability."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass

        try:
            import subprocess
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def _check_internet(self) -> bool:
        """Check internet connectivity."""
        try:
            import urllib.request
            urllib.request.urlopen("https://www.google.com", timeout=3)
            return True
        except Exception:
            return False

    def select_config(self, profile: HardwareProfile | None = None) -> InferenceConfig:
        """
        Select the appropriate inference configuration based on hardware.

        §22: LOW_RESOURCE / BALANCED / HIGH_PERFORMANCE
        """
        profile = profile or self.detect()

        # HIGH_PERFORMANCE: GPU with 4+ GB VRAM, 16+ GB RAM
        if profile.gpu_available and profile.gpu_vram_gb >= 4 and profile.ram_total_gb >= 16:
            return InferenceConfig(
                mode="HIGH_PERFORMANCE",
                max_concurrent=8,
                batch_size=32,
                context_length=8192,
                use_gpu=True,
                quantization="none",
                cache_size=10000,
                web_fetch_enabled=True,
                description="Full GPU acceleration with large batch processing",
            )

        # BALANCED: 8+ GB RAM, decent CPU
        if profile.ram_total_gb >= 8 and profile.cpu_count >= 4:
            return InferenceConfig(
                mode="BALANCED",
                max_concurrent=4,
                batch_size=16,
                context_length=4096,
                use_gpu=profile.gpu_available,
                quantization="float16" if profile.gpu_available else "none",
                cache_size=5000,
                web_fetch_enabled=True,
                description="Balanced mode for standard hardware",
            )

        # LOW_RESOURCE: limited RAM/CPU
        return InferenceConfig(
            mode="LOW_RESOURCE",
            max_concurrent=1,
            batch_size=4,
            context_length=2048,
            use_gpu=False,
            quantization="int8",
            cache_size=1000,
            web_fetch_enabled=profile.has_internet,
            description="Conservative mode for limited hardware",
        )

    def get_system_report(self) -> dict[str, Any]:
        """Generate a full system report."""
        profile = self.detect()
        config = self.select_config(profile)
        return {
            "hardware": profile.to_dict(),
            "inference_config": config.to_dict(),
        }
