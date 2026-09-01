"""Structured compute-capability detector.

Probes the host without importing any heavy framework: CPU topology, RAM,
disk, GPU presence (NVIDIA CUDA via ``nvidia-smi``, AMD ROCm via
``rocm-smi``, Apple MPS), Python details and per-framework installed versions
(from import metadata — never an import). The result is a
`ComputeEnvironment` that the backend manager, scheduler and diagnostics all
consume.

Every probe here is defensive: a missing binary, a timeout or a platform we
don't recognise simply degrades to "unknown", never an exception.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata
from typing import Optional

from ..tools.common import module_available

# Optional frameworks we report versions for. Each maps to the requirements
# profile that installs it, so diagnostics can say exactly which profile is
# missing.
FRAMEWORK_PROFILES: dict[str, str] = {
    "torch": "pytorch",
    "torchvision": "pytorch",
    "torchaudio": "pytorch",
    "tensorflow": "tensorflow",
    "keras": "tensorflow",
    "onnxruntime": "onnx",
    "onnx": "onnx",
    "transformers": "pytorch",
    "diffusers": "pytorch",
    "accelerate": "pytorch",
    "timm": "pytorch",
    "vllm": "pytorch",
    "bitsandbytes": "training",
    "triton": "training",
    "jax": "training",
    "jaxlib": "training",
    "litellm": "ai",
    "llama_index": "ai",
    "langchain": "ai",
    "crewai": "ai",
    "autogen": "ai",
    "xgboost": "ai",
    "lightgbm": "ai",
    "cv2": "ai",
    "spacy": "ai",
    "nltk": "ai",
    "pymilvus": "ai",
    "llama_cpp": "ai",
    "qdrant_client": "core",
}

PROFILE_REQUIREMENTS: dict[str, str] = {
    "core": "requirements.companion.txt",
    "pytorch": "requirements.companion-pytorch.txt",
    "tensorflow": "requirements.companion-tensorflow.txt",
    "onnx": "requirements.companion-onnx.txt",
    "cuda": "requirements.companion-cuda.txt",
    "training": "requirements.companion-training.txt",
    "ai": "requirements.companion-ai.txt",
}


@dataclass(frozen=True)
class PythonInfo:
    version: str
    implementation: str
    bitness: str


@dataclass(frozen=True)
class CPUInfo:
    physical_cores: int
    logical_cores: int
    arch: str
    model: str = ""


@dataclass(frozen=True)
class MemoryInfo:
    total_mb: int
    available_mb: Optional[int]


@dataclass(frozen=True)
class DiskInfo:
    total_mb: int
    free_mb: int
    path: str


@dataclass(frozen=True)
class GPUInfo:
    name: str
    vendor: str  # "nvidia" | "amd" | "apple" | "unknown"
    api: str  # "cuda" | "rocm" | "mps"
    index: int
    memory_total_mb: int
    memory_free_mb: Optional[int] = None
    driver: str = ""
    compute_capability: str = ""


@dataclass(frozen=True)
class FrameworkInfo:
    name: str
    installed: bool
    version: str
    profile: str
    profile_requirements: str


@dataclass
class ComputeEnvironment:
    python: PythonInfo
    cpu: CPUInfo
    memory: MemoryInfo
    disk: DiskInfo
    gpus: list[GPUInfo]
    frameworks: dict[str, FrameworkInfo]
    compute_devices: list[str] = field(default_factory=list)
    os_name: str = ""
    os_release: str = ""

    @property
    def has_nvidia(self) -> bool:
        return any(g.api == "cuda" for g in self.gpus)

    @property
    def has_amd(self) -> bool:
        return any(g.api == "rocm" for g in self.gpus)

    @property
    def has_mps(self) -> bool:
        return any(g.api == "mps" for g in self.gpus)

    def has_framework(self, name: str) -> bool:
        info = self.frameworks.get(name)
        return bool(info and info.installed)


# ── probes ──────────────────────────────────────────────────────────────

def _read_os() -> tuple[str, str]:
    try:
        return platform.system() or "unknown", platform.release() or ""
    except Exception:  # noqa: BLE001 - platform never raises but stay defensive
        return "unknown", ""


def _probe_python() -> PythonInfo:
    try:
        import platform as _p

        impl = _p.python_implementation() or "CPython"
    except Exception:  # noqa: BLE001
        impl = "CPython"
    return PythonInfo(
        version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        implementation=impl,
        bitness=f"{sys.maxsize.bit_length() + 1}",
    )


def _probe_cpu() -> CPUInfo:
    arch = platform.machine() or platform.processor() or "unknown"
    try:
        logical = os.cpu_count() or 0
    except Exception:  # noqa: BLE001
        logical = 0
    physical = logical
    # Best-effort physical-core count — degraded silently on failure.
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).NumberOfCores"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                physical = int(result.stdout.strip().splitlines()[-1])
        else:
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
                    physical = sum(1 for line in fh if line.startswith("core id"))
                if physical == 0:
                    physical = logical
            except OSError:
                physical = logical
    except (OSError, ValueError, subprocess.TimeoutExpired):
        physical = logical
    return CPUInfo(
        physical_cores=max(physical, 1),
        logical_cores=max(logical, 1),
        arch=arch,
    )


def _probe_memory() -> MemoryInfo:
    try:
        if sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
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

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return MemoryInfo(
                    total_mb=int(status.ullTotalPhys // (1024 * 1024)),
                    available_mb=int(status.ullAvailPhys // (1024 * 1024)),
                )
        else:
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                values: dict[str, int] = {}
                for line in fh:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0]
                        try:
                            values[key] = int(parts[1].strip().split()[0]) // 1024  # kB -> MB
                        except (ValueError, IndexError):
                            continue
                total = values.get("MemTotal")
                available = values.get("MemAvailable")
                if total is not None:
                    return MemoryInfo(total_mb=total, available_mb=available)
    except Exception:  # noqa: BLE001 - best-effort memory probe
        pass
    return MemoryInfo(total_mb=0, available_mb=None)


def _probe_disk() -> DiskInfo:
    try:
        usage = shutil.disk_usage(".")
        return DiskInfo(
            total_mb=int(usage.total // (1024 * 1024)),
            free_mb=int(usage.free // (1024 * 1024)),
            path=os.getcwd(),
        )
    except OSError:
        return DiskInfo(total_mb=0, free_mb=0, path=os.getcwd())


def _run_smi(command: list[str], timeout: float = 8.0) -> str:
    """Run a CLI GPU probe and return stdout; '' on any failure."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _probe_nvidia_gpus() -> list[GPUInfo]:
    out = _run_smi([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []
    gpus: list[GPUInfo] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            index = int(parts[0])
            total = int(float(parts[2]))
            free = int(float(parts[3]))
        except ValueError:
            continue
        gpus.append(
            GPUInfo(
                name=parts[1],
                vendor="nvidia",
                api="cuda",
                index=index,
                memory_total_mb=total,
                memory_free_mb=free,
                driver=parts[4] if len(parts) > 4 else "",
                compute_capability=parts[5] if len(parts) > 5 else "",
            )
        )
    return gpus


def _probe_amd_gpus() -> list[GPUInfo]:
    out = _run_smi(["rocm-smi", "--showproductname"])
    if not out:
        return []
    names: list[str] = []
    for line in out.splitlines():
        if ":" in line:
            value = line.split(":", 1)[1].strip()
            if value and value.lower() not in ("not found", "n/a", ""):
                names.append(value)
    if not names:
        return []
    gpus: list[GPUInfo] = []
    for idx, name in enumerate(names):
        gpus.append(
            GPUInfo(
                name=name, vendor="amd", api="rocm", index=idx,
                memory_total_mb=0, memory_free_mb=None,
            )
        )
    return gpus


def _probe_apple_mps() -> list[GPUInfo]:
    if sys.platform != "darwin":
        return []
    # Apple Silicon exposes MPS via PyTorch; the detector stays import-free so
    # it reports the capable device family and lets the mps backend confirm.
    machine = platform.machine()
    if machine not in ("arm64", "aarch64"):
        return []
    total = 0
    out = _run_smi(["sysctl", "-n", "hw.memsize"])
    if out:
        try:
            total = int(int(out.strip()) // (1024 * 1024))
        except ValueError:
            total = 0
    return [
        GPUInfo(
            name="Apple Silicon (MPS)",
            vendor="apple",
            api="mps",
            index=0,
            memory_total_mb=total,
            memory_free_mb=None,
        )
    ]


def _probe_frameworks() -> dict[str, FrameworkInfo]:
    frameworks: dict[str, FrameworkInfo] = {}
    for name, profile in sorted(FRAMEWORK_PROFILES.items()):
        version = ""
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = ""
        installed = bool(version) or module_available(name)
        frameworks[name] = FrameworkInfo(
            name=name,
            installed=installed,
            version=version,
            profile=profile,
            profile_requirements=PROFILE_REQUIREMENTS.get(profile, "requirements.companion-ai.txt"),
        )
    return frameworks


def _compute_devices(gpus: list[GPUInfo], memory: MemoryInfo) -> list[str]:
    devices = ["cpu"]
    for gpu in gpus:
        devices.append(f"{gpu.api}:{gpu.index}")
    return devices


# ── public API ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def detect() -> ComputeEnvironment:
    """Scan the host once and return a structured ComputeEnvironment.

    Cached per process — a probe involves subprocess calls (nvidia-smi) that
    shouldn't repeat for every status listing. Diagnostics forces a refresh
    with ``detect.cache_clear()`` when a fresh scan is wanted.
    """
    python_info = _probe_python()
    cpu = _probe_cpu()
    memory = _probe_memory()
    disk = _probe_disk()
    gpus = _probe_nvidia_gpus() or _probe_amd_gpus() or _probe_apple_mps()
    if not gpus:
        gpus = []
    frameworks = _probe_frameworks()
    os_name, os_release = _read_os()
    return ComputeEnvironment(
        python=python_info,
        cpu=cpu,
        memory=memory,
        disk=disk,
        gpus=gpus,
        frameworks=frameworks,
        compute_devices=_compute_devices(gpus, memory),
        os_name=os_name,
        os_release=os_release,
    )


__all__ = [
    "CPUInfo",
    "ComputeEnvironment",
    "DiskInfo",
    "FRAMEWORK_PROFILES",
    "FrameworkInfo",
    "GPUInfo",
    "MemoryInfo",
    "PROFILE_REQUIREMENTS",
    "PythonInfo",
    "detect",
]
