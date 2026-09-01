"""Compute backends — the hardware-aware execution layer for Relay's brain.

The core brain service (companion/) ships dependency-free. Every optional
compute framework (PyTorch, TensorFlow, ONNX Runtime) and every GPU provider
(CUDA, ROCm, MPS) is modelled as a `ComputeBackend` implementation that is
discovered, probed, enabled/disabled and scheduled through the
`backend_manager`. Heavy frameworks are only ever imported when a backend is
actually used for a task — never at startup and never for unrelated requests.

This module defines the interface (`ComputeBackend`) and the task/status
shapes the rest of the layer speaks.
"""

from __future__ import annotations

import platform
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..tools.common import CapabilityUnavailable, module_available


class BackendKind(str, Enum):
    """The family of a compute backend.

    ``framework`` backends execute tasks directly (CPU/PyTorch/TensorFlow/
    ONNX). ``device`` backends are hardware providers (CUDA/ROCm/MPS) that
    select a device for a framework backend to run on.
    """

    CPU = "cpu"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"


@dataclass(frozen=True)
class DeviceInfo:
    """A concrete compute device a backend can run on."""

    name: str
    kind: str  # "cpu" | "cuda" | "rocm" | "mps"
    index: int = 0
    memory_total_mb: float = 0.0
    memory_free_mb: Optional[float] = None
    arch: str = ""

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.index}" if self.kind != "cpu" else "cpu"


class ComputeTaskKind(str, Enum):
    """What a task asks a backend to do.

    ``CAPABILITY`` delegates to the existing capability-engine tools
    (companion/tools/*) — this is how the compute layer extends, rather than
    replaces, the toolbox. ``SMOKE`` runs a tiny framework-native operation
    (e.g. a CUDA matmul) so diagnostics can prove a backend truly works.
    ``TRAINING``/``INFERENCE`` run a real model via a worker or tool.
    """

    CAPABILITY = "capability"
    SMOKE = "smoke"
    TRAINING = "training"
    INFERENCE = "inference"
    TRANSFORM = "transform"


@dataclass
class ComputeTask:
    """A framework-independent task for the backend layer.

    ``capability`` names the capability-engine tool to run for
    ``kind == CAPABILITY`` (e.g. "math", "deep-learning", "onnx"). ``payload``
    is the exact dict the tool runner consumes (task/data/params/...).
    ``device_preference`` is a soft hint ("cuda:0", "mps", "cpu").
    """

    kind: ComputeTaskKind = ComputeTaskKind.CAPABILITY
    capability: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    model_format: str = ""  # "pytorch" | "tensorflow" | "onnx" | "gguf" | ...
    framework_hint: str = ""  # backend id, e.g. "pytorch" or "cuda"
    device_preference: str = ""
    estimated_memory_mb: Optional[float] = None
    precision: str = ""  # "fp32" | "fp16" | "int8" | "bf16"

    @classmethod
    def from_capability(
        cls,
        capability: str,
        payload: dict[str, Any],
        device_preference: str = "",
        framework_hint: str = "",
    ) -> "ComputeTask":
        return cls(
            kind=ComputeTaskKind.CAPABILITY,
            capability=capability,
            payload=payload,
            device_preference=device_preference,
            framework_hint=framework_hint,
        )

    @classmethod
    def smoke(cls, device_preference: str = "") -> "ComputeTask":
        return cls(kind=ComputeTaskKind.SMOKE, device_preference=device_preference)


@dataclass
class ComputeJobResult:
    """The outcome of a backend execution."""

    ok: bool
    backend: str
    device: str = ""
    result: Any = None
    summary: str = ""
    error: str = ""
    libraries_used: list[str] = field(default_factory=list)


@dataclass
class BackendStatus:
    """Live status of one backend — the wire shape for UI/API consumers."""

    id: str
    label: str
    kind: str
    available: bool
    enabled: bool
    required_libraries: list[str] = field(default_factory=list)
    missing_libraries: list[str] = field(default_factory=list)
    version: str = ""
    reason: str = ""
    devices: list[dict[str, Any]] = field(default_factory=list)
    install_hint: str = ""


class ComputeBackend(ABC):
    """The contract every compute backend implements.

    Implementations must be *cheap to probe*: ``is_available()`` and
    ``reason()`` use only ``module_available()`` (a ``find_spec`` probe — never
    an import). Importing the heavy framework happens lazily in ``load()`` /
    ``run()``.
    """

    id: str = ""
    label: str = ""
    kind: BackendKind = BackendKind.CPU
    required_libraries: tuple[str, ...] = ()
    install_hint: str = ""
    # Preferred for which task kinds (used by the scheduler as a tie-breaker).
    preferred_for: tuple[str, ...] = ()
    # True when ``devices()`` needs no framework import (CPU + device backends).
    # Status listing only calls ``devices()`` for these — probing a backend must
    # never pull in a heavy framework.
    devices_import_free: bool = False

    # ── availability (import-free) ────────────────────────────────────

    @property
    @abstractmethod
    def available(self) -> bool:
        """True when the backend can run tasks right now (no heavy import)."""

    @property
    def missing_libraries(self) -> list[str]:
        return [lib for lib in self.required_libraries if not module_available(lib)]

    def reason(self) -> str:
        """Human-readable explanation of why the backend is unavailable."""
        missing = self.missing_libraries
        if missing:
            pretty = ", ".join(missing)
            return f"Missing optional framework(s): {pretty}. {self.install_hint}"
        return ""

    # ── lazy loading (may import the heavy framework) ─────────────────

    def load(self) -> Any:
        """Import and return the framework module (raises CapabilityUnavailable)."""
        for lib in self.required_libraries:
            if not module_available(lib):
                raise CapabilityUnavailable(self.reason() or f"{self.label} is not installed.")
        if not self.required_libraries:
            return None
        from ..tools.common import load as _lazy_load

        return _lazy_load(self.required_libraries[0])

    def version(self) -> str:
        """Framework version without importing it (from installed metadata)."""
        for lib in self.required_libraries:
            from importlib import metadata

            try:
                return metadata.version(lib)
            except metadata.PackageNotFoundError:
                continue
        return ""

    def devices(self) -> list[DeviceInfo]:
        """Devices this backend exposes. May import the framework."""
        return [DeviceInfo(name="CPU", kind="cpu", index=0)]

    # ── execution ─────────────────────────────────────────────────────

    def supports(self, task: ComputeTask) -> bool:
        """Whether this backend is a reasonable target for ``task``."""
        if task.kind == ComputeTaskKind.SMOKE:
            return self.available
        if task.framework_hint and task.framework_hint != self.id:
            return False
        return True

    def run(self, task: ComputeTask) -> ComputeJobResult:
        """Execute ``task`` on this backend.

        Base behaviour: SMOKE tasks run a tiny framework-native op; CAPABILITY
        tasks are dispatched by ``run_capability`` (subclasses override
        ``run_capability`` to route to the matching tool). Raises
        ``CapabilityUnavailable`` when the framework is missing.
        """
        if not self.available:
            raise CapabilityUnavailable(self.reason())
        if task.kind == ComputeTaskKind.SMOKE:
            return self.run_smoke(task)
        if task.kind == ComputeTaskKind.CAPABILITY:
            return self.run_capability(task)
        return ComputeJobResult(
            ok=False, backend=self.id, error=f"{self.label} does not execute {task.kind.value} tasks."
        )

    def run_smoke(self, task: ComputeTask) -> ComputeJobResult:
        """A tiny native operation proving the backend + device works."""
        raise NotImplementedError(f"{self.label} does not implement run_smoke")

    def run_capability(self, task: ComputeTask) -> ComputeJobResult:
        """Route a CAPABILITY task to the matching tool runner.

        Subclasses implement the frameworks they power. The default falls
        through to a generic handler in the backend manager.
        """
        raise CapabilityUnavailable(f"{self.label} has no capability handler for '{task.capability}'.")

    # ── helpers ───────────────────────────────────────────────────────

    def status(self, enabled: bool) -> BackendStatus:
        devices: list[DeviceInfo] = []
        if self.devices_import_free:
            try:
                devices = self.devices()
            except Exception:  # noqa: BLE001 - status listing must never raise
                devices = []
        return BackendStatus(
            id=self.id,
            label=self.label,
            kind=self.kind.value,
            available=self.available,
            enabled=enabled,
            required_libraries=list(self.required_libraries),
            missing_libraries=self.missing_libraries,
            version=self.version(),
            reason=self.reason(),
            devices=[_device_to_dict(d) for d in devices],
            install_hint=self.install_hint,
        )

    @staticmethod
    def python_arch() -> str:
        return platform.machine() or platform.processor() or "unknown"

    @staticmethod
    def python_version() -> str:
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _device_to_dict(device: DeviceInfo) -> dict[str, Any]:
    return {
        "id": device.id,
        "name": device.name,
        "kind": device.kind,
        "index": device.index,
        "memory_total_mb": device.memory_total_mb,
        "memory_free_mb": device.memory_free_mb,
        "arch": device.arch,
    }


__all__ = [
    "BackendKind",
    "BackendStatus",
    "ComputeBackend",
    "ComputeJobResult",
    "ComputeTask",
    "ComputeTaskKind",
    "DeviceInfo",
]
