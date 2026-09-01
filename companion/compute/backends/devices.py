"""Device backends — CUDA, ROCm and Apple MPS hardware providers.

A device backend does not execute tasks itself: it selects a device for a
framework backend (PyTorch, TensorFlow, ONNX Runtime) to run on, and gates the
frameworks' availability on the matching hardware being present. This keeps
the scheduler hardware-aware while a single source of truth (the framework
backends) owns execution.

All three are optional and degrade gracefully:
  * CUDA available  → an NVIDIA GPU is present AND at least one framework
                      backend can use it (torch.cuda / TF GPU / onnx CUDA EP).
  * ROCm available  → an AMD GPU is present AND torch reports HIP.
  * MPS available   → macOS Apple Silicon AND torch reports MPS.
"""

from __future__ import annotations

from ...tools.common import module_available
from ..base import BackendKind, ComputeBackend, ComputeJobResult, ComputeTask
from ..capability_detector import detect


class _DeviceBackend(ComputeBackend):
    """Base for hardware-provider backends (never imported at probe time)."""

    gpu_api = ""
    label = "GPU device"
    required_libraries: tuple[str, ...] = ()
    devices_import_free = True

    @property
    def available(self) -> bool:
        env = detect()
        if self.gpu_api == "cuda" and not env.has_nvidia:
            return False
        if self.gpu_api == "rocm" and not env.has_amd:
            return False
        if self.gpu_api == "mps" and not env.has_mps:
            return False
        return self._framework_ready

    def reason(self) -> str:
        env = detect()
        if self.gpu_api == "cuda" and not env.has_nvidia:
            return "No NVIDIA GPU detected (nvidia-smi). Install a CUDA-capable GPU or run on CPU."
        if self.gpu_api == "rocm" and not env.has_amd:
            return "No AMD GPU detected (rocm-smi). Install a ROCm-capable GPU or run on CPU."
        if self.gpu_api == "mps" and not env.has_mps:
            return "MPS requires macOS on Apple Silicon."
        if not self._framework_ready:
            return (
                f"{self.label} hardware was detected but no installed framework can use it. "
                f"{self.install_hint}"
            )
        return ""

    def devices(self):
        from ..base import DeviceInfo

        env = detect()
        out = []
        for gpu in env.gpus:
            if gpu.api == self.gpu_api:
                out.append(
                    DeviceInfo(
                        name=gpu.name, kind=self.gpu_api, index=gpu.index,
                        memory_total_mb=gpu.memory_total_mb,
                        memory_free_mb=gpu.memory_free_mb,
                        arch=gpu.compute_capability or self.gpu_api,
                    )
                )
        return out

    def supports(self, task: ComputeTask) -> bool:
        # Device backends run GPU-friendly work (via a framework backend).
        if task.kind.value in ("training", "inference", "generation", "embedding"):
            return True
        if task.kind.value == "capability":
            return False  # capability jobs go to the backend that owns them
        return super().supports(task)

    def run_smoke(self, task: ComputeTask) -> ComputeJobResult:
        # Delegate the smoke to the PyTorch backend pinned to this device.
        from .pytorch import PyTorchBackend

        backend = PyTorchBackend()
        if not backend.available:
            return ComputeJobResult(
                ok=False, backend=self.id,
                error="PyTorch is required to smoke-test the GPU. " + self.install_hint,
            )
        hint = f"{self.gpu_api}:{task.device_preference.split(':')[-1]}" if ":" in task.device_preference else self.gpu_api
        return backend.run_smoke(ComputeTask.smoke(device_preference=hint))


class CUDABackend(_DeviceBackend):
    id = "cuda"
    label = "NVIDIA CUDA"
    kind = BackendKind.CUDA
    gpu_api = "cuda"
    install_hint = "Install a framework that supports CUDA (e.g. pip install -r requirements.companion-pytorch.txt)"

    @property
    def _framework_ready(self) -> bool:
        # Import-free approximation: an NVIDIA GPU is present and a framework
        # that can address it is installed. The exact check (torch.cuda /
        # ONNX CUDA EP / TF GPU) happens in the smoke test, which may import.
        if module_available("torch"):
            return True
        if module_available("onnxruntime"):
            return True
        if module_available("tensorflow"):
            return True
        return False


class ROCmBackend(_DeviceBackend):
    id = "rocm"
    label = "AMD ROCm"
    kind = BackendKind.ROCM
    gpu_api = "rocm"
    install_hint = "Install a framework with ROCm support (e.g. a HIP build of PyTorch)."

    @property
    def _framework_ready(self) -> bool:
        return module_available("torch")


class MPSBackend(_DeviceBackend):
    id = "mps"
    label = "Apple MPS"
    kind = BackendKind.MPS
    gpu_api = "mps"
    install_hint = "Install PyTorch (requirements.companion-pytorch.txt) to use MPS."

    @property
    def _framework_ready(self) -> bool:
        return module_available("torch")


__all__ = ["CUDABackend", "MPSBackend", "ROCmBackend"]
