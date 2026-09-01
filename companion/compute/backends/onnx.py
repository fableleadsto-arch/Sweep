"""ONNX Runtime backend — cross-platform model inference.

Optional: available only when ``onnxruntime`` is installed. Provides provider
introspection (CPU / CUDA / TensorRT / ...) and model inference.
"""

from __future__ import annotations

from typing import Any

from ...tools.common import CapabilityUnavailable
from ..base import BackendKind, ComputeBackend, ComputeJobResult, ComputeTask, ComputeTaskKind


class ONNXBackend(ComputeBackend):
    id = "onnx"
    label = "ONNX Runtime"
    kind = BackendKind.ONNX
    required_libraries: tuple[str, ...] = ("onnxruntime",)
    install_hint = "Install: pip install -r requirements.companion-onnx.txt"
    preferred_for = ("onnx",)

    @property
    def available(self) -> bool:
        from ...tools.common import module_available

        return module_available("onnxruntime")

    def supports(self, task: ComputeTask) -> bool:
        # ONNX only runs the "onnx" capability; anything else must go elsewhere
        # (CPU for light tasks, PyTorch/TensorFlow for framework work).
        if task.kind == ComputeTaskKind.CAPABILITY:
            return task.capability == "onnx"
        return True

    def devices(self):
        from ..base import DeviceInfo

        ort = self.load()
        devices: list[DeviceInfo] = []
        try:
            providers = ort.get_available_providers()
        except Exception:  # noqa: BLE001
            providers = []
        for idx, provider in enumerate(providers):
            kind = "cuda" if "CUDA" in provider or "TensorRT" in provider else "cpu"
            devices.append(
                DeviceInfo(name=provider, kind=kind, index=idx, arch=provider)
            )
        return devices

    def run_smoke(self, task: ComputeTask) -> ComputeJobResult:
        ort = self.load()
        try:
            providers = ort.get_available_providers()
            # CUDA providers prove the runtime can see the GPU even without a model.
            has_gpu = any("CUDA" in p or "TensorRT" in p for p in providers)
            return ComputeJobResult(
                ok=True,
                backend=self.id,
                device="cuda" if has_gpu else "cpu",
                result={"providers": providers},
                summary=(
                    "ONNX Runtime smoke test passed"
                    + (f" ({len(providers)} providers, GPU available)." if has_gpu else " (CPU).")
                ),
                libraries_used=["onnxruntime"],
            )
        except Exception as exc:  # noqa: BLE001
            return ComputeJobResult(ok=False, backend=self.id, error=f"ONNX smoke failed: {exc}")

    def run_capability(self, task: ComputeTask) -> ComputeJobResult:
        if task.capability != "onnx":
            raise CapabilityUnavailable(
                f"ONNX backend has no handler for capability '{task.capability}'."
            )
        from ...tools import inference

        try:
            outcome = inference.run_onnx(task.payload)
        except Exception as exc:  # noqa: BLE001
            return ComputeJobResult(
                ok=False, backend=self.id, error=str(exc)[:500],
                summary=f"ONNX computation failed: {exc}"[:500],
            )
        result = outcome.get("result")
        if result is None or result is False:
            return ComputeJobResult(
                ok=False, backend=self.id, error=outcome.get("summary", "No result"),
                summary=str(outcome.get("summary", "No result")),
            )
        return ComputeJobResult(
            ok=True, backend=self.id, result=result,
            summary=outcome.get("summary", "Done."),
            libraries_used=outcome.get("libraries_used", []),
        )
