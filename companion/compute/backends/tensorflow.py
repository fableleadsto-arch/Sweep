"""TensorFlow / Keras backend — optional, lazy-loaded on execution."""

from __future__ import annotations

from typing import Any

from ...tools.common import CapabilityUnavailable
from ..base import BackendKind, ComputeBackend, ComputeJobResult, ComputeTask, ComputeTaskKind


class TensorFlowBackend(ComputeBackend):
    id = "tensorflow"
    label = "TensorFlow / Keras"
    kind = BackendKind.TENSORFLOW
    required_libraries: tuple[str, ...] = ("tensorflow", "keras")
    install_hint = "Install: pip install -r requirements.companion-tensorflow.txt"
    preferred_for = ("tensorflow",)

    @property
    def available(self) -> bool:
        from ...tools.common import module_available

        return module_available("tensorflow") and module_available("keras")

    def supports(self, task: ComputeTask) -> bool:
        if task.kind == ComputeTaskKind.CAPABILITY:
            return task.capability == "tensorflow"
        return True

    def devices(self):
        from ..base import DeviceInfo

        tf = self.load()
        devices: list[DeviceInfo] = [DeviceInfo(name="CPU", kind="cpu", index=0)]
        try:
            gpus = tf.config.list_physical_devices("GPU")
            for idx, gpu in enumerate(gpus):
                devices.append(
                    DeviceInfo(name=getattr(gpu, "name", f"GPU:{idx}"), kind="cuda", index=idx)
                )
        except Exception:  # noqa: BLE001 - probe defensively
            pass
        return devices

    def run_smoke(self, task: ComputeTask) -> ComputeJobResult:
        tf = self.load()
        try:
            x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            y = tf.reduce_sum(tf.matmul(x, x)).numpy().item()
            return ComputeJobResult(
                ok=True,
                backend=self.id,
                device="cpu",
                result={"matrix_sum": round(float(y), 4)},
                summary="TensorFlow smoke test passed.",
                libraries_used=["tensorflow", "keras"],
            )
        except Exception as exc:  # noqa: BLE001 - surface cleanly
            return ComputeJobResult(ok=False, backend=self.id, error=f"TensorFlow smoke failed: {exc}")

    def run_capability(self, task: ComputeTask) -> ComputeJobResult:
        if task.capability != "tensorflow":
            raise CapabilityUnavailable(
                f"TensorFlow backend has no handler for capability '{task.capability}'."
            )
        from ...tools import ai

        try:
            outcome = ai.run_tensorflow(task.payload)
        except Exception as exc:  # noqa: BLE001 - tools surface errors as results
            return ComputeJobResult(
                ok=False, backend=self.id, error=str(exc)[:500],
                summary=f"TensorFlow computation failed: {exc}"[:500],
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
