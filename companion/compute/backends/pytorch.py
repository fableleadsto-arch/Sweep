"""PyTorch backend — neural nets, local LLMs, diffusion, vLLM.

Optional: available only when ``torch`` is installed. Everything here is
lazy — probing never imports torch, execution loads it on demand.
"""

from __future__ import annotations

from typing import Any

from ...tools.common import CapabilityUnavailable
from ..base import BackendKind, ComputeBackend, ComputeJobResult, ComputeTask, ComputeTaskKind


class PyTorchBackend(ComputeBackend):
    id = "pytorch"
    label = "PyTorch"
    kind = BackendKind.PYTORCH
    required_libraries: tuple[str, ...] = ("torch",)
    install_hint = "Install: pip install -r requirements.companion-pytorch.txt"
    preferred_for = ("deep-learning", "local-llm", "diffusion", "timm", "vllm", "accelerate")
    runner_capabilities = {"deep-learning", "local-llm", "diffusion", "timm", "accelerate"}

    @property
    def available(self) -> bool:
        from ...tools.common import module_available

        return module_available("torch")

    def supports(self, task: ComputeTask) -> bool:
        # Framework backends only claim the capabilities they actually execute;
        # anything else falls through to CPU so light tasks never die on us.
        if task.kind == ComputeTaskKind.CAPABILITY:
            return task.capability in self.runner_capabilities
        return True

    def devices(self):
        from ..base import DeviceInfo

        torch = self.load()
        devices: list[DeviceInfo] = []
        try:
            cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:  # noqa: BLE001 - probe defensively
            cuda_count = 0
        for idx in range(cuda_count):
            try:
                name = torch.cuda.get_device_name(idx)
                props = torch.cuda.get_device_properties(idx)
                free = torch.cuda.mem_get_info(idx)[0] / (1024**2)
                total = (props.total_memory or 0) / (1024**2)
            except Exception:  # noqa: BLE001
                name, total, free = f"CUDA:{idx}", 0.0, None
            devices.append(
                DeviceInfo(
                    name=name, kind="cuda", index=idx,
                    memory_total_mb=round(total, 1),
                    memory_free_mb=round(free, 1) if free is not None else None,
                    arch=torch.version.cuda or "cuda",
                )
            )
        try:
            if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                devices.append(DeviceInfo(name="Apple MPS", kind="mps", index=0, arch="mps"))
        except Exception:  # noqa: BLE001 - probe defensively
            pass
        devices.append(DeviceInfo(name="CPU", kind="cpu", index=0))
        return devices

    def run_smoke(self, task: ComputeTask) -> ComputeJobResult:
        torch = self.load()
        device = self._resolve_device(task.device_preference)
        try:
            x = torch.randn(64, 64, device=device)
            y = (x @ x).sum().item()
            return ComputeJobResult(
                ok=True,
                backend=self.id,
                device=device,
                result={"matrix_sum": round(float(y), 4)},
                summary=f"PyTorch smoke test passed on {device}.",
                libraries_used=["torch"],
            )
        except Exception as exc:  # noqa: BLE001 - surface device errors cleanly
            return ComputeJobResult(
                ok=False, backend=self.id, device=device,
                error=f"PyTorch failed on {device}: {exc}",
            )

    def _resolve_device(self, preference: str) -> str:
        torch = self.load()
        pref = (preference or "").lower()
        try:
            if pref.startswith("cuda") and torch.cuda.is_available():
                return pref if pref.startswith("cuda:") else "cuda"
        except Exception:  # noqa: BLE001
            pass
        try:
            if pref.startswith("mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                return "mps"
        except Exception:  # noqa: BLE001
            pass
        return "cuda" if _torch_cuda(torch) else "cpu"

    def run_capability(self, task: ComputeTask) -> ComputeJobResult:
        from ...tools import ai, diffusion

        runners: dict[str, Any] = {
            "deep-learning": ai.run_deep_learning,
            "local-llm": ai.run_local_llm,
            "diffusion": diffusion.run_diffusion,
            "timm": diffusion.run_timm,
            "accelerate": diffusion.run_accelerate,
        }
        runner = runners.get(task.capability)
        if runner is None:
            raise CapabilityUnavailable(
                f"PyTorch backend has no handler for capability '{task.capability}'."
            )
        payload = dict(task.payload)
        payload["device_preference"] = task.device_preference or _device_hint()
        try:
            outcome = runner(payload)
        except CapabilityUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - tools surface errors as results
            return ComputeJobResult(
                ok=False, backend=self.id, error=str(exc)[:500],
                summary=f"PyTorch computation failed: {exc}"[:500],
            )
        result = outcome.get("result")
        if result is None or result is False:
            return ComputeJobResult(
                ok=False, backend=self.id, error=outcome.get("summary", "No result"),
                summary=str(outcome.get("summary", "No result")),
            )
        return ComputeJobResult(
            ok=True, backend=self.id, device=_device_hint(),
            result=result, summary=outcome.get("summary", "Done."),
            libraries_used=outcome.get("libraries_used", []),
        )


def _torch_cuda(torch) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _device_hint() -> str:
    return ""
