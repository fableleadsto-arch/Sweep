"""Hardware-aware task scheduler.

Picks the best backend for a `ComputeTask` by scoring candidates on:
  * framework availability + enable state,
  * task-kind preference (training → GPU; inference → ONNX/PyTorch; ...),
  * explicit hints (framework_hint, device_preference),
  * memory fit when a size estimate is supplied,
  * GPU presence when the device backends can use it.

The result explains *why* a backend won, which the diagnostics UI and the
compute endpoints surface directly. Never imports a heavy framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .base import BackendKind, ComputeTask
from .model_manager import ModelManager, ModelSpec

# Scoring constants (additive; the largest score wins).
_SCORE_CPU_BASE = 10
_SCORE_FRAMEWORK_BASE = 30
_SCORE_GPU_DEVICE = 40
_SCORE_HINT_MATCH = 50
_SCORE_PREFERRED_TASK = 20
_SCORE_DEVICE_PREFERENCE = 15

# Which task kinds a GPU device backend is especially good at.
_GPU_BOOST_KINDS = {"training", "inference", "generation", "embedding"}


@dataclass
class ScheduleDecision:
    """The scheduler's verdict for one task."""

    backend: Optional[str]
    device: str = ""
    score: float = 0.0
    reason: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "score": self.score,
            "reason": self.reason,
            "candidates": self.candidates,
            "rejected": self.rejected,
        }


def schedule(
    task: ComputeTask,
    manager: Any = None,
    available_backends: Optional[list[Any]] = None,
) -> ScheduleDecision:
    """Pick the best backend for ``task``.

    ``manager`` may be a BackendManager (recommended: supplies enable state,
    environment). ``available_backends`` overrides discovery for tests.
    """
    from .backend_manager import BackendManager

    if available_backends is None:
        manager = manager or BackendManager()
        available = manager.available_backends()
        env = manager.environment()
    else:
        available = available_backends
        env = None

    model = ModelSpec.from_dict(
        {
            "model_format": task.model_format,
            "task": task.kind.value,
            "precision": task.precision,
            "estimated_memory_mb": task.estimated_memory_mb,
        }
    )
    model_manager = ModelManager(env) if env is not None else ModelManager()

    hint = task.framework_hint
    device_pref = task.device_preference.lower()

    candidates: list[tuple[str, float, str, ComputeTask]] = []
    rejected: list[str] = []
    memory_rejected: list[str] = []

    for backend in available:
        if not backend.supports(task):
            rejected.append(backend.id)
            continue
        score = _base_score(backend.kind)
        reasons: list[str] = []

        if hint and backend.id == hint:
            score += _SCORE_HINT_MATCH
            reasons.append(f"framework_hint matches '{hint}'")
        elif hint:
            score -= 5
            reasons.append(f"ignored framework_hint '{hint}'")

        if task.kind.value in backend.preferred_for:
            score += _SCORE_PREFERRED_TASK
            reasons.append(f"preferred for {task.kind.value}")

        device = "cpu"
        if _is_gpu_backend(backend):
            if task.kind.value in _GPU_BOOST_KINDS:
                score += _SCORE_GPU_DEVICE
                reasons.append("GPU device for training/inference")
            if device_pref and (device_pref == backend.kind.value or device_pref.startswith(backend.kind.value)):
                score += _SCORE_DEVICE_PREFERENCE
                reasons.append(f"device_preference '{device_pref}'")
            device = device_pref if device_pref.startswith(backend.kind.value) else backend.kind.value
        else:
            if device_pref and device_pref.startswith(backend.kind.value):
                score += _SCORE_DEVICE_PREFERENCE
                reasons.append(f"device_preference '{device_pref}'")
                device = device_pref
            # CPU is always a valid fallback for any task.
            score += 2
            reasons.append("CPU fallback always valid")

        # Memory fit — a hard rejection when the estimate exceeds free memory.
        # A backend that cannot hold the model must never be scheduled, even as
        # a fallback, because the run would OOM.
        fits, fit_reason = model_manager.fits_memory(model, backend.id)
        if not fits:
            memory_rejected.append(f"{backend.id} ({fit_reason})")
            continue

        # The model format's natural backend gets a bonus.
        if task.model_format and task.model_format in ("pytorch", "tensorflow", "onnx"):
            natural = {"pytorch": "pytorch", "tensorflow": "tensorflow", "onnx": "onnx"}[task.model_format]
            if backend.id == natural:
                score += 5
                reasons.append(f"model_format '{task.model_format}' maps here")

        candidates.append(
            (backend.id, float(score), "; ".join(reasons) or backend.label, task)
        )

    if not candidates:
        if memory_rejected:
            return ScheduleDecision(
                backend=None,
                reason="No backend has enough free memory for this model. "
                f"({memory_rejected[0]})",
                rejected=rejected + memory_rejected,
            )
        return ScheduleDecision(
            backend=None,
            reason="No compute backend is available for this task. "
            "Install the matching framework profile (see diagnostics).",
            rejected=rejected,
        )

    candidates.sort(key=lambda c: c[1], reverse=True)
    best_id, best_score, best_reason, _ = candidates[0]

    best_backend = next((b for b in available if b.id == best_id), None)
    device = "cpu"
    if best_backend is not None:
        device = _best_device(best_backend, task)
    return ScheduleDecision(
        backend=best_id,
        device=device,
        score=best_score,
        reason=best_reason,
        candidates=[{"backend": b, "score": s, "reason": r} for b, s, r, _ in candidates],
        rejected=rejected,
    )


def _base_score(kind: BackendKind) -> float:
    if kind == BackendKind.CPU:
        return _SCORE_CPU_BASE
    if kind in (BackendKind.CUDA, BackendKind.ROCM, BackendKind.MPS):
        return _SCORE_FRAMEWORK_BASE + _SCORE_GPU_DEVICE
    return _SCORE_FRAMEWORK_BASE


def _is_gpu_backend(backend) -> bool:
    return backend.kind in (BackendKind.CUDA, BackendKind.ROCM, BackendKind.MPS)


def _best_device(backend, task: ComputeTask) -> str:
    pref = (task.device_preference or "").lower()
    if pref and pref.startswith(backend.kind.value):
        return pref
    if _is_gpu_backend(backend):
        return backend.kind.value
    return "cpu"


__all__ = ["ScheduleDecision", "schedule"]
