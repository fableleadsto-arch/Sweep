"""Backend manager — discovery, state, controlled installs.

Holds the canonical registry of `ComputeBackend` instances, probes them,
persists which are enabled/disabled, and exposes the *controlled* install
path (never auto-installs: a backend is installed only when the host enables
wheel installs or the operator runs the explicit install action).

Persistence is a small JSON file (``compute_config_file``) recording the
user's enable/disable choices and any resolved install state.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from .backends import backend_for, instantiate_all
from .base import BackendStatus, ComputeBackend, ComputeTask
from .capability_detector import detect

# Backends whose wheel can be installed from the local registry, mapped to the
# registry wheel name (see companion/tools/wheels.py).
INSTALLABLE_WHEELS: dict[str, str] = {
    "pytorch": "torch-cu126-win",
    "cuda": "torch-cu126-win",
    "tensorflow": "tensorflow-cp313-win",
    "onnx": "onnxruntime-gpu-win",
}

# Backends whose install is a plain `pip install -r <profile>`.
INSTALLABLE_PROFILES: dict[str, str] = {
    "pytorch": "requirements.companion-pytorch.txt",
    "tensorflow": "requirements.companion-tensorflow.txt",
    "onnx": "requirements.companion-onnx.txt",
    "training": "requirements.companion-training.txt",
    "cuda": "requirements.companion-cuda.txt",
}


class BackendManager:
    """Registry + state for the compute backends."""

    def __init__(self, settings: Any = None, config_path: Optional[str] = None) -> None:
        self.settings = settings
        if config_path:
            self._config_path = Path(config_path)
        else:
            default = getattr(settings, "compute_config_file", None) or ".relayhub/compute-config.json"
            self._config_path = Path(default)
        self._state: dict[str, Any] = {"enabled": {}, "installed": {}}
        self._load_state()
        self._backends: dict[str, ComputeBackend] = {
            backend.id: backend for backend in instantiate_all()
        }
        self._environment = None

    # ── persistence ───────────────────────────────────────────────────

    def _load_state(self) -> None:
        try:
            if self._config_path.exists():
                data = json.loads(self._config_path.read_text(encoding="utf-8"))
                self._state["enabled"] = data.get("enabled", {})
                self._state["installed"] = data.get("installed", {})
        except (OSError, ValueError):
            self._state = {"enabled": {}, "installed": {}}

    def save_state(self) -> None:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            self._config_path.write_text(
                json.dumps(self._state, indent=2), encoding="utf-8"
            )
        except OSError:
            pass  # state persistence is best-effort; never crash a request

    # ── registry access ───────────────────────────────────────────────

    @property
    def backends(self) -> dict[str, ComputeBackend]:
        return self._backends

    def get(self, backend_id: str) -> Optional[ComputeBackend]:
        return self._backends.get(backend_id)

    def environment(self):
        if self._environment is None:
            self._environment = detect()
        return self._environment

    def is_enabled(self, backend_id: str) -> bool:
        # cpu is always enabled; device backends start enabled but gated on HW.
        if backend_id in ("cpu",):
            return True
        return bool(self._state["enabled"].get(backend_id, True))

    def set_enabled(self, backend_id: str, enabled: bool) -> bool:
        if backend_id not in self._backends:
            return False
        self._state["enabled"][backend_id] = bool(enabled)
        self.save_state()
        return True

    # ── status ────────────────────────────────────────────────────────

    def status(self) -> list[BackendStatus]:
        """Live status for every known backend (import-free probes only)."""
        return [backend.status(enabled=self.is_enabled(backend.id)) for backend in self._backends.values()]

    def status_of(self, backend_id: str) -> Optional[BackendStatus]:
        backend = self._backends.get(backend_id)
        if backend is None:
            return None
        return backend.status(enabled=self.is_enabled(backend_id))

    def available_backends(self) -> list[ComputeBackend]:
        """Backends that are enabled AND currently available."""
        return [
            b for b in self._backends.values()
            if self.is_enabled(b.id) and b.available
        ]

    # ── controlled install ────────────────────────────────────────────

    def install_plan(self, backend_id: str) -> dict[str, Any]:
        """What installing ``backend_id`` WOULD do (never executes anything)."""
        backend = self._backends.get(backend_id)
        if backend is None:
            return {"ok": False, "error": f"Unknown backend '{backend_id}'."}
        plan: dict[str, Any] = {"backend": backend_id, "method": None, "wheel": None, "profile": None}
        if backend.available:
            plan["already_available"] = True
            plan["note"] = f"{backend.label} is already available — nothing to install."
            return plan
        wheel = INSTALLABLE_WHEELS.get(backend_id)
        if wheel:
            plan["method"] = "wheel"
            plan["wheel"] = wheel
            from ..tools.vendor_loader import wheel_present

            plan["wheel_stored"] = wheel_present(wheel)
        else:
            profile = INSTALLABLE_PROFILES.get(backend_id)
            if profile:
                plan["method"] = "pip-profile"
                plan["profile"] = profile
        if plan["method"] is None:
            plan["note"] = f"No bundled install path for '{backend_id}'. Install its framework manually."
        return plan

    def install(self, backend_id: str, dry_run: bool = True) -> dict[str, Any]:
        """Install a backend through the *controlled* wheel-install capability.

        Wheel installs mutate the Python environment, so they are gated on
        ``allow_wheel_install`` and prefer ``dry_run`` (validate only). This
        mirrors the capability engine's wheel-install tool.
        """
        plan = self.install_plan(backend_id)
        if plan.get("already_available"):
            return {**plan, "ok": True}
        if plan.get("method") != "wheel":
            return {
                **plan,
                "ok": False,
                "error": (
                    f"No bundled wheel is registered for '{backend_id}'. "
                    f"Install its framework profile manually: pip install -r {plan.get('profile') or 'requirements.companion-ai.txt'}"
                ),
            }
        from ..tools import wheels
        from ..tools.common import CapabilityUnavailable

        allowed = bool(getattr(self.settings, "allow_wheel_install", False))
        if not allowed and not dry_run:
            return {
                **plan,
                "ok": False,
                "error": (
                    "Wheel installs are disabled. Set COMPANION_ALLOW_WHEEL_INSTALL=1 "
                    "in the brain service environment to permit installing stored wheels, "
                    "or re-run with dry_run=True to validate only."
                ),
            }
        try:
            outcome = wheels.run_wheel_install(
                {
                    "params": {"mode": "dry_run" if dry_run else "install", "wheel": plan["wheel"]},
                    "_settings": self.settings,
                }
            )
        except (ValueError, CapabilityUnavailable) as exc:
            return {**plan, "ok": False, "error": str(exc)}
        result = outcome.get("result", {})
        return {
            **plan,
            "ok": bool(result.get("ready")),
            "dry_run": dry_run,
            "wheel_result": result,
            "summary": outcome.get("summary", ""),
        }

    # ── execution ─────────────────────────────────────────────────────

    def execute(self, task: ComputeTask) -> Any:
        """Schedule ``task`` to its best backend and run it.

        Returns a ComputeJobResult. Raises CapabilityUnavailable when no
        backend can run the task.
        """
        from ..tools.common import CapabilityUnavailable
        from .scheduler import schedule as _schedule

        decision = _schedule(task, manager=self)
        if decision.backend is None:
            raise CapabilityUnavailable(
                decision.reason or "No compute backend is available for this task."
            )
        backend = self._backends.get(decision.backend)
        if backend is None:
            raise CapabilityUnavailable(f"Backend '{decision.backend}' not found.")
        run_task = task
        if task.device_preference and decision.device and ":" in task.device_preference:
            pass
        return backend.run(run_task)


# ── module-level convenience ────────────────────────────────────────────

def discover_status(settings: Any = None) -> list[BackendStatus]:
    return BackendManager(settings).status()


__all__ = ["BackendManager", "INSTALLABLE_PROFILES", "INSTALLABLE_WHEELS", "discover_status"]
