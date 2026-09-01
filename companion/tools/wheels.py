"""Bundled-wheel provisioning capability.

Relay stores pre-built platform wheels (the PyTorch CUDA bundle, TensorFlow,
ONNX Runtime GPU, ...) in ``companion/vendor/archives/``. Those wheels are too
large to commit (GitHub rejects files > 100MB), so they live in the working
tree. This capability lets Relay *inspect* what is stored and — when the host
is explicitly configured with ``COMPANION_ALLOW_WHEEL_INSTALL=1`` — install a
stored wheel so the brain can self-provision heavy frameworks.

Safety:

  * **Registry-locked.** The wheel name must exist in the vendored registry
    (``vendor_loader.bundled_wheels()``). Arbitrary file paths, URLs or
    package names are never accepted.
  * **Gated.** Installing mutates the Python environment, so it requires the
    ``allow_wheel_install`` setting (default OFF).
  * **Local only.** Installation always targets the exact wheel file stored in
    the registry — pip never resolves from the network unless the wheel's own
    declared dependencies are missing (e.g. nvidia runtime packages).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import Any

from .vendor_loader import ARCHIVES_DIR, VENDOR_ROOT, bundled_wheels, wheel_path, wheel_present

# Known top-level module(s) each registry wheel provides once installed, used
# for the "installed?" status column (import-based, no pip subprocess needed).
_WHEEL_MODULES = {
    "torch-cu126-win": "torch",
    "torchvision-cu126-win": "torchvision",
    "torchaudio-cu126-win": "torchaudio",
    "tensorflow-cp313-win": "tensorflow",
    "onnxruntime-gpu-win": "onnxruntime",
}


def _module_installed(name: str) -> bool:
    """True only when the module resolves from site-packages.

    ``find_spec`` would also match the vendored source trees on ``sys.path``
    (companion/vendor/), and those are NOT installed — e.g. vendored
    tensorflow still needs its compiled dependencies. Resolving specs whose
    origin lives under VENDOR_ROOT keeps the status column honest.
    """
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    origin = (spec.origin or "").replace("\\", "/")
    vendor_root = str(VENDOR_ROOT).replace("\\", "/") + "/"
    return not origin.startswith(vendor_root)


def _status(settings: Any) -> dict[str, Any]:
    """Inventory of known bundled wheels: stored on disk + installed in env."""
    entries = []
    for wheel in bundled_wheels():
        module = _WHEEL_MODULES.get(wheel.name)
        entries.append(
            {
                "name": wheel.name,
                "version": wheel.version,
                "filename": wheel.filename,
                "platform": wheel.platform or "any",
                "stored": wheel_present(wheel.name),
                "installed": bool(module) and _module_installed(module),
                "module": module,
                "source": wheel.source,
            }
        )
    return {
        "result": {
            "registry_root": str(ARCHIVES_DIR),
            "wheels": entries,
            "install_allowed": _install_allowed(settings),
        },
        "summary": (
            f"{sum(1 for w in entries if w['stored'])}/{len(entries)} bundled "
            f"wheel(s) stored locally; "
            f"{sum(1 for w in entries if w['installed'])} installed in this environment."
        ),
        "libraries_used": [],
    }


def _install_allowed(settings: Any) -> bool:
    if settings is None:
        return False
    return bool(getattr(settings, "allow_wheel_install", False))


def run_wheel_install(payload: dict[str, Any]) -> dict[str, Any]:
    """Inspect stored wheels (status) or install one from the registry."""
    params = payload.get("params") or {}
    settings = payload.get("_settings")
    mode = str(params.get("mode") or "status").lower()

    if mode == "status":
        return _status(settings)

    if mode not in ("install", "dry_run"):
        raise ValueError("wheel-install mode must be 'status', 'install' or 'dry_run'.")

    dry_run = mode == "dry_run" or bool(params.get("dry_run"))

    name = str(params.get("wheel") or params.get("name") or "").strip()
    if not name:
        raise ValueError("`params.wheel` is required (a registry wheel name, e.g. 'torch-cu126-win').")
    if not wheel_present(name):
        known = sorted(w.name for w in bundled_wheels())
        if name not in known:
            raise ValueError(
                f"Unknown bundled wheel '{name}'. Known registry: {', '.join(known) or 'none'}."
            )
        wheel = next(w for w in bundled_wheels() if w.name == name)
        raise ValueError(
            f"Bundled wheel '{name}' is registered but not stored on disk. "
            f"Download it to: {wheel.path}  (source: {wheel.source})"
        )

    if not _install_allowed(settings):
        return {
            "result": {
                "capability": "wheel-install",
                "ready": False,
                "install_allowed": False,
                "wheel": name,
                "hint": (
                    "Set COMPANION_ALLOW_WHEEL_INSTALL=1 in the brain service "
                    "environment to permit installing stored wheels."
                ),
            },
            "summary": (
                "Wheel installs are disabled. Enable them with "
                "COMPANION_ALLOW_WHEEL_INSTALL=1 (this mutates the Python "
                "environment, so it is off by default)."
            ),
            "libraries_used": [],
        }

    path = wheel_path(name)
    if path is None:
        raise ValueError(f"Bundled wheel '{name}' is not on disk.")

    # `--no-deps` is deliberately NOT used: the wheel's declared dependencies
    # (e.g. nvidia CUDA runtime packages) must resolve. The wheel itself comes
    # from the locked local path; only declared deps may come from the index.
    command = [sys.executable, "-m", "pip", "install", "--no-cache-dir"]
    if dry_run:
        # --dry-run resolves + reports what WOULD change without touching the
        # environment — a safe validation pass for provisioning decisions.
        command.append("--dry-run")
    command.append(str(path))
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=float(params.get("timeout_seconds") or 600),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "result": {"capability": "wheel-install", "ready": False, "wheel": name, "timed_out": True},
            "summary": f"pip install of '{name}' exceeded the timeout and was cancelled.",
            "libraries_used": [],
        }
    except Exception as exc:  # noqa: BLE001 - surface subprocess failures cleanly
        return {
            "result": {"capability": "wheel-install", "ready": False, "wheel": name, "error": str(exc)},
            "summary": f"Failed to run pip for '{name}': {exc}",
            "libraries_used": [],
        }

    ok = proc.returncode == 0
    tail = "\n".join((proc.stdout or proc.stderr or "").strip().splitlines()[-12:])
    installed = _module_installed(_WHEEL_MODULES.get(name, "")) if _WHEEL_MODULES.get(name) else ok
    if dry_run:
        return {
            "result": {
                "capability": "wheel-install",
                "dry_run": True,
                "ready": ok,
                "wheel": name,
                "version": next((w.version for w in bundled_wheels() if w.name == name), ""),
                "installed": installed,
                "returncode": proc.returncode,
                "pip_output_tail": tail[-2000:],
            },
            "summary": (
                f"Dry-run for '{name}': pip reports install would succeed "
                f"(nothing was changed)."
                if ok
                else f"Dry-run for '{name}' failed (rc={proc.returncode}). See pip_output_tail."
            ),
            "libraries_used": [],
        }
    return {
        "result": {
            "capability": "wheel-install",
            "ready": ok,
            "wheel": name,
            "version": next((w.version for w in bundled_wheels() if w.name == name), ""),
            "installed": installed,
            "returncode": proc.returncode,
            "pip_output_tail": tail[-2000:],
        },
        "summary": (
            f"Installed '{name}' from the stored bundle."
            if ok
            else f"pip install of '{name}' failed (rc={proc.returncode}). See pip_output_tail."
        ),
        "libraries_used": [],
    }

__all__ = ["run_wheel_install"]
