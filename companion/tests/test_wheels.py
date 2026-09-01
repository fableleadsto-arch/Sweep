"""Tests for the bundled-wheel provisioning capability (tools/wheels.py).

Verifies the status inventory, the registry lock (unknown wheels rejected),
the gating (installs disabled unless COMPANION_ALLOW_WHEEL_INSTALL is set),
and that a real install command targets the exact stored wheel path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from companion.capabilities import CAPABILITIES
from companion.tools import wheels
from companion.tools.vendor_loader import bundled_wheels, wheel_present

ALLOWED = SimpleNamespace(allow_wheel_install=True)
DENIED = SimpleNamespace(allow_wheel_install=False)


# ── status ───────────────────────────────────────────────────────────────

def test_status_reports_inventory() -> None:
    outcome = wheels.run_wheel_install({"params": {"mode": "status"}, "_settings": DENIED})
    assert outcome["result"]["registry_root"]
    entries = {w["name"]: w for w in outcome["result"]["wheels"]}
    for wheel in bundled_wheels():
        assert wheel.name in entries
        assert entries[wheel.name]["stored"] is wheel_present(wheel.name)
        # Platform surfaces so agents know which wheels can install on THIS host
        # (nvidia wheels are Linux-only and would fail a Windows pip install).
        assert entries[wheel.name]["platform"] in ("win", "linux", "any")
        # Windows wheels map to an importable module; Linux-only nvidia wheels
        # report module=None (they provide nvidia.* submodules, not importable
        # on this platform) — that is honest, not an error.
        if wheel.platform == "linux":
            assert entries[wheel.name]["module"] is None
            assert entries[wheel.name]["platform"] == "linux"
        else:
            assert entries[wheel.name]["module"]
    assert outcome["result"]["install_allowed"] is False


def test_status_reflects_gating_setting() -> None:
    outcome = wheels.run_wheel_install({"params": {"mode": "status"}, "_settings": ALLOWED})
    assert outcome["result"]["install_allowed"] is True


def test_status_default_mode() -> None:
    outcome = wheels.run_wheel_install({"_settings": DENIED})
    assert "wheels" in outcome["result"]


def test_installed_rejects_vendored_spec(monkeypatch) -> None:
    # A spec resolving to companion/vendor/ (vendored source tree) must NOT
    # count as installed — only site-packages installs do.
    import importlib.util

    class _FakeSpec:
        origin = str(__import__("companion.tools.vendor_loader", fromlist=["VENDOR_ROOT"]).VENDOR_ROOT) + "/tensorflow/__init__.py"

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: _FakeSpec())
    assert wheels._module_installed("tensorflow") is False


# ── registry lock ────────────────────────────────────────────────────────

def test_unknown_wheel_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown bundled wheel"):
        wheels.run_wheel_install(
            {"params": {"mode": "install", "wheel": "not-a-real-wheel"}, "_settings": ALLOWED}
        )


def test_registered_but_missing_wheel_rejected() -> None:
    # Monkeypatch disk presence to simulate a registered-but-not-downloaded
    # wheel: the error must explain what to download, never run pip.
    import companion.tools.wheels as wheels_mod

    orig = wheels_mod.wheel_present
    wheels_mod.wheel_present = lambda name: False
    try:
        name = bundled_wheels()[0].name
        with pytest.raises(ValueError, match="not stored on disk"):
            wheels.run_wheel_install(
                {"params": {"mode": "install", "wheel": name}, "_settings": ALLOWED}
            )
    finally:
        wheels_mod.wheel_present = orig


def test_missing_wheel_name_rejected() -> None:
    with pytest.raises(ValueError, match="params.wheel"):
        wheels.run_wheel_install({"params": {"mode": "install"}, "_settings": ALLOWED})


def test_bad_mode_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        wheels.run_wheel_install({"params": {"mode": "explode"}, "_settings": DENIED})


def test_capability_keywords_mention_dry_run() -> None:
    cap = next(c for c in CAPABILITIES if c.id == "wheel-install")
    assert "dry" in " ".join(cap.keywords) or "dry_run" in cap.description


# ── gating ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not wheel_present("torch-cu126-win"),
    reason="PyTorch CUDA wheel not on disk (download to companion/vendor/archives/)",
)
def test_install_gated_off_by_default() -> None:
    outcome = wheels.run_wheel_install(
        {"params": {"mode": "install", "wheel": "torch-cu126-win"}, "_settings": DENIED}
    )
    assert outcome["result"]["ready"] is False
    assert outcome["result"]["install_allowed"] is False
    assert "COMPANION_ALLOW_WHEEL_INSTALL" in outcome["summary"]


# ── install command targets the exact stored wheel ───────────────────────

@pytest.mark.skipif(
    not wheel_present("torch-cu126-win"),
    reason="PyTorch CUDA wheel not on disk (download to companion/vendor/archives/)",
)
def test_install_runs_pip_on_registry_path(monkeypatch) -> None:
    import subprocess

    calls: list[list[str]] = []

    def _fake_run(command, capture_output=False, text=False, timeout=0, check=False):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, returncode=0, stdout="Success", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    outcome = wheels.run_wheel_install(
        {"params": {"mode": "install", "wheel": "torch-cu126-win"}, "_settings": ALLOWED}
    )
    assert outcome["result"]["ready"] is True
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0].endswith("python") or cmd[0].endswith("python.exe")
    assert "-m" in cmd and "pip" in cmd
    wheel_path = next(w.path for w in bundled_wheels() if w.name == "torch-cu126-win")
    assert str(wheel_path) in cmd
    # The command must contain no URL and no arbitrary package name.
    assert not any("http" in part for part in cmd)


@pytest.mark.skipif(
    not wheel_present("torch-cu126-win"),
    reason="PyTorch CUDA wheel not on disk (download to companion/vendor/archives/)",
)
def test_install_failure_surfaces_output(monkeypatch) -> None:
    import subprocess

    def _fake_run(command, capture_output=False, text=False, timeout=0, check=False):
        return subprocess.CompletedProcess(command, returncode=1, stdout="", stderr="boom: deps")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    outcome = wheels.run_wheel_install(
        {"params": {"mode": "install", "wheel": "torch-cu126-win"}, "_settings": ALLOWED}
    )
    assert outcome["result"]["ready"] is False
    assert outcome["result"]["returncode"] == 1
    assert "boom: deps" in outcome["result"]["pip_output_tail"]


@pytest.mark.skipif(
    not wheel_present("torch-cu126-win"),
    reason="PyTorch CUDA wheel not on disk (download to companion/vendor/archives/)",
)
def test_dry_run_passes_flag_and_reports_no_change(monkeypatch) -> None:
    import subprocess

    calls: list[list[str]] = []

    def _fake_run(command, capture_output=False, text=False, timeout=0, check=False):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, returncode=0, stdout="Would install", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    outcome = wheels.run_wheel_install(
        {"params": {"mode": "dry_run", "wheel": "torch-cu126-win"}, "_settings": ALLOWED}
    )
    assert outcome["result"]["ready"] is True
    assert outcome["result"]["dry_run"] is True
    assert "--dry-run" in calls[0]
    assert "nothing was changed" in outcome["summary"]


@pytest.mark.skipif(
    not wheel_present("torch-cu126-win"),
    reason="PyTorch CUDA wheel not on disk (download to companion/vendor/archives/)",
)
def test_dry_run_gated_off_by_default() -> None:
    # Even validation (dry-run) must respect the gate: installs — including
    # the dry-run resolution pass — are only permitted when enabled.
    outcome = wheels.run_wheel_install(
        {"params": {"mode": "dry_run", "wheel": "torch-cu126-win"}, "_settings": DENIED}
    )
    assert outcome["result"]["ready"] is False
    assert outcome["result"]["install_allowed"] is False


# ── engine-level integration (CapabilityEngine → tool → pip) ────────────

def _asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


@pytest.mark.skipif(
    not wheel_present("onnxruntime-gpu-win"),
    reason="onnxruntime-gpu wheel not on disk (download to companion/vendor/archives/)",
)
def test_engine_dry_run_routes_and_targets_stored_wheel(monkeypatch) -> None:
    """The full CapabilityEngine path — resolve → gate → tool → pip command —
    must route a dry_run request to wheel-install and build the pip command
    against the EXACT stored wheel path (proves gating + registry lock hold
    end-to-end, not just at the tool layer)."""
    import subprocess

    from companion.capabilities import CapabilityEngine
    from companion.schemas import ComputeRequest

    calls: list[list[str]] = []

    def _fake_run(command, capture_output=False, text=False, timeout=0, check=False):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, returncode=0, stdout="Would install onnxruntime-gpu", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    engine = CapabilityEngine(ALLOWED)
    result = _asyncio_run(
        engine.run(
            ComputeRequest(
                task="dry run the onnxruntime gpu wheel",
                capability="wheel-install",
                params={"mode": "dry_run", "wheel": "onnxruntime-gpu-win"},
            )
        )
    )
    assert result.ok
    assert result.capability == "wheel-install"
    assert result.result["dry_run"] is True
    assert result.result["ready"] is True
    assert len(calls) == 1
    cmd = calls[0]
    assert "--dry-run" in cmd
    wheel_path = next(w.path for w in bundled_wheels() if w.name == "onnxruntime-gpu-win")
    assert str(wheel_path) in cmd
    assert not any("http" in part for part in cmd)  # registry lock: no URLs reach pip


def test_engine_wheel_install_catalog_available() -> None:
    from companion.capabilities import CapabilityEngine

    catalog = {c.id: c for c in CapabilityEngine().catalog()}
    assert "wheel-install" in catalog
    assert catalog["wheel-install"].available  # framework-free, always available


# ── boot inventory log ──────────────────────────────────────────────────

def test_boot_inventory_log_runs(caplog) -> None:
    """The FastAPI startup hook must log the vendor inventory without raising
    (even though /health and API routes do not depend on it)."""
    import logging

    from companion.main import log_vendor_inventory

    with caplog.at_level(logging.INFO, logger="companion"):
        log_vendor_inventory()
    combined = "\n".join(r.getMessage() for r in caplog.records)
    assert "vendor:" in combined
    assert "wheels" in combined


# ── catalog integration ──────────────────────────────────────────────────

def test_wheel_install_capability_registered() -> None:
    ids = {cap.id for cap in CAPABILITIES}
    assert "wheel-install" in ids
    cap = next(c for c in CAPABILITIES if c.id == "wheel-install")
    assert cap.available  # framework-free capability is always available
    assert cap.libraries == []
    assert "install torch" in cap.keywords
