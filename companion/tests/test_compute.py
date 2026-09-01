"""Tests for the optional compute-backend layer (companion/compute/).

Covers the spec's core guarantees:

  * **Startup is dependency-free** — importing the brain service never loads a
    heavy ML framework (PyTorch, TensorFlow, ONNX Runtime, Hugging Face stack,
    ...). Verified in a clean interpreter so test ordering can't mask a leak.
  * **Probing is import-free** — `module_available` / the capability detector
    report availability from metadata + `find_spec`, never an import.
  * **Backend registry** — every framework/device backend instantiates, CPU is
    always available, status shapes are stable.
  * **Scheduler** — hint matching, GPU preference, memory hard-fail, graceful
    "install this" fallback when nothing can run the task.
  * **Backend manager** — enable/disable persistence, controlled install plans
    (never auto-install).
  * **Diagnostics** — structured report, recommendations, no crash when
    frameworks are missing.
  * **Workers** — the isolated subprocess worker boots and runs a CPU job to
    completion without any heavy framework present.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from companion.compute.backend_manager import BackendManager, INSTALLABLE_WHEELS
from companion.compute.backends import instantiate_all
from companion.compute.base import BackendKind, ComputeTask, ComputeTaskKind
from companion.compute.capability_detector import FRAMEWORK_PROFILES, PROFILE_REQUIREMENTS, detect
from companion.compute.diagnostics import diagnose
from companion.compute.scheduler import schedule
from companion.compute.workers import LocalSubprocessWorker, WorkerJob, WorkerStatus
from companion.tools.common import module_available

# Top-level packages that must never be imported by the brain service boot.
HEAVY_TOP_LEVEL_PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "tensorflow",
    "keras",
    "onnxruntime",
    "onnx",
    "transformers",
    "diffusers",
    "accelerate",
    "vllm",
    "bitsandbytes",
    "triton",
    "jax",
    "jaxlib",
    "llama_cpp",
    "spacy",
    "nltk",
    "xgboost",
    "lightgbm",
    "cv2",
    "litellm",
    "llama_index",
    "langchain",
    "crewai",
    "autogen",
    "pymilvus",
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _heavy_top_level_in_modules() -> set[str]:
    """Top-level heavy packages currently resident in ``sys.modules``."""
    return {m.split(".")[0] for m in sys.modules if m.split(".")[0] in HEAVY_TOP_LEVEL_PACKAGES}


def _assert_no_new_heavy(before: set[str], what: str) -> None:
    """Assert ``what`` did not import any heavy framework on top of ``before``.

    Other test files legitimately import heavy modules (e.g. xgboost) into the
    shared pytest process, so we compare against a pre-operation snapshot rather
    than expecting an empty process. The clean-interpreter guarantee itself is
    covered by `test_brain_service_boot_imports_no_heavy_framework`.
    """
    new = _heavy_top_level_in_modules() - before
    assert new == set(), f"{what} imported heavy frameworks: {sorted(new)}"


# ── startup guarantee (environments A–F) ─────────────────────────────────

def test_brain_service_boot_imports_no_heavy_framework() -> None:
    """A clean interpreter must boot the full app without any heavy ML import.

    This is the environment A–F proof: core-only installs (requirements.companion.txt)
    and every optional profile above it share this one invariant.
    """
    script = (
        "import sys\n"
        "import companion.main\n"  # noqa: F401 - full FastAPI app surface
        "heavy = sorted({m.split('.')[0] for m in sys.modules if m.split('.')[0] in "
        f"{HEAVY_TOP_LEVEL_PACKAGES!r}}})\n"
        "assert not heavy, f'heavy frameworks imported at boot: {heavy}'\n"
        "print('OK: no heavy framework imported at boot')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_heavy_backend_modules_not_loaded_by_app_import() -> None:
    """Importing the compute stack directly adds no heavy framework."""
    before = _heavy_top_level_in_modules()
    import companion.compute  # noqa: F401
    import companion.compute.backends  # noqa: F401
    import companion.compute.backend_manager  # noqa: F401
    import companion.compute.diagnostics  # noqa: F401
    import companion.compute.scheduler  # noqa: F401
    import companion.compute.workers  # noqa: F401

    _assert_no_new_heavy(before, "companion.compute stack")


# ── capability detector ──────────────────────────────────────────────────

def test_detect_returns_structured_environment() -> None:
    before = _heavy_top_level_in_modules()
    env = detect()
    _assert_no_new_heavy(before, "capability detect()")
    assert env.python.version
    assert env.cpu.physical_cores >= 1
    assert env.cpu.logical_cores >= 1
    assert env.memory.total_mb >= 0
    assert env.disk.total_mb >= 0
    assert "cpu" in env.compute_devices
    # Framework report includes every known profile mapping.
    assert set(FRAMEWORK_PROFILES) <= set(env.frameworks)
    torch_info = env.frameworks.get("torch")
    assert torch_info is not None
    assert torch_info.profile == "pytorch"
    assert torch_info.profile_requirements == "requirements.companion-pytorch.txt"


def test_every_profile_maps_to_a_requirements_file() -> None:
    for profile in set(FRAMEWORK_PROFILES.values()):
        assert profile in PROFILE_REQUIREMENTS, f"no requirements file for profile '{profile}'"
    for path in PROFILE_REQUIREMENTS.values():
        assert (REPO_ROOT / path).exists(), f"missing requirements file {path}"


def test_module_available_is_import_free() -> None:
    # Probing torch (installed or not) must not pull it into sys.modules.
    before = _heavy_top_level_in_modules()
    module_available("torch")
    module_available("tensorflow")
    module_available("onnxruntime")
    _assert_no_new_heavy(before, "module_available probes")


# ── backend registry ─────────────────────────────────────────────────────

EXPECTED_BACKEND_IDS = {"cpu", "pytorch", "tensorflow", "onnx", "cuda", "rocm", "mps"}


def test_registry_instantiates_every_backend() -> None:
    backends = instantiate_all()
    ids = {b.id for b in backends}
    assert ids == EXPECTED_BACKEND_IDS
    labels = {b.id: b.label for b in backends}
    assert labels["cpu"] == "CPU (core)"
    assert labels["cuda"].startswith("NVIDIA")


def test_cpu_backend_always_available() -> None:
    cpu = next(b for b in instantiate_all() if b.id == "cpu")
    assert cpu.available is True
    assert cpu.missing_libraries == []


def test_backend_status_shape() -> None:
    cpu = next(b for b in instantiate_all() if b.id == "cpu")
    status = cpu.status(enabled=True)
    assert status.id == "cpu"
    assert status.kind == BackendKind.CPU.value
    assert status.available is True
    assert status.enabled is True
    assert isinstance(status.required_libraries, list)
    assert isinstance(status.missing_libraries, list)
    assert isinstance(status.version, str)
    # CPU probes devices without importing anything heavy.
    assert status.devices and status.devices[0]["kind"] == "cpu"


def test_framework_backend_probe_never_imports() -> None:
    before = _heavy_top_level_in_modules()
    for backend in instantiate_all():
        if backend.id == "cpu":
            continue
        # Probes are find_spec-only; probing must never pull the framework in.
        assert backend.available in (True, False)
    _assert_no_new_heavy(before, "backend availability probes")


# ── scheduler ────────────────────────────────────────────────────────────

class _FakeBackend:
    """Minimal stand-in so the scheduler can be tested without frameworks."""

    def __init__(self, id: str, kind: BackendKind, preferred_for: tuple[str, ...] = ()) -> None:
        self.id = id
        self.kind = kind
        self.preferred_for = preferred_for

    def supports(self, task: ComputeTask) -> bool:
        return True


def test_schedule_hint_match_wins() -> None:
    available = [_FakeBackend("cpu", BackendKind.CPU), _FakeBackend("pytorch", BackendKind.PYTORCH)]
    task = ComputeTask.from_capability("deep-learning", {}, framework_hint="pytorch")
    decision = schedule(task, available_backends=available)
    assert decision.backend == "pytorch"
    assert any("hint" in reason.lower() for reason in [decision.reason] + [c["reason"] for c in decision.candidates])


def test_schedule_no_backend_explains_install() -> None:
    decision = schedule(ComputeTask.from_capability("deep-learning", {}), available_backends=[])
    assert decision.backend is None
    assert "install" in (decision.reason or "").lower()


def test_schedule_gpu_boost_for_training() -> None:
    available = [_FakeBackend("cpu", BackendKind.CPU), _FakeBackend("cuda", BackendKind.CUDA)]
    task = ComputeTask(kind=ComputeTaskKind.TRAINING, capability="fine-tune", payload={})
    decision = schedule(task, available_backends=available)
    assert decision.backend == "cuda"


def test_schedule_device_preference() -> None:
    available = [_FakeBackend("cpu", BackendKind.CPU), _FakeBackend("cuda", BackendKind.CUDA)]
    task = ComputeTask(kind=ComputeTaskKind.TRAINING, capability="fine-tune", payload={}, device_preference="cuda:0")
    decision = schedule(task, available_backends=available)
    assert decision.device == "cuda:0"


def test_schedule_memory_hard_fail() -> None:
    available = [_FakeBackend("cpu", BackendKind.CPU)]
    task = ComputeTask(kind=ComputeTaskKind.INFERENCE, capability="deep-learning", payload={}, estimated_memory_mb=1e15)
    decision = schedule(task, available_backends=available)
    assert decision.backend is None
    assert "memory" in (decision.reason or "").lower()


# ── backend manager ──────────────────────────────────────────────────────

def test_manager_enable_disable_persists(tmp_path: Path) -> None:
    config = str(tmp_path / "compute.json")
    manager = BackendManager(config_path=config)
    assert manager.is_enabled("cpu") is True  # cpu can never be disabled
    assert manager.set_enabled("pytorch", False) is True
    assert manager.is_enabled("pytorch") is False
    assert manager.set_enabled("does-not-exist", True) is False

    reloaded = BackendManager(config_path=config)
    assert reloaded.is_enabled("pytorch") is False
    reloaded.set_enabled("pytorch", True)
    assert BackendManager(config_path=config).is_enabled("pytorch") is True


def test_manager_install_plan_is_never_auto_install(tmp_path: Path) -> None:
    manager = BackendManager(config_path=str(tmp_path / "compute.json"))
    plan = manager.install_plan("does-not-exist")
    assert plan["ok"] is False

    plan = manager.install_plan("cuda")
    # CUDA has a registered wheel or profile path — the plan must say so but
    # must never have executed anything.
    assert plan["backend"] == "cuda"
    assert plan["method"] in ("wheel", "pip-profile", None)
    # A plan is pure information: backends stay in their default enabled state.
    assert manager.is_enabled("cuda") is True


def test_manager_execute_raises_when_no_backend(tmp_path: Path) -> None:
    from companion.tools.common import CapabilityUnavailable

    manager = BackendManager(config_path=str(tmp_path / "compute.json"))
    for backend in manager.backends.values():
        if backend.id != "cpu":
            manager.set_enabled(backend.id, False)
    with pytest.raises(CapabilityUnavailable):
        manager.execute(ComputeTask.from_capability("deep-learning", {}))


def test_manager_execute_cpu_capability(tmp_path: Path) -> None:
    manager = BackendManager(config_path=str(tmp_path / "compute.json"))
    result = manager.execute(ComputeTask.from_capability("math", {"data": "1 2 3 4"}))
    assert result.ok is True
    assert result.backend == "cpu"


# ── diagnostics ──────────────────────────────────────────────────────────

def test_diagnostics_report_shape(tmp_path: Path) -> None:
    report = diagnose(settings=None)
    data = report.to_dict()
    for key in ("python", "os", "cpu", "memory", "disk", "gpus", "compute_devices",
                "frameworks", "backends", "smoke_tests", "recommendations", "status"):
        assert key in data
    assert data["status"] in ("ready", "degraded", "minimal")
    backend_ids = {b["id"] for b in data["backends"]}
    assert EXPECTED_BACKEND_IDS <= backend_ids
    # Smoke results only ever cover known backends.
    assert set(data["smoke_tests"]) <= EXPECTED_BACKEND_IDS
    # CPU is the guaranteed working backend → status can never be worse than "minimal".
    assert data["status"] in ("ready", "degraded", "minimal")
    assert isinstance(data["recommendations"], list)


def test_diagnostics_recommendations_when_no_deep_learning(tmp_path: Path) -> None:
    report = diagnose(settings=None)
    data = report.to_dict()
    torch_info = next(f for f in data["frameworks"] if f["name"] == "torch")
    tf_info = next(f for f in data["frameworks"] if f["name"] == "tensorflow")
    if not torch_info["installed"] and not tf_info["installed"]:
        assert any("deep-learning framework" in rec for rec in data["recommendations"])


# ── workers ──────────────────────────────────────────────────────────────

def test_worker_status_serialization() -> None:
    status = WorkerStatus(job_id="abc", state="running", progress=0.5, message="warm")
    data = status.to_dict()
    assert data["state"] == "running"
    roundtrip = WorkerStatus(**data)
    assert roundtrip.progress == 0.5


def test_local_subprocess_worker_runs_cpu_job(tmp_path: Path) -> None:
    """The isolated worker process must boot and finish a CPU job end-to-end."""
    worker = LocalSubprocessWorker(workdir=str(tmp_path / "workers"))
    job = WorkerJob(
        id="cpu-smoke",
        kind="capability",
        backend="cpu",
        spec={"capability": "math", "data": "1 2 3 4"},
    )
    job_id = worker.submit(job)
    try:
        deadline = time.monotonic() + 60
        status = worker.status(job_id)
        while time.monotonic() < deadline and status.state in ("queued", "running", "unknown"):
            time.sleep(0.5)
            status = worker.status(job_id)
        assert status.state == "done", f"worker did not finish: {status.state} — {status.error}"
        assert status.result is not None
    finally:
        worker.shutdown()


# ── wheels registry sanity ───────────────────────────────────────────────

def test_registered_wheel_names_are_explicit() -> None:
    assert INSTALLABLE_WHEELS
    for name in INSTALLABLE_WHEELS.values():
        assert name, "every installable backend must name a concrete bundled wheel"
