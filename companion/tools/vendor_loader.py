"""Vendored framework registry + import fallback (see companion/vendor/README.md).

Relay ships the source of the largest AI/ML frameworks inside the repo so the
brain service can work from local source instead of re-fetching them:

  * Importable packages — pure-Python source trees (transformers, langchain,
    llama_index, autogen, crewai) that the capability engine can import
    directly when the framework is not pip-installed.
  * Source archives — tarballs of compiled/CUDA giants (torch, tensorflow,
    vllm, jax, keras) stored for offline builds.
  * Bundled wheels — pre-built platform wheels (e.g. the PyTorch CUDA bundle:
    torch/torchvision/torchaudio +cu126 for cp313-win_amd64) downloaded and
    stored locally. These are large (multi-GB) and are NOT committed to git
    (GitHub rejects files > 100MB) — they live in the working tree and are
    reported as "present"/"missing" so Relay knows exactly what is on disk.

Rules:

  * pip-installed frameworks always win — vendored paths are appended to
    ``sys.path`` (never prepended), so ``find_spec`` / ``import`` resolve to
    site-packages first.
  * ``module_available()``/``load()`` in ``tools.common`` consult this module,
    so a framework is "available" when it is installed *or* vendored.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

# companion/vendor/ — the root that holds both importable packages and
# archives. Resolved from this file's location so it works regardless of CWD.
VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor"
ARCHIVES_DIR = VENDOR_ROOT / "archives"


@dataclass(frozen=True)
class VendoredArchive:
    """One stored source tarball or wheel (offline build/install source).

    ``platform`` is optional and only meaningful for wheels: "win", "linux"
    or "" (any/unknown). Source archives always carry "".
    """

    name: str
    version: str
    license: str
    filename: str
    source: str
    platform: str = ""

    @property
    def path(self) -> Path:
        return ARCHIVES_DIR / self.filename


# Importable pure-Python packages: dir name under VENDOR_ROOT.
_IMPORTABLE: dict[str, Path] = {
    "transformers": VENDOR_ROOT / "transformers",
    "langchain": VENDOR_ROOT / "langchain",
    "llama_index": VENDOR_ROOT / "llama_index",
    "autogen": VENDOR_ROOT / "autogen",
    "crewai": VENDOR_ROOT / "crewai",
}

# Source archives (no PyPI sdist for torch/tensorflow → GitHub tags).
_ARCHIVES: dict[str, VendoredArchive] = {
    "torch": VendoredArchive(
        name="torch",
        version="2.9.1",
        license="BSD-3-Clause",
        filename="torch-v2.9.1-src.tar.gz",
        source="https://github.com/pytorch/pytorch (tag v2.9.1)",
    ),
    "tensorflow": VendoredArchive(
        name="tensorflow",
        version="2.21.0",
        license="Apache-2.0",
        filename="tensorflow-v2.21.0-src.tar.gz",
        source="https://github.com/tensorflow/tensorflow (tag v2.21.0)",
    ),
    "vllm": VendoredArchive(
        name="vllm",
        version="0.27.0",
        license="Apache-2.0",
        filename="vllm-0.27.0.tar.gz",
        source="https://pypi.org/project/vllm",
    ),
    "jax": VendoredArchive(
        name="jax",
        version="0.11.0",
        license="Apache-2.0",
        filename="jax-0.11.0.tar.gz",
        source="https://pypi.org/project/jax",
    ),
    "keras": VendoredArchive(
        name="keras",
        version="3.15.1",
        license="Apache-2.0",
        filename="keras-3.15.1.tar.gz",
        source="https://pypi.org/project/keras",
    ),
}

# Bundled pre-built wheels stored locally (not committed to git — too large
# for GitHub's 100MB/file limit). Filename: exact wheel file under ARCHIVES_DIR.
_BUNDLED_WHEELS: dict[str, VendoredArchive] = {
    "torch-cu126-win": VendoredArchive(
        name="torch-cu126-win",
        version="2.9.1+cu126",
        license="BSD-3-Clause (torch) + NVIDIA CUDA runtime license",
        filename="torch-2.9.1+cu126-cp313-cp313-win_amd64.whl",
        source="https://download.pytorch.org/whl/cu126/torch-2.9.1%2Bcu126-cp313-cp313-win_amd64.whl",
        platform="win",
    ),
    "torchvision-cu126-win": VendoredArchive(
        name="torchvision-cu126-win",
        version="0.24.1+cu126",
        license="BSD-3-Clause",
        filename="torchvision-0.24.1+cu126-cp313-cp313-win_amd64.whl",
        source="https://download.pytorch.org/whl/cu126/torchvision-0.24.1%2Bcu126-cp313-cp313-win_amd64.whl",
        platform="win",
    ),
    "torchaudio-cu126-win": VendoredArchive(
        name="torchaudio-cu126-win",
        version="2.9.1+cu126",
        license="BSD-3-Clause",
        filename="torchaudio-2.9.1+cu126-cp313-cp313-win_amd64.whl",
        source="https://download.pytorch.org/whl/cu126/torchaudio-2.9.1%2Bcu126-cp313-cp313-win_amd64.whl",
        platform="win",
    ),
    # TensorFlow ships NO GPU wheel for Windows (GPU builds are Linux/WSL-only
    # since 2.11) — this cp313-win wheel is the CPU build; the GPU build on
    # Linux is `tensorflow[and-cuda]`.
    "tensorflow-cp313-win": VendoredArchive(
        name="tensorflow-cp313-win",
        version="2.21.0",
        license="Apache-2.0",
        filename="tensorflow-2.21.0-cp313-cp313-win_amd64.whl",
        source="https://pypi.org/project/tensorflow",
        platform="win",
    ),
    "onnxruntime-gpu-win": VendoredArchive(
        name="onnxruntime-gpu-win",
        version="1.28.0",
        license="MIT",
        filename="onnxruntime_gpu-1.28.0-cp313-cp313-win_amd64.whl",
        source="https://pypi.org/project/onnxruntime-gpu",
        platform="win",
    ),
}

# nvidia CUDA 12.6 runtime wheels — exact pins torch 2.9.1+cu126 declares on
# Linux (see archives/nvidia-cu126-linux.txt). Linux-only: the Windows torch
# wheel bundles the CUDA runtime in-wheel and needs none of these.
_NVIDIA_LINUX_WHEELS: dict[str, VendoredArchive] = {
    f"nvidia-{pkg}-linux": VendoredArchive(
        name=f"nvidia-{pkg}-linux",
        version=version,
        license="NVIDIA proprietary (redistributable runtime)",
        filename=filename,
        source=f"https://pypi.org/project/{pkg}",
        platform="linux",
    )
    for pkg, (version, filename) in {
        "cuda-nvrtc-cu12": ("12.6.77", "nvidia_cuda_nvrtc_cu12-12.6.77-py3-none-manylinux2014_x86_64.whl"),
        "cuda-runtime-cu12": ("12.6.77", "nvidia_cuda_runtime_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
        "cuda-cupti-cu12": ("12.6.80", "nvidia_cuda_cupti_cu12-12.6.80-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
        "cudnn-cu12": ("9.10.2.21", "nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl"),
        "cublas-cu12": ("12.6.4.1", "nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
        "cufft-cu12": ("11.3.0.4", "nvidia_cufft_cu12-11.3.0.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
        "curand-cu12": ("10.3.7.77", "nvidia_curand_cu12-10.3.7.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
        "cusolver-cu12": ("11.7.1.2", "nvidia_cusolver_cu12-11.7.1.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
        "cusparse-cu12": ("12.5.4.2", "nvidia_cusparse_cu12-12.5.4.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
        "cusparselt-cu12": ("0.7.1", "nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl"),
        "nccl-cu12": ("2.27.5", "nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
        "nvshmem-cu12": ("3.3.20", "nvidia_nvshmem_cu12-3.3.20-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
        "nvtx-cu12": ("12.6.77", "nvidia_nvtx_cu12-12.6.77-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
        "nvjitlink-cu12": ("12.6.85", "nvidia_nvjitlink_cu12-12.6.85-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl"),
        "cufile-cu12": ("1.11.1.6", "nvidia_cufile_cu12-1.11.1.6-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
    }.items()
}



def vendored_packages() -> list[str]:
    """Names of importable vendored packages whose source dir exists on disk."""
    return [name for name, path in _IMPORTABLE.items() if path.is_dir()]


def vendored_path(name: str) -> Optional[Path]:
    """Path of an importable vendored package, or None when not vendored/missing."""
    path = _IMPORTABLE.get(name)
    if path is not None and path.is_dir():
        return path
    return None


def is_vendored(name: str) -> bool:
    """True when the importable vendored source tree exists for ``name``."""
    return vendored_path(name) is not None


def source_archives() -> list[VendoredArchive]:
    """Metadata for every stored source archive (regardless of disk presence)."""
    return list(_ARCHIVES.values())


def archive_present(name: str) -> bool:
    """True when the archive tarball for ``name`` exists on disk."""
    archive = _ARCHIVES.get(name)
    return archive is not None and archive.path.is_file()


def archive_path(name: str) -> Optional[Path]:
    """Path of the stored source tarball for ``name``, or None."""
    archive = _ARCHIVES.get(name)
    if archive is not None and archive.path.is_file():
        return archive.path
    return None


def bundled_wheels() -> list[VendoredArchive]:
    """Metadata for every known bundled wheel (regardless of disk presence)."""
    return list(_BUNDLED_WHEELS.values()) + list(_NVIDIA_LINUX_WHEELS.values())


def _find_wheel(name: str) -> Optional[VendoredArchive]:
    """Registry entry for a wheel name across both wheel registries."""
    return _BUNDLED_WHEELS.get(name) or _NVIDIA_LINUX_WHEELS.get(name)


def wheel_present(name: str) -> bool:
    """True when the bundled wheel for ``name`` exists on disk."""
    wheel = _find_wheel(name)
    return wheel is not None and wheel.path.is_file()


def wheel_path(name: str) -> Optional[Path]:
    """Path of the bundled wheel for ``name``, or None when missing."""
    wheel = _find_wheel(name)
    if wheel is not None and wheel.path.is_file():
        return wheel.path
    return None


def framework_source(name: str) -> Optional[Path]:
    """Where the source for a framework lives in-repo: vendored package dir or
    archive tarball. Used for honest 'available locally as source' reporting."""
    pkg = vendored_path(name)
    if pkg is not None:
        return pkg
    return archive_path(name)


def framework_available(name: str) -> bool:
    """A framework is usable here when pip-installed OR importable-vendored.

    Compiled archives alone do NOT make a framework importable (they need a
    native build), so this only counts importable vendored packages — the
    same condition ``tools.common.module_available`` reports.
    """
    return is_vendored(name) or importlib.util.find_spec(name) is not None


def add_vendored_paths() -> None:
    """Idempotently append the vendored root to ``sys.path``.

    Append-only (never prepend) so pip-installed packages keep priority.
    Only the root is added — Python's namespace/regular package resolution
    then finds ``transformers``, ``langchain``, etc. inside it. Missing roots
    are simply skipped by the import machinery, so a stripped deployment
    without companion/vendor/ keeps working.
    """
    root = str(VENDOR_ROOT)
    if root not in sys.path:
        sys.path.append(root)


def describe_importable() -> list[dict]:
    """Structured description of importable vendored packages."""
    return [
        {
            "name": name,
            "kind": "importable",
            "path": str(path),
            "license_file": (path / "LICENSE").name if (path / "LICENSE").is_file() else "unknown",
        }
        for name, path in _IMPORTABLE.items()
        if path.is_dir()
    ]


def _describe_entries(
    entries: list[VendoredArchive], kind: str, present_fn: Callable[[str], bool]
) -> list[dict]:
    """Shared shape for archive/wheel inventory entries."""
    return [
        {
            "name": entry.name,
            "kind": kind,
            "version": entry.version,
            "license": entry.license,
            "filename": entry.filename,
            "present": present_fn(entry.name),
            "path": str(entry.path),
            "source": entry.source,
        }
        for entry in entries
    ]


def describe_archives() -> list[dict]:
    """Structured description of stored source archives."""
    return _describe_entries(list(_ARCHIVES.values()), "archive", archive_present)


def describe_wheels() -> list[dict]:
    """Structured description of bundled wheels (local-only, not committed)."""
    all_wheels = list(_BUNDLED_WHEELS.values()) + list(_NVIDIA_LINUX_WHEELS.values())
    entries = _describe_entries(all_wheels, "wheel", wheel_present)
    for entry in entries:
        # Normalize empty platform (source archive default) to "any".
        entry["platform"] = _find_wheel(entry["name"]).platform or "any"
    return entries


def describe_all() -> dict:
    """Full inventory for `/api/brain/vendor` or capability reporting."""
    return {
        "root": str(VENDOR_ROOT),
        "importable": describe_importable(),
        "archives": describe_archives(),
        "wheels": describe_wheels(),
    }


def inventory_summary() -> dict:
    """Compact boot-time summary of what is stored locally and what is missing.

    Returns counts for each tier plus the lists of missing wheel/platform
    entries so a deployment can log (or alert on) provisioning gaps at startup.
    """
    inventory = describe_all()
    importable = [p["name"] for p in inventory["importable"]]
    archives = {a["name"]: a["present"] for a in inventory["archives"]}
    wheels = inventory["wheels"]
    present_wheels = [w["name"] for w in wheels if w["present"]]
    missing_wheels = [w for w in wheels if not w["present"]]
    return {
        "root": inventory["root"],
        "importable_packages": sorted(importable),
        "source_archives_present": sum(1 for v in archives.values() if v),
        "source_archives_total": len(archives),
        "wheels_present": len(present_wheels),
        "wheels_total": len(wheels),
        "wheels_missing": [
            {"name": w["name"], "platform": w["platform"], "filename": w["filename"]}
            for w in missing_wheels
        ],
    }


__all__ = [
    "VENDOR_ROOT",
    "ARCHIVES_DIR",
    "VendoredArchive",
    "vendored_packages",
    "vendored_path",
    "is_vendored",
    "source_archives",
    "archive_present",
    "archive_path",
    "bundled_wheels",
    "wheel_present",
    "wheel_path",
    "framework_source",
    "framework_available",
    "add_vendored_paths",
    "describe_importable",
    "describe_archives",
    "describe_all",
    "inventory_summary",
]
