"""
Environment Detection — Records exact hardware, OS, and software configuration.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def detect_hardware() -> dict[str, Any]:
    """Detect CPU, GPU, RAM, and storage."""
    hw: dict[str, Any] = {}

    hw["cpu_count"] = os.cpu_count()
    hw["cpu_name"] = platform.processor() or "UNKNOWN"
    try:
        import psutil
        hw["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
        hw["ram_available_gb"] = round(psutil.virtual_memory().available / (1024**3), 1)
        hw["disk_free_gb"] = round(psutil.disk_usage("/").free / (1024**3), 1)
    except ImportError:
        hw["ram_gb"] = "psutil not installed"
        hw["ram_available_gb"] = "psutil not installed"
        hw["disk_free_gb"] = "psutil not installed"

    hw["gpu"] = "NONE DETECTED"
    hw["gpu_model"] = "NONE"
    hw["gpu_vram_gb"] = 0
    try:
        import torch
        if torch.cuda.is_available():
            hw["gpu"] = "CUDA"
            hw["gpu_model"] = torch.cuda.get_device_name(0)
            hw["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
            hw["cuda_version"] = torch.version.cuda or "N/A"
            hw["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
    except Exception:
        pass

    return hw


def detect_os() -> dict[str, Any]:
    """Detect operating system details."""
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python_implementation": platform.python_implementation(),
    }


def detect_software() -> dict[str, Any]:
    """Detect all relevant software versions."""
    sw: dict[str, Any] = {"python": sys.version}

    packages = [
        "torch", "tensorflow", "numpy", "scipy", "scikit-learn",
        "transformers", "sentence_transformers", "onnxruntime",
        "cv2", "spacy", "whisper",
    ]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            sw[pkg] = getattr(mod, "__version__", getattr(mod, "version", "installed (version unknown)"))
        except ImportError:
            sw[pkg] = "NOT INSTALLED"

    return sw


def detect_sweep_config() -> dict[str, Any]:
    """Record Sweep neural mesh configuration."""
    cfg: dict[str, Any] = {}

    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        cfg["git_commit"] = result.stdout.strip()
    except Exception:
        cfg["git_commit"] = "UNKNOWN"

    try:
        from sweep_neural_mesh.neurons.mesh import NeuralMesh
        mesh = NeuralMesh()
        cfg["num_cores"] = len(mesh.cores)
        cfg["total_neurons"] = sum(len(c.neurons) for c in mesh.cores)
        cfg["mesh_topology"] = "scalable_neural_mesh"
    except Exception as e:
        cfg["mesh_init_error"] = str(e)

    cfg["sweep_version"] = "0.1.0"
    cfg["neural_mesh_version"] = "9-stage-biological"
    cfg["precision"] = "float32"
    cfg["quantization"] = "none"
    cfg["max_reasoning_steps"] = 100
    cfg["random_seed"] = 42

    return cfg


def generate_environment_json(output_dir: Path) -> Path:
    """Generate complete environment.json."""
    env = {
        "hardware": detect_hardware(),
        "os": detect_os(),
        "software": detect_software(),
        "sweep_config": detect_sweep_config(),
    }
    out = output_dir / "environment.json"
    out.write_text(json.dumps(env, indent=2, default=str))
    return out
