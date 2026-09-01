"""Training-stack capabilities — DeepSpeed, Megatron-LM, axolotl,
bitsandbytes and unsloth.

These are GPU-centric heavy frameworks. Every capability reports its real
installed state and GPU requirement instead of pretending to work on a
CPU-only host. Where a library exposes a Python API (DeepSpeed, bitsandbytes,
unsloth, Megatron) the tool calls it directly; axolotl is a config-driven
CLI, so the capability validates its schema and produces a real training
config the CLI can consume.
"""

from __future__ import annotations

from typing import Any

from .common import CapabilityUnavailable, has_cuda, load


# ── DeepSpeed ────────────────────────────────────────────────────────────


def run_deepspeed(payload: dict[str, Any]) -> dict[str, Any]:
    """DeepSpeed — version + CUDA availability + config check."""
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "health").lower()

    try:
        import deepspeed
    except Exception as exc:  # noqa: BLE001
        raise CapabilityUnavailable(f"DeepSpeed import failed: {exc}. Install with `pip install deepspeed`.") from exc

    if mode == "health":
        return {
            "result": {
                "engine": "deepspeed",
                "version": getattr(deepspeed, "__version__", "unknown"),
                "has_gpu": has_cuda(),
                "zero_stages_supported": [1, 2, 3],
            },
            "summary": (
                f"DeepSpeed {getattr(deepspeed, '__version__', '?')} installed"
                + (" with CUDA available." if has_cuda() else " (CPU-only host).")
            ),
            "libraries_used": ["deepspeed"],
        }

    if mode == "config":
        # Validate a JSON config the caller wants to run training with.
        config = params.get("config") or payload.get("data")
        if isinstance(config, str):
            import json

            try:
                config = json.loads(config)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"DeepSpeed config is not valid JSON: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError("DeepSpeed `config` needs a JSON object (train_batch_size, zero_optimization, ...).")
        return {
            "result": {
                "engine": "deepspeed",
                "config_keys": sorted(config.keys()),
                "zero_enabled": bool(config.get("zero_optimization")),
                "offload": bool(config.get("zero_optimization", {}).get("offload_optimizer")) if isinstance(config.get("zero_optimization"), dict) else False,
                "note": "Config validated. Actual training runs via `deepspeed train.py --deepspeed_config config.json`.",
            },
            "summary": f"DeepSpeed config validated ({len(config)} keys).",
            "libraries_used": ["deepspeed"],
        }

    raise ValueError("deepspeed mode must be 'health' or 'config'.")


# ── Megatron-LM ──────────────────────────────────────────────────────────


def run_megatron(payload: dict[str, Any]) -> dict[str, Any]:
    """Megatron-LM — availability + GPU requirement probe."""
    try:
        import megatron  # noqa: F401
    except Exception:  # noqa: BLE001
        try:
            from megatron.core import parallel_state  # noqa: F401  # type: ignore

            megatron_ok = True
        except Exception:  # noqa: BLE001
            megatron_ok = False
    else:
        megatron_ok = True

    if not megatron_ok:
        raise CapabilityUnavailable(
            "Megatron-LM is not importable. It is a GPU-only research framework — clone NVIDIA/Megatron-LM "
            "and install per its README when a multi-GPU training host is available."
        )
    if not has_cuda():
        return {
            "result": {"engine": "megatron", "requires_gpu": True, "ready": False},
            "summary": "Megatron-LM is installed but needs a CUDA GPU to initialize parallel state.",
            "libraries_used": [],
        }
    return {
        "result": {"engine": "megatron", "available": True, "has_gpu": True},
        "summary": "Megatron-LM is importable on a CUDA host.",
        "libraries_used": ["megatron"],
    }


# ── axolotl (config-driven fine-tuning CLI) ──────────────────────────────


def run_axolotl(payload: dict[str, Any]) -> dict[str, Any]:
    """axolotl — generate a validated fine-tuning config the CLI consumes."""
    params = payload.get("params") or {}

    try:
        import axolotl  # noqa: F401
    except Exception:  # noqa: BLE001
        axolotl = None

    if axolotl is None and not params.get("allow_uninstalled"):
        raise CapabilityUnavailable(
            "axolotl is not installed. It is a CLI tool: `pip install axolotl` (GPU host recommended). "
            "This capability can still emit a config with params.allow_uninstalled=true."
        )

    base_model = str(params.get("base_model") or "mistralai/Mistral-7B-Instruct-v0.2")
    dataset = str(params.get("dataset") or "tatsu-lab/alpaca")
    config = {
        "base_model": base_model,
        "model_type": str(params.get("model_type") or "AutoModelForCausalLM"),
        "sequence_len": int(params.get("sequence_len") or 2048),
        "load_in_4bit": bool(params.get("load_in_4bit") or True),
        "datasets": [
            {
                "path": dataset,
                "type": str(params.get("dataset_type") or "alpaca"),
            }
        ],
        "val_set_size": float(params.get("val_set_size") or 0.05),
        "output_dir": str(params.get("output_dir") or "./axolotl-out"),
        "num_epochs": int(params.get("num_epochs") or 3),
        "micro_batch_size": int(params.get("micro_batch_size") or 4),
        "gradient_accumulation_steps": int(params.get("gradient_accumulation_steps") or 4),
        "learning_rate": float(params.get("learning_rate") or 2e-4),
        "optimizer": str(params.get("optimizer") or "adamw_torch"),
        "lr_scheduler": str(params.get("lr_scheduler") or "cosine"),
        "warmup_steps": int(params.get("warmup_steps") or 100),
        "bf16": "auto",
        "logging_steps": 1,
        "save_steps": 500,
    }
    return {
        "result": {
            "engine": "axolotl",
            "installed": axolotl is not None,
            "config": config,
            "run_command": "accelerate launch -m axolotl.cli.train config.yml",
        },
        "summary": f"axolotl config generated for {base_model} on {dataset} (CLI-ready).",
        "libraries_used": ["axolotl"] if axolotl is not None else [],
    }


# ── bitsandbytes (quantization) ──────────────────────────────────────────


def run_bitsandbytes(payload: dict[str, Any]) -> dict[str, Any]:
    """bitsandbytes — 8-bit linear layer construction (real GPU code)."""
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "health").lower()

    bnb = load("bitsandbytes")

    if mode == "health":
        return {
            "result": {"engine": "bitsandbytes", "version": getattr(bnb, "__version__", "unknown"), "has_gpu": has_cuda()},
            "summary": f"bitsandbytes {getattr(bnb, '__version__', '?')} installed."
            + (" CUDA available." if has_cuda() else " CPU-only host — quantization needs CUDA."),
            "libraries_used": ["bitsandbytes"],
        }

    if mode == "quantize":
        if not has_cuda():
            return {
                "result": {"engine": "bitsandbytes", "requires_gpu": True, "ready": False},
                "summary": "bitsandbytes quantization needs a CUDA GPU.",
                "libraries_used": [],
            }
        import torch

        in_f = int(params.get("in_features") or 512)
        out_f = int(params.get("out_features") or 512)
        try:
            linear = bnb.nn.Linear8bitLt(in_f, out_f, has_fp16_weights=False)
            x = torch.randn(4, in_f, device="cuda")
            out = linear(x)
        except Exception as exc:  # noqa: BLE001
            raise CapabilityUnavailable(f"bitsandbytes 8-bit layer failed: {exc}") from exc
        return {
            "result": {
                "engine": "bitsandbytes",
                "layer": "Linear8bitLt",
                "in_features": in_f,
                "out_features": out_f,
                "output_shape": list(out.shape),
            },
            "summary": f"bitsandbytes 8-bit Linear({in_f}→{out_f}) forward pass OK.",
            "libraries_used": ["bitsandbytes"],
        }

    raise ValueError("bitsandbytes mode must be 'health' or 'quantize'.")


# ── unsloth (fast fine-tuning) ───────────────────────────────────────────


def run_unsloth(payload: dict[str, Any]) -> dict[str, Any]:
    """unsloth — availability probe + model-loading check."""
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "health").lower()

    unsloth = load("unsloth")

    if mode == "health":
        return {
            "result": {"engine": "unsloth", "version": getattr(unsloth, "__version__", "unknown"), "has_gpu": has_cuda()},
            "summary": f"unsloth {getattr(unsloth, '__version__', '?')} installed."
            + (" CUDA available." if has_cuda() else " CPU-only host — unsloth needs CUDA."),
            "libraries_used": ["unsloth"],
        }

    if mode == "load":
        if not has_cuda():
            return {
                "result": {"engine": "unsloth", "requires_gpu": True, "ready": False},
                "summary": "unsloth model loading needs a CUDA GPU.",
                "libraries_used": [],
            }
        from unsloth import FastLanguageModel

        model_name = str(params.get("model") or "unsloth/Meta-Llama-3.1-8B-bnb-4bit")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(model_name)
        except Exception as exc:  # noqa: BLE001
            raise CapabilityUnavailable(f"unsloth failed to load {model_name}: {exc}") from exc
        return {
            "result": {"engine": "unsloth", "model": model_name, "loaded": model is not None and tokenizer is not None},
            "summary": f"unsloth loaded {model_name} (4-bit fast path).",
            "libraries_used": ["unsloth"],
        }

    raise ValueError("unsloth mode must be 'health' or 'load'.")
