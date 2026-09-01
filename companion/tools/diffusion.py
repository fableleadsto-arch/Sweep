"""Diffusion + vision-model capabilities — Hugging Face diffusers,
pytorch-image-models (timm) and accelerate.

All heavy/optional: lazy imports, honest GPU reporting. Diffusion runs real
`StableDiffusionPipeline` code; timm runs real `timm.create_model` inference;
accelerate reports device placement via its real API.
"""

from __future__ import annotations

import base64
import io
from typing import Any

from .common import CapabilityUnavailable, has_cuda, load


# ── diffusers (text-to-image) ────────────────────────────────────────────


def run_diffusion(payload: dict[str, Any]) -> dict[str, Any]:
    """Text-to-image generation with Hugging Face diffusers."""
    params = payload.get("params") or {}

    diffusers = load("diffusers")
    if not has_cuda():
        return {
            "result": {
                "engine": "diffusers",
                "requires_gpu": True,
                "ready": False,
                "note": "diffusers is installed but text-to-image needs a CUDA GPU (or set params.device='cpu' with a tiny model).",
            },
            "summary": "diffusers is installed; no GPU detected for image generation.",
            "libraries_used": [],
        }

    prompt = str(params.get("prompt") or payload.get("data") or params.get("text") or "")
    if not prompt:
        raise ValueError("Diffusion needs `params.prompt` (or `data`) describing the image.")

    from diffusers import StableDiffusionPipeline
    import torch

    model = str(params.get("model") or "runwayml/stable-diffusion-v1-5")
    device = str(params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        pipe = pipe.to(device)
        image = pipe(
            prompt[:2000],
            num_inference_steps=int(params.get("steps") or 25),
            guidance_scale=float(params.get("guidance") or 7.5),
            seed=int(params.get("seed") or 0),
            generator=torch.Generator(device).manual_seed(int(params.get("seed") or 0)),
        ).images[0]
    except Exception as exc:  # noqa: BLE001 - model download / OOM surface cleanly
        raise CapabilityUnavailable(f"Diffusion generation failed: {exc}") from exc

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "result": {
            "engine": "diffusers",
            "model": model,
            "prompt": prompt,
            "width": image.width,
            "height": image.height,
            "png_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
        },
        "summary": f"diffusers generated a {image.width}×{image.height} image from your prompt.",
        "libraries_used": ["diffusers"],
    }


# ── timm (pytorch-image-models) ──────────────────────────────────────────


def run_timm(payload: dict[str, Any]) -> dict[str, Any]:
    """Vision-model catalogue + inference with timm (pytorch-image-models)."""
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "list").lower()

    timm = load("timm")

    if mode == "list":
        models = timm.list_models(str(params.get("filter") or "*"), pretrained=bool(params.get("pretrained")))
        return {
            "result": {"engine": "timm", "count": len(models), "models": models[:50], "filter": params.get("filter") or "*"},
            "summary": f"timm knows {len(models)} model architecture(s) matching the filter.",
            "libraries_used": ["timm"],
        }

    if mode == "inference":
        if not has_cuda():
            return {
                "result": {"engine": "timm", "requires_gpu": True, "ready": False},
                "summary": "timm inference needs a CUDA GPU for a real forward pass.",
                "libraries_used": [],
            }
        import torch

        model_name = str(params.get("model") or "resnet18")
        try:
            model = timm.create_model(model_name, pretrained=bool(params.get("pretrained")), num_classes=0)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"timm could not create '{model_name}': {exc}") from exc
        model = model.eval().cuda()
        with torch.no_grad():
            batch = torch.randn(1, 3, 224, 224, device="cuda")
            features = model(batch)
        params_count = sum(p.numel() for p in model.parameters())
        return {
        "result": {
            "engine": "timm",
            "model": model_name,
            "output_dim": int(features.shape[1]),
            "parameters": params_count,
            "feature_preview": [round(float(v), 4) for v in features[0, :8]],
        },
        "summary": f"timm {model_name} forward pass OK ({params_count:,} params, {features.shape[1]} features).",
        "libraries_used": ["timm"],
    }

    raise ValueError("timm mode must be 'list' or 'inference'.")


# ── accelerate (model loading / device placement) ────────────────────────


def run_accelerate(payload: dict[str, Any]) -> dict[str, Any]:
    """Device placement + state inspection with Hugging Face accelerate."""
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "state").lower()

    accelerate = load("accelerate")
    import torch

    if mode == "state":
        return {
            "result": {
                "engine": "accelerate",
                "version": getattr(accelerate, "__version__", "unknown"),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "device_map_supported": True,
            },
            "summary": "accelerate is installed; device placement reported.",
            "libraries_used": ["accelerate"],
        }

    if mode == "offload":
        model_name = str(params.get("model") or "distilgpt2")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from accelerate import dispatch_model, infer_auto_device_map

        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        except Exception as exc:  # noqa: BLE001
            raise CapabilityUnavailable(f"accelerate could not load '{model_name}': {exc}") from exc
        device_map = infer_auto_device_map(model, max_memory={0: "2GiB"})
        dispatched = dispatch_model(model, device_map=device_map)
        return {
            "result": {
                "engine": "accelerate",
                "model": model_name,
                "device_map": {str(k): str(v) for k, v in device_map.items()},
                "dispatched": dispatched is not None,
            },
            "summary": f"accelerate dispatched {model_name} across the available devices.",
            "libraries_used": ["accelerate"],
        }

    raise ValueError("accelerate mode must be 'state' or 'offload'.")
