"""Inference-engine capabilities — vLLM, Ollama, llama.cpp, ONNX Runtime,
Hugging Face TGI, TensorRT-LLM, Triton, TensorRT and ggml.

Every framework here is a heavy/optional dependency: each tool lazy-imports
its framework only when invoked, and GPU-only engines report their hardware
requirement honestly instead of crashing. Local model servers (Ollama, TGI,
langflow) are reached over their public HTTP APIs so no extra SDK is needed.
"""

from __future__ import annotations

from typing import Any

from .common import CapabilityUnavailable, has_cuda, is_safe_http_url, load

# ── vLLM ──────────────────────────────────────────────────────────────────


def run_vllm(payload: dict[str, Any]) -> dict[str, Any]:
    """High-throughput LLM serving with vLLM (needs a GPU)."""
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "generate").lower()

    vllm = load("vllm")
    if not has_cuda():
        return _gpu_required("vllm")

    if mode == "health":
        return {
            "result": {"engine": "vllm", "version": getattr(vllm, "__version__", "unknown")},
            "summary": "vLLM installed and GPU is available.",
            "libraries_used": ["vllm"],
        }

    from vllm import LLM, SamplingParams

    model = str(params.get("model") or "facebook/opt-125m")
    prompt = str(params.get("prompt") or payload.get("data") or params.get("text") or "")
    if not prompt:
        raise ValueError("vLLM generation needs `params.prompt` (or `data`).")
    llm = LLM(model=model, gpu_memory_utilization=0.5, max_model_len=1024)
    sampling = SamplingParams(
        max_tokens=int(params.get("max_tokens") or 128),
        temperature=float(params.get("temperature") or 0.8),
        top_p=float(params.get("top_p") or 0.95),
    )
    outputs = llm.generate([prompt[:2000]], sampling)
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    return {
        "result": {"engine": "vllm", "model": model, "generated_text": text, "prompt_tokens": len(prompt.split())},
        "summary": f"vLLM ({model}) generated {len(text)} chars.",
        "libraries_used": ["vllm"],
    }


# ── Ollama (local model server, HTTP API) ────────────────────────────────


def run_ollama(payload: dict[str, Any]) -> dict[str, Any]:
    """Talk to a local Ollama server through its HTTP API."""
    import httpx

    params = payload.get("params") or {}
    settings = payload.get("_settings")
    mode = str(params.get("mode") or "generate").lower()

    base_url = ""
    if settings and getattr(settings, "ollama_base_url", ""):
        base_url = settings.ollama_base_url
    base_url = base_url or "http://localhost:11434"
    base_url = base_url.rstrip("/")

    def _post(path: str, body: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(f"{base_url}{path}", json=body)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:  # noqa: BLE001 - surface server errors cleanly
            raise CapabilityUnavailable(
                f"Ollama server at {base_url} unreachable: {exc}. Start `ollama serve` or set OLLAMA_BASE_URL."
            ) from exc

    if mode == "list":
        models = _post("/api/tags", {}) or {}
        names = [m.get("name") for m in models.get("models", [])]
        return {
            "result": {"engine": "ollama", "models": names, "base_url": base_url},
            "summary": f"Ollama has {len(names)} model(s) loaded locally.",
            "libraries_used": ["ollama"],
        }

    if mode == "embed":
        from .common import as_text

        text = as_text(payload.get("data"), params)
        model = str(params.get("model") or settings and getattr(settings, "ollama_embed_model", "") or "nomic-embed-text")
        if not text:
            raise ValueError("Ollama embedding needs `data` text.")
        out = _post("/api/embeddings", {"model": model, "prompt": text[:4000]})
        emb = out.get("embedding") or []
        return {
            "result": {"engine": "ollama", "model": model, "dimensions": len(emb), "embedding_preview": emb[:8]},
            "summary": f"Ollama embedded the text with {model} ({len(emb)} dims).",
            "libraries_used": ["ollama"],
        }

    model = str(params.get("model") or settings and getattr(settings, "ollama_model", "") or "llama3.1:8b")
    prompt = str(params.get("prompt") or payload.get("data") or params.get("text") or "")
    if not prompt:
        raise ValueError("Ollama generation needs `params.prompt` (or `data`).")
    out = _post(
        "/api/generate",
        {
            "model": model,
            "prompt": prompt[:4000],
            "stream": False,
            "options": {
                "temperature": float(params.get("temperature") or 0.8),
                "num_predict": int(params.get("max_tokens") or 256),
            },
        },
    )
    return {
        "result": {"engine": "ollama", "model": model, "generated_text": out.get("response", ""), "eval_count": out.get("eval_count", 0)},
        "summary": f"Ollama {model} generated a response ({out.get('eval_count', 0)} tokens).",
        "libraries_used": ["ollama"],
    }


# ── llama.cpp (GGUF models via llama-cpp-python) ─────────────────────────


def run_llamacpp(payload: dict[str, Any]) -> dict[str, Any]:
    """Run GGUF models locally with llama-cpp-python (CPU/GPU)."""
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "generate").lower()

    llama_cpp = load("llama_cpp")

    if mode == "health":
        return {
            "result": {"engine": "llama.cpp", "backend": "llama-cpp-python", "has_gpu": has_cuda()},
            "summary": "llama.cpp bindings installed.",
            "libraries_used": ["llama_cpp"],
        }

    model_path = str(params.get("model_path") or params.get("model") or "")
    if not model_path:
        raise ValueError("llama.cpp needs `params.model_path` to a GGUF file.")
    prompt = str(params.get("prompt") or payload.get("data") or "")
    if not prompt:
        raise ValueError("llama.cpp generation needs `params.prompt` (or `data`).")

    Llama = getattr(llama_cpp, "Llama", None)
    if Llama is None:
        raise CapabilityUnavailable("llama-cpp-python exposes no `Llama` class.")
    llm = Llama(model_path=model_path, n_ctx=int(params.get("n_ctx") or 2048), n_gpu_layers=int(params.get("n_gpu_layers") or 0))
    out = llm.create_completion(
        prompt[:4000],
        max_tokens=int(params.get("max_tokens") or 128),
        temperature=float(params.get("temperature") or 0.8),
        echo=False,
    )
    text = (out.get("choices") or [{}])[0].get("text", "")
    return {
        "result": {"engine": "llama.cpp", "model_path": model_path, "generated_text": text},
        "summary": f"llama.cpp generated {len(text)} chars from {model_path}.",
        "libraries_used": ["llama_cpp"],
    }


# ── ONNX Runtime ─────────────────────────────────────────────────────────


def run_onnx(payload: dict[str, Any]) -> dict[str, Any]:
    """Cross-platform ONNX inference with onnxruntime."""
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "providers").lower()

    ort = load("onnxruntime")

    if mode == "providers":
        return {
            "result": {
                "engine": "onnxruntime",
                "version": getattr(ort, "__version__", "unknown"),
                "providers": ort.get_available_providers(),
                "default_provider": getattr(ort, "get_default_provider", lambda: "CPUExecutionProvider")(),
            },
            "summary": "onnxruntime is installed; providers listed.",
            "libraries_used": ["onnxruntime"],
        }

    if mode == "run":
        model_path = str(params.get("model_path") or params.get("model") or "")
        if not model_path:
            raise ValueError("ONNX `run` needs `params.model_path` to a .onnx file.")
        provider = str(params.get("provider") or "CPUExecutionProvider")
        try:
            session = ort.InferenceSession(model_path, providers=[provider])
        except Exception as exc:  # noqa: BLE001
            raise CapabilityUnavailable(f"ONNX session failed to load {model_path}: {exc}") from exc
        inputs = payload.get("data")
        if inputs is None:
            # Describe the graph instead of inventing inputs.
            meta = [
                {"name": i.name, "shape": list(i.shape) if i.shape else None, "type": i.type}
                for i in session.get_inputs()
            ]
            outputs = [o.name for o in session.get_outputs()]
            return {
                "result": {"engine": "onnxruntime", "model_path": model_path, "inputs": meta, "outputs": outputs},
                "summary": f"ONNX model {model_path} loaded; graph has {len(meta)} input(s).",
                "libraries_used": ["onnxruntime"],
            }
        try:
            result = session.run(None, inputs)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"ONNX inference failed: {exc}") from exc
        return {
            "result": {"engine": "onnxruntime", "model_path": model_path, "outputs": [o for o in result]},
            "summary": "ONNX inference completed.",
            "libraries_used": ["onnxruntime"],
        }

    raise ValueError("onnx mode must be 'providers' or 'run'.")


# ── Hugging Face Text Generation Inference (HTTP) ────────────────────────


def run_tgi(payload: dict[str, Any]) -> dict[str, Any]:
    """Call a Hugging Face TGI server through its HTTP API."""
    import httpx

    settings = payload.get("_settings")
    params = payload.get("params") or {}
    base_url = str(
        params.get("base_url")
        or (settings and getattr(settings, "tgi_base_url", ""))
        or ""
    ).rstrip("/")
    if not base_url:
        raise ValueError("TGI needs `params.base_url` (or a TGI_BASE_URL setting) pointing at a public TGI server.")
    if not is_safe_http_url(base_url):
        raise ValueError("TGI base_url must be a public http(s) endpoint (SSRF guard).")
    prompt = str(params.get("prompt") or payload.get("data") or "")
    if not prompt:
        raise ValueError("TGI needs `params.prompt` (or `data`).")
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{base_url}/generate",
                json={
                    "inputs": prompt[:4000],
                    "parameters": {
                        "max_new_tokens": int(params.get("max_tokens") or 256),
                        "temperature": float(params.get("temperature") or 0.8),
                    },
                },
            )
            resp.raise_for_status()
            body = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise CapabilityUnavailable(f"TGI server at {base_url} unreachable: {exc}") from exc
    return {
        "result": {"engine": "tgi", "base_url": base_url, "generated_text": body.get("generated_text", "")},
        "summary": f"TGI server responded with {len(body.get('generated_text', ''))} chars.",
        "libraries_used": ["tgi"],
    }


# ── TensorRT / TensorRT-LLM ──────────────────────────────────────────────


def run_tensorrt(payload: dict[str, Any]) -> dict[str, Any]:
    """NVIDIA TensorRT — version + device introspection (GPU only)."""
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "health").lower()

    try:
        import tensorrt as trt
    except Exception as exc:  # noqa: BLE001
        raise CapabilityUnavailable(f"TensorRT import failed: {exc}. It requires the NVIDIA TensorRT wheel.") from exc

    if not has_cuda():
        return _gpu_required("tensorrt")

    version = f"{trt.__version__}" if hasattr(trt, "__version__") else "unknown"
    if mode == "health":
        return {
            "result": {"engine": "tensorrt", "version": version, "builder": getattr(trt, "Builder", None) is not None},
            "summary": f"TensorRT {version} is available on this GPU host.",
            "libraries_used": ["tensorrt"],
        }
    raise ValueError("tensorrt mode must be 'health' (engine building needs an ONNX model + GPU workspace).")


def run_tensorrt_llm(payload: dict[str, Any]) -> dict[str, Any]:
    """NVIDIA TensorRT-LLM — availability + GPU check."""
    params = payload.get("params") or {}
    try:
        import tensorrt_llm  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise CapabilityUnavailable(f"TensorRT-LLM import failed: {exc}. It is GPU-only and needs its wheel.") from exc
    if not has_cuda():
        return _gpu_required("tensorrt-llm")
    return {
        "result": {"engine": "tensorrt-llm", "available": True, "has_gpu": True},
        "summary": "TensorRT-LLM installed with a CUDA GPU present.",
        "libraries_used": ["tensorrt_llm"],
    }


# ── Triton (GPU kernel language) ─────────────────────────────────────────


def run_triton(payload: dict[str, Any]) -> dict[str, Any]:
    """Triton — JIT-compile a tiny GPU kernel and run it."""
    params = payload.get("params") or {}
    mode = str(params.get("mode") or "probe").lower()

    try:
        import triton
    except Exception as exc:  # noqa: BLE001
        raise CapabilityUnavailable(f"Triton import failed: {exc}. It needs a CUDA toolkit.") from exc

    if not has_cuda():
        return _gpu_required("triton")

    if mode == "probe":
        return {
            "result": {"engine": "triton", "version": getattr(triton, "__version__", "unknown")},
            "summary": "Triton installed and GPU available.",
            "libraries_used": ["triton"],
        }

    if mode == "compile":
        import torch

        @triton.jit
        def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: triton.language.constexpr):
            pid = triton.language.program_id(0)
            offs = pid * BLOCK + triton.language.arange(0, BLOCK)
            mask = offs < n
            x = triton.language.load(x_ptr + offs, mask=mask)
            y = triton.language.load(y_ptr + offs, mask=mask)
            triton.language.store(out_ptr + offs, x + y, mask=mask)

        n = int(params.get("n") or 1024)
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        out = torch.empty_like(x)
        grid = ((n + 255) // 256,)
        _add_kernel[grid](x, y, out, n, BLOCK=256)
        torch.cuda.synchronize()
        max_err = float((out - (x + y)).abs().max())
        return {
            "result": {"engine": "triton", "n": n, "max_error": max_err, "kernel": "add_kernel"},
            "summary": f"Triton JIT kernel compiled and ran over {n} elements (max err {max_err:.2e}).",
            "libraries_used": ["triton"],
        }

    raise ValueError("triton mode must be 'probe' or 'compile'.")


# ── ggml (backend — surfaces through llama.cpp bindings) ─────────────────


def run_ggml(payload: dict[str, Any]) -> dict[str, Any]:
    """ggml backend — availability probe (usually consumed via llama.cpp)."""
    params = payload.get("params") or {}
    try:
        import ggml  # noqa: F401
    except Exception:  # noqa: BLE001 - python bindings are rare
        ggml = None
    llama_cpp_ok = False
    try:
        import llama_cpp  # noqa: F401

        llama_cpp_ok = True
    except Exception:  # noqa: BLE001
        pass
    return {
        "result": {
            "engine": "ggml",
            "python_ggml_bindings": ggml is not None,
            "via_llama_cpp": llama_cpp_ok,
            "note": "ggml is a C backend; the practical Python path is llama-cpp-python (capability `llamacpp`).",
        },
        "summary": "ggml backend probed.",
        "libraries_used": [],
    }


# ── shared helpers ───────────────────────────────────────────────────────


def _gpu_required(engine: str) -> dict[str, Any]:
    return {
        "result": {"engine": engine, "requires_gpu": True, "ready": False},
        "summary": f"{engine} needs a CUDA GPU. None detected on this host — the capability is installed but not runnable here.",
        "libraries_used": [],
    }
