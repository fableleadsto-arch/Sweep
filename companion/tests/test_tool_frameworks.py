"""Tests for the expanded AI-framework capability tool modules.

These frameworks are optional/GPU-heavy: the tests verify the *availability
honesty* contract (never pretend a missing/CPU-only framework works), the
pure paths that do not need a GPU or live server, and that the catalog still
builds without importing any heavy framework.
"""

from __future__ import annotations

from companion.capabilities import CAPABILITIES, list_capabilities
from companion.tools.common import module_available


def test_catalog_lists_expanded_frameworks() -> None:
    ids = {cap.id for cap in CAPABILITIES}
    for expected in (
        "vllm",
        "ollama",
        "llamacpp",
        "onnx",
        "tgi",
        "tensorrt",
        "tensorrt-llm",
        "triton",
        "ggml",
        "diffusion",
        "timm",
        "accelerate",
        "deepspeed",
        "megatron",
        "axolotl",
        "bitsandbytes",
        "unsloth",
        "langchain",
        "crewai",
        "autogen",
        "langflow",
        "qdrant",
        "milvus",
    ):
        assert expected in ids, f"{expected} is missing from the capability catalog"


def test_expanded_capabilities_do_not_import_heavy_frameworks() -> None:
    import sys

    # Some heavy modules may already be in sys.modules because an earlier test
    # in the same session exercised them. Snapshot the baseline so we only
    # fail when *building the catalog* introduces a NEW heavy import.
    baseline = set(sys.modules)
    list_capabilities()
    newly_imported = set(sys.modules) - baseline

    for heavy in (
        "torch",
        "tensorflow",
        "jax",
        "transformers",
        "diffusers",
        "timm",
        "vllm",
        "deepspeed",
        "bitsandbytes",
        "unsloth",
        "langchain",
        "crewai",
        "autogen",
        "pymilvus",
        "llama_cpp",
        "onnxruntime",
        "triton",
        "tensorrt",
    ):
        # qdrant_client is a core dependency used by the memory layer, so it is
        # legitimately loaded at package import time — excluded from this check.
        assert heavy not in newly_imported, f"building the catalog imported {heavy}"


def test_gpu_frameworks_report_missing_gpu_honestly() -> None:
    from companion.tools import inference, training

    # No-GPU paths must return a structured requires_gpu result, not raise.
    result = inference._gpu_required("vllm")
    assert result["result"]["requires_gpu"] is True

    # DeepSpeed health works without GPU (it only reports).
    out = training.run_deepspeed({"params": {"mode": "health"}}) if module_available("deepspeed") else None
    if out is not None:
        assert out["result"]["engine"] == "deepspeed"


def test_onnx_providers_is_import_optional() -> None:
    # When onnxruntime is absent the capability must raise CapabilityUnavailable,
    # never a bare ImportError.
    if module_available("onnxruntime"):
        from companion.tools.inference import run_onnx

        out = run_onnx({"params": {"mode": "providers"}})
        assert out["result"]["engine"] == "onnxruntime"
        assert isinstance(out["result"]["providers"], list)
    else:
        from companion.tools.common import CapabilityUnavailable
        from companion.tools.inference import run_onnx

        try:
            run_onnx({"params": {"mode": "providers"}})
        except CapabilityUnavailable:
            pass
        else:  # pragma: no cover - should never happen
            raise AssertionError("onnxruntime missing but run_onnx did not raise")


def test_ollama_requires_server_or_raises_cleanly() -> None:
    # Without a running server the HTTP capability raises CapabilityUnavailable
    # with a helpful message (never a raw httpx traceback).
    from companion.tools.common import CapabilityUnavailable
    from companion.tools.inference import run_ollama

    try:
        run_ollama({"params": {"mode": "list", "base_url": "http://127.0.0.1:1"}})
    except CapabilityUnavailable as exc:
        assert "unreachable" in str(exc) or "serve" in str(exc)
    else:  # pragma: no cover - a server on port 1 would be surprising
        pass


def test_axolotl_emits_config_without_install() -> None:
    # The config-emission path works even when the CLI is not installed, as
    # long as allow_uninstalled is set.
    from companion.tools.training import run_axolotl

    out = run_axolotl({"params": {"base_model": "mistralai/Mistral-7B-Instruct-v0.2", "allow_uninstalled": True}})
    assert out["result"]["config"]["base_model"] == "mistralai/Mistral-7B-Instruct-v0.2"
    assert "axolotl.cli.train" in out["result"]["run_command"]


def test_catalog_endpoint_includes_expanded_ids() -> None:
    catalog = list_capabilities()
    ids = {c.id for c in catalog}
    for expected in ("vllm", "diffusion", "langchain", "qdrant", "milvus", "deepspeed"):
        assert expected in ids
