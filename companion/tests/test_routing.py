"""Tests for the Python port of the model router (routing.py)."""

from __future__ import annotations

from companion.routing import (
    RELAY_DEFAULT_MODEL,
    RELAY_FALLBACK_MODEL,
    MODEL_PROFILES,
    RouterContext,
    classify_task,
    get_available_models,
    select_model,
)


# ── task classifier ──────────────────────────────────────────


def test_classify_task_tool_signals() -> None:
    assert classify_task("anything", tools_requested=["analyze:code"]) == "coding"
    assert classify_task("anything", tools_requested=["review:code"]) == "coding"
    assert classify_task("anything", tools_requested=["search:web"]) == "research"
    assert classify_task("anything", tools_requested=["osint:sweep"]) == "research"
    assert classify_task("anything", tools_requested=["draft:message"]) == "writing"
    assert classify_task("anything", tools_requested=["analyze:intent"]) == "analysis"
    assert classify_task("anything", tools_requested=["extract:text"]) == "analysis"


def test_classify_task_no_tool_signals_uses_keywords() -> None:
    assert classify_task("write an email to the team") == "writing"
    assert classify_task("summarize this document") == "summarization"
    assert classify_task("translate this to spanish") == "translation"
    assert classify_task("compare iphone vs android") == "reasoning"
    assert classify_task("debug the login bug") == "coding"
    assert classify_task("automate my workflow") == "planning"
    assert classify_task("analyze this trend") == "reasoning"


def test_classify_task_defaults_to_general() -> None:
    assert classify_task("hello there friend") == "general"
    assert classify_task("") == "general"


# ── availability / env gating ────────────────────────────────


def test_get_available_models_respects_env() -> None:
    all_models = get_available_models(env={})
    assert all_models == []
    gemini_only = get_available_models(env={"GEMINI_API_KEY": "x"})
    assert {m.provider for m in gemini_only} == {"gemini"}
    assert {m.id for m in gemini_only} == {"gemini-2.5-flash", "gemini-2.5-pro", "gemini-flash-lite"}


def test_dataset_overlay_applied() -> None:
    by_id = {m.id: m for m in MODEL_PROFILES}
    assert by_id["ollama-llama3.1"].max_context == 131_072
    assert by_id["gemini-flash-lite"].supports_vision is True
    assert by_id["groq-deepseek-r1-70b"].supports_tools is False
    assert by_id["groq-mixtral-8x7b"].supports_tools is False


# ── select_model ─────────────────────────────────────────────


def test_select_model_no_providers_degrades_to_default() -> None:
    decision = select_model(RouterContext(task_type="chat"), env={})
    assert decision.model == RELAY_DEFAULT_MODEL
    assert decision.provider == "gemini"
    assert decision.used_override is False


def test_select_model_quality_prefers_pro() -> None:
    decision = select_model(
        RouterContext(task_type="reasoning", priority="quality"),
        env={"GEMINI_API_KEY": "x"},
    )
    assert decision.model == "gemini-2.5-pro"
    assert decision.provider == "gemini"
    assert decision.fallback_model == "gemini-2.5-flash"


def test_select_model_latency_prefers_flash_lite() -> None:
    decision = select_model(
        RouterContext(task_type="chat", priority="latency"),
        env={"GEMINI_API_KEY": "x"},
    )
    assert decision.model == "gemini-flash-lite"


def test_select_model_grounding_filters_to_gemini() -> None:
    decision = select_model(
        RouterContext(task_type="research", priority="quality", needs_grounding=True),
        env={"GEMINI_API_KEY": "x", "OPENAI_API_KEY": "x"},
    )
    assert decision.provider == "gemini"


def test_select_model_capability_filters_drop_no_tool_models() -> None:
    decision = select_model(
        RouterContext(task_type="coding", priority="quality", needs_tools=True),
        env={"GROQ_API_KEY": "x"},
    )
    assert decision.provider == "groq"
    assert decision.model in {"groq-llama-3.3-70b", "groq-qwen-2.5-32b"}


def test_select_model_user_override_wins() -> None:
    decision = select_model(
        RouterContext(task_type="chat", user_override="gpt-4o"),
        env={"GEMINI_API_KEY": "x", "OPENAI_API_KEY": "x"},
    )
    assert decision.model == "gpt-4o"
    assert decision.provider == "openai"
    assert decision.used_override is True


def test_select_model_unknown_override_passes_through() -> None:
    decision = select_model(
        RouterContext(task_type="chat", user_override="my-custom-model"),
        env={},
    )
    assert decision.model == "my-custom-model"
    assert decision.used_override is True
    assert decision.fallback_model == RELAY_FALLBACK_MODEL
