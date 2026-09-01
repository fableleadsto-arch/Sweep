"""Relay AI model routing — port of ``src/RelAI/core/router.server.ts``.

Deterministic, zero-LLM decision layer that maps a classified task to the best
available model profile given priority, capability requirements and cost.
Availability is gated on env vars, so callers can inject an ``env`` mapping in
tests instead of touching the process environment.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional

RouterTaskType = Literal[
    "coding",
    "reasoning",
    "writing",
    "translation",
    "ocr",
    "image_generation",
    "voice",
    "vision",
    "research",
    "analysis",
    "chat",
    "extraction",
    "classification",
    "summarization",
    "planning",
    "general",
]

RouterPriority = Literal["latency", "quality", "cost", "balanced"]

ProviderType = Literal[
    "gemini",
    "ollama",
    "openai",
    "anthropic",
    "groq",
    "openrouter",
]

# Defaults mirrored from `src/RelAI/core/gemini.server.ts` — used whenever no
# provider/model wins or a non-Gemini pick must degrade to a Gemini id.
RELAY_DEFAULT_MODEL = "gemini-flash-latest"
RELAY_FALLBACK_MODEL = "gemini-flash-lite-latest"
RELAY_FALLBACK_PROVIDER: ProviderType = "gemini"


@dataclass(frozen=True)
class RouterContext:
    """Everything the router needs to pick a model for one request."""

    task_type: RouterTaskType = "general"
    priority: RouterPriority = "balanced"
    estimated_input_tokens: Optional[int] = None
    estimated_output_tokens: Optional[int] = None
    needs_tools: bool = False
    needs_vision: bool = False
    needs_json: bool = False
    needs_grounding: bool = False
    user_override: Optional[str] = None
    selected_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelProfile:
    """One routable model: strengths, capabilities and economics."""

    id: str
    name: str
    provider: ProviderType
    best_for: tuple[RouterTaskType, ...]
    max_context: int
    max_output: int
    cost_per_thousand_input: float
    cost_per_thousand_output: float
    typical_latency_ms: int
    supports_tools: bool
    supports_vision: bool
    supports_json: bool
    supports_grounding: bool
    quality: int
    available: bool = True
    required_env: Optional[str] = None


@dataclass(frozen=True)
class RouterDecision:
    """The routing result — which model, why, and what to fall back to."""

    model: str
    provider: ProviderType
    rationale: str
    estimated_cost_usd: float
    estimated_latency_ms: int
    used_override: bool
    fallback_model: str
    fallback_provider: ProviderType


# ─────────────────────────────────────────────────────────────
#  Model profiles
# ─────────────────────────────────────────────────────────────

# Capability fields are the *final* resolved values: the TS router lifts
# maxContext / vision / tools / json from `src/lib/ai/model-dataset.ts` via
# `knownCapabilities`, so the raw table below is overlayed with the same
# dataset values to reproduce identical routing decisions.
_RAW_PROFILES: list[dict] = [
    # ── Gemini ──────────────────────────────────────────────────
    {
        "id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "provider": "gemini",
        "best_for": ["chat", "general", "summarization", "classification", "extraction", "writing", "analysis"],
        "max_context": 1_000_000,
        "max_output": 8_000,
        "cost_per_thousand_input": 0.00015,
        "cost_per_thousand_output": 0.0006,
        "typical_latency_ms": 800,
        "supports_tools": True,
        "supports_vision": True,
        "supports_json": True,
        "supports_grounding": True,
        "quality": 70,
        "required_env": "GEMINI_API_KEY",
    },
    {
        "id": "gemini-2.5-pro",
        "name": "Gemini 2.5 Pro",
        "provider": "gemini",
        "best_for": ["reasoning", "coding", "research", "analysis", "planning", "vision"],
        "max_context": 1_000_000,
        "max_output": 8_000,
        "cost_per_thousand_input": 0.00125,
        "cost_per_thousand_output": 0.01,
        "typical_latency_ms": 2000,
        "supports_tools": True,
        "supports_vision": True,
        "supports_json": True,
        "supports_grounding": True,
        "quality": 92,
        "required_env": "GEMINI_API_KEY",
    },
    {
        "id": "gemini-flash-lite",
        "name": "Gemini Flash Lite",
        "provider": "gemini",
        "best_for": ["chat", "general", "classification", "summarization", "translation"],
        "max_context": 256_000,
        "max_output": 4_000,
        "cost_per_thousand_input": 0.000075,
        "cost_per_thousand_output": 0.0003,
        "typical_latency_ms": 400,
        "supports_tools": True,
        "supports_vision": False,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 55,
        "required_env": "GEMINI_API_KEY",
    },
    # ── Ollama (local) ──────────────────────────────────────────
    {
        "id": "ollama-llama3.1",
        "name": "Llama 3.1 8B (Local)",
        "provider": "ollama",
        "best_for": ["chat", "general", "classification", "extraction", "summarization", "writing"],
        "max_context": 128_000,
        "max_output": 4_000,
        "cost_per_thousand_input": 0.0,
        "cost_per_thousand_output": 0.0,
        "typical_latency_ms": 1500,
        "supports_tools": True,
        "supports_vision": False,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 55,
        "required_env": "OLLAMA_BASE_URL",
    },
    {
        "id": "ollama-llama3.3",
        "name": "Llama 3.3 70B (Local)",
        "provider": "ollama",
        "best_for": ["reasoning", "coding", "analysis", "planning", "research"],
        "max_context": 128_000,
        "max_output": 4_000,
        "cost_per_thousand_input": 0.0,
        "cost_per_thousand_output": 0.0,
        "typical_latency_ms": 4000,
        "supports_tools": True,
        "supports_vision": False,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 75,
        "required_env": "OLLAMA_BASE_URL",
    },
    {
        "id": "ollama-qwen2.5",
        "name": "Qwen 2.5 32B (Local)",
        "provider": "ollama",
        "best_for": ["coding", "analysis", "reasoning", "general"],
        "max_context": 32_000,
        "max_output": 4_000,
        "cost_per_thousand_input": 0.0,
        "cost_per_thousand_output": 0.0,
        "typical_latency_ms": 2500,
        "supports_tools": True,
        "supports_vision": False,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 68,
        "required_env": "OLLAMA_BASE_URL",
    },
    {
        "id": "ollama-mistral",
        "name": "Mistral 7B (Local)",
        "provider": "ollama",
        "best_for": ["chat", "general", "classification", "summarization"],
        "max_context": 32_000,
        "max_output": 4_000,
        "cost_per_thousand_input": 0.0,
        "cost_per_thousand_output": 0.0,
        "typical_latency_ms": 800,
        "supports_tools": True,
        "supports_vision": False,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 50,
        "required_env": "OLLAMA_BASE_URL",
    },
    # ── OpenAI ──────────────────────────────────────────────────
    {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "provider": "openai",
        "best_for": ["chat", "general", "summarization", "classification", "extraction", "writing"],
        "max_context": 128_000,
        "max_output": 16_000,
        "cost_per_thousand_input": 0.00015,
        "cost_per_thousand_output": 0.0006,
        "typical_latency_ms": 600,
        "supports_tools": True,
        "supports_vision": True,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 65,
        "required_env": "OPENAI_API_KEY",
    },
    {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "provider": "openai",
        "best_for": ["reasoning", "coding", "analysis", "planning", "vision", "research"],
        "max_context": 128_000,
        "max_output": 16_000,
        "cost_per_thousand_input": 0.0025,
        "cost_per_thousand_output": 0.01,
        "typical_latency_ms": 1200,
        "supports_tools": True,
        "supports_vision": True,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 88,
        "required_env": "OPENAI_API_KEY",
    },
    {
        "id": "o3-mini",
        "name": "o3 Mini",
        "provider": "openai",
        "best_for": ["reasoning", "coding", "analysis"],
        "max_context": 200_000,
        "max_output": 100_000,
        "cost_per_thousand_input": 0.0011,
        "cost_per_thousand_output": 0.0044,
        "typical_latency_ms": 3000,
        "supports_tools": True,
        "supports_vision": False,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 90,
        "required_env": "OPENAI_API_KEY",
    },
    # ── Anthropic ───────────────────────────────────────────────
    {
        "id": "claude-sonnet-4",
        "name": "Claude Sonnet 4",
        "provider": "anthropic",
        "best_for": ["reasoning", "coding", "analysis", "writing", "planning", "research"],
        "max_context": 200_000,
        "max_output": 8_000,
        "cost_per_thousand_input": 0.003,
        "cost_per_thousand_output": 0.015,
        "typical_latency_ms": 1500,
        "supports_tools": True,
        "supports_vision": True,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 90,
        "required_env": "ANTHROPIC_API_KEY",
    },
    {
        "id": "claude-haiku-3.5",
        "name": "Claude Haiku 3.5",
        "provider": "anthropic",
        "best_for": ["chat", "general", "classification", "extraction", "summarization"],
        "max_context": 200_000,
        "max_output": 8_000,
        "cost_per_thousand_input": 0.0008,
        "cost_per_thousand_output": 0.004,
        "typical_latency_ms": 600,
        "supports_tools": True,
        "supports_vision": True,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 65,
        "required_env": "ANTHROPIC_API_KEY",
    },
    # ── Groq ────────────────────────────────────────────────────
    {
        "id": "groq-llama-3.3-70b",
        "name": "Llama 3.3 70B (Groq)",
        "provider": "groq",
        "best_for": ["reasoning", "coding", "analysis", "general", "chat"],
        "max_context": 128_000,
        "max_output": 32_768,
        "cost_per_thousand_input": 0.00059,
        "cost_per_thousand_output": 0.00079,
        "typical_latency_ms": 800,
        "supports_tools": True,
        "supports_vision": False,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 82,
        "required_env": "GROQ_API_KEY",
    },
    {
        "id": "groq-llama-3.1-8b",
        "name": "Llama 3.1 8B (Groq)",
        "provider": "groq",
        "best_for": ["chat", "general", "classification", "summarization"],
        "max_context": 128_000,
        "max_output": 8_192,
        "cost_per_thousand_input": 0.00005,
        "cost_per_thousand_output": 0.00008,
        "typical_latency_ms": 300,
        "supports_tools": True,
        "supports_vision": False,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 55,
        "required_env": "GROQ_API_KEY",
    },
    {
        "id": "groq-mixtral-8x7b",
        "name": "Mixtral 8x7B (Groq)",
        "provider": "groq",
        "best_for": ["chat", "general", "writing", "analysis"],
        "max_context": 32_768,
        "max_output": 8_192,
        "cost_per_thousand_input": 0.00024,
        "cost_per_thousand_output": 0.00024,
        "typical_latency_ms": 500,
        "supports_tools": False,
        "supports_vision": False,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 60,
        "required_env": "GROQ_API_KEY",
    },
    {
        "id": "groq-deepseek-r1-70b",
        "name": "DeepSeek R1 Distill 70B (Groq)",
        "provider": "groq",
        "best_for": ["reasoning", "analysis", "coding", "research"],
        "max_context": 128_000,
        "max_output": 16_384,
        "cost_per_thousand_input": 0.00075,
        "cost_per_thousand_output": 0.00099,
        "typical_latency_ms": 1500,
        "supports_tools": False,
        "supports_vision": False,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 85,
        "required_env": "GROQ_API_KEY",
    },
    {
        "id": "groq-qwen-2.5-32b",
        "name": "Qwen 2.5 32B (Groq)",
        "provider": "groq",
        "best_for": ["coding", "analysis", "general", "chat"],
        "max_context": 128_000,
        "max_output": 8_192,
        "cost_per_thousand_input": 0.00020,
        "cost_per_thousand_output": 0.00020,
        "typical_latency_ms": 600,
        "supports_tools": True,
        "supports_vision": False,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 68,
        "required_env": "GROQ_API_KEY",
    },
    # ── OpenRouter ──────────────────────────────────────────────
    {
        "id": "openrouter-gpt-oss-20b",
        "name": "GPT-OSS 20B (OpenRouter, free)",
        "provider": "openrouter",
        "best_for": ["chat", "general", "writing", "classification", "summarization", "extraction"],
        "max_context": 131_072,
        "max_output": 16_384,
        "cost_per_thousand_input": 0,
        "cost_per_thousand_output": 0,
        "typical_latency_ms": 700,
        "supports_tools": True,
        "supports_vision": True,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 78,
        "required_env": "OPENROUTER_API_KEY",
    },
    {
        "id": "openrouter-gpt-4o",
        "name": "GPT-4o (OpenRouter)",
        "provider": "openrouter",
        "best_for": ["reasoning", "coding", "analysis", "research", "translation"],
        "max_context": 128_000,
        "max_output": 16_384,
        "cost_per_thousand_input": 0.0025,
        "cost_per_thousand_output": 0.010,
        "typical_latency_ms": 1400,
        "supports_tools": True,
        "supports_vision": True,
        "supports_json": True,
        "supports_grounding": False,
        "quality": 92,
        "required_env": "OPENROUTER_API_KEY",
    },
]

# Maps each routing profile id to the *real* model id the provider API speaks
# so capability baselines come from the same keys the TS dataset uses.
_PROFILE_MODEL_MAP: dict[str, str] = {
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-flash-lite": "gemini-2.5-flash-lite",
    "ollama-llama3.1": "llama3.1:8b",
    "ollama-llama3.3": "llama3.3:70b",
    "ollama-qwen2.5": "qwen2.5:7b",
    "ollama-mistral": "mistral:7b",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "o3-mini": "o3-mini",
    "claude-sonnet-4": "claude-sonnet-4",
    "claude-haiku-3.5": "claude-haiku-3.5",
    "groq-llama-3.3-70b": "llama-3.3-70b-versatile",
    "groq-llama-3.1-8b": "llama-3.1-8b-instant",
    "groq-mixtral-8x7b": "mixtral-8x7b-32768",
    "groq-deepseek-r1-70b": "deepseek-r1-distill-llama-70b",
    "groq-qwen-2.5-32b": "qwen-2.5-32b",
    "openrouter-gpt-oss-20b": "openai/gpt-oss-20b:free",
    "openrouter-gpt-4o": "openai/gpt-4o",
}

# Capability overlays lifted from `src/lib/ai/model-dataset.ts` (the final
# resolved values the TS router computes at module load).
_CAPABILITY_OVERLAY: dict[str, dict] = {
    "gemini-2.5-flash-lite": {"max_context": 256_000, "supports_vision": True, "supports_tools": True, "supports_json": True},
    "llama3.1:8b": {"max_context": 131_072, "supports_tools": True, "supports_json": True},
    "llama3.3:70b": {"max_context": 131_072, "supports_tools": True, "supports_json": True},
    "qwen2.5:7b": {"max_context": 131_072, "supports_tools": True, "supports_json": True},
    "mistral:7b": {"max_context": 131_072, "supports_tools": True, "supports_json": True},
    "llama-3.3-70b-versatile": {"supports_tools": True, "supports_json": True},
    "llama-3.1-8b-instant": {"supports_tools": True, "supports_json": True},
    "mixtral-8x7b-32768": {"supports_tools": False, "supports_json": True},
    "deepseek-r1-distill-llama-70b": {"supports_tools": False, "supports_json": True},
    "qwen-2.5-32b": {"supports_tools": True, "supports_json": True},
    "openai/gpt-oss-20b:free": {"supports_tools": True, "supports_json": True},
    "openai/gpt-4o": {"supports_tools": True, "supports_json": True},
}


def _build_profiles() -> list[ModelProfile]:
    resolved: list[ModelProfile] = []
    for raw in _RAW_PROFILES:
        dataset_key = _PROFILE_MODEL_MAP.get(raw["id"], raw["id"])
        caps = _CAPABILITY_OVERLAY.get(dataset_key, {})
        fields = dict(raw)
        fields["best_for"] = tuple(raw["best_for"])
        for field_name in ("max_context", "supports_tools", "supports_vision", "supports_json"):
            if field_name in caps:
                fields[field_name] = caps[field_name]
        resolved.append(ModelProfile(**fields))
    return resolved


MODEL_PROFILES: tuple[ModelProfile, ...] = tuple(_build_profiles())


# ─────────────────────────────────────────────────────────────
#  Task classifier
# ─────────────────────────────────────────────────────────────

_TOOL_CODING = re.compile(r"code|review|refactor|debug|test|commit", re.IGNORECASE)
_TOOL_RESEARCH = re.compile(r"search|research|osint|crawl", re.IGNORECASE)
_TOOL_WRITING = re.compile(r"draft|message|email|reply", re.IGNORECASE)
_TOOL_ANALYSIS = re.compile(r"analyze|extract|classify", re.IGNORECASE)

_KEYWORD_CODING = re.compile(r"\b(code|function|bug|fix|debug|refactor|test|api|endpoint|component|implement|build)\b", re.IGNORECASE)
_KEYWORD_REASONING = re.compile(r"\b(think|reason|why|how does|explain|compare|analyze|evaluate|solve|prove)\b", re.IGNORECASE)
_KEYWORD_WRITING = re.compile(r"\b(draft|write|rewrite|edit|compose|proofread|grammar|tone)\b", re.IGNORECASE)
_KEYWORD_TRANSLATION = re.compile(r"\b(translate|translation)\b", re.IGNORECASE)
_KEYWORD_OCR = re.compile(r"\b(ocr|extract text|read image|scan document)\b", re.IGNORECASE)
_KEYWORD_IMAGE = re.compile(r"\b(image|generate|create|design|illustration|icon|photo)\b", re.IGNORECASE)
_KEYWORD_VOICE = re.compile(r"\b(voice|speak|speech|transcribe|listen|audio)\b", re.IGNORECASE)
_KEYWORD_VISION = re.compile(r"\b(see|look|what'?s in this image|analyze (image|photo|screenshot|picture))\b", re.IGNORECASE)
_KEYWORD_RESEARCH = re.compile(r"\b(research|find|search|investigate|look up|gather|report|sources|citation)\b", re.IGNORECASE)
_KEYWORD_SUMMARIZATION = re.compile(r"\b(summarize|summarise|tl;dr|key points|brief|overview)\b", re.IGNORECASE)
_KEYWORD_CLASSIFICATION = re.compile(r"\b(classify|categorize|label|tag|sort|type of)\b", re.IGNORECASE)
_KEYWORD_PLANNING = re.compile(r"\b(plan|schedule|workflow|automation|strategy|roadmap)\b", re.IGNORECASE)
_KEYWORD_ANALYSIS = re.compile(r"\b(analyze|analysis|insight|trend|pattern|metric)\b", re.IGNORECASE)


def classify_task(
    message: str,
    tools_requested: Optional[list[str]] = None,
) -> RouterTaskType:
    """Classify a request into a router task type (heuristic, no LLM)."""
    lower = message.lower()

    if tools_requested:
        if any(_TOOL_CODING.search(t) for t in tools_requested):
            return "coding"
        if any(_TOOL_RESEARCH.search(t) for t in tools_requested):
            return "research"
        if any(_TOOL_WRITING.search(t) for t in tools_requested):
            return "writing"
        if any(_TOOL_ANALYSIS.search(t) for t in tools_requested):
            return "analysis"

    if _KEYWORD_CODING.search(lower):
        return "coding"
    if _KEYWORD_REASONING.search(lower):
        return "reasoning"
    if _KEYWORD_WRITING.search(lower):
        return "writing"
    if _KEYWORD_TRANSLATION.search(lower):
        return "translation"
    if _KEYWORD_OCR.search(lower):
        return "ocr"
    if _KEYWORD_IMAGE.search(lower):
        return "image_generation"
    if _KEYWORD_VOICE.search(lower):
        return "voice"
    if _KEYWORD_VISION.search(lower):
        return "vision"
    if _KEYWORD_RESEARCH.search(lower):
        return "research"
    if _KEYWORD_SUMMARIZATION.search(lower):
        return "summarization"
    if _KEYWORD_CLASSIFICATION.search(lower):
        return "classification"
    if _KEYWORD_PLANNING.search(lower):
        return "planning"
    if _KEYWORD_ANALYSIS.search(lower):
        return "analysis"

    return "general"


# ─────────────────────────────────────────────────────────────
#  Router
# ─────────────────────────────────────────────────────────────


def env_available(env_var: Optional[str], env: Mapping[str, str]) -> bool:
    """True when the provider's required env var is set (or none required)."""
    if not env_var:
        return True
    return bool(env.get(env_var))


def get_available_models(env: Optional[Mapping[str, str]] = None) -> list[ModelProfile]:
    """Profiles whose required env vars are configured in this deployment."""
    env = env if env is not None else os.environ
    return [m for m in MODEL_PROFILES if m.available and env_available(m.required_env, env)]


def select_model(
    context: RouterContext,
    env: Optional[Mapping[str, str]] = None,
) -> RouterDecision:
    """Pick the best model for a task context. User overrides always win."""
    available = get_available_models(env)

    if context.user_override:
        for profile in available:
            if profile.id == context.user_override:
                return RouterDecision(
                    model=profile.id,
                    provider=profile.provider,
                    rationale=f"User explicitly requested {profile.name} ({profile.provider}).",
                    estimated_cost_usd=estimate_call_cost(profile, context),
                    estimated_latency_ms=profile.typical_latency_ms,
                    used_override=True,
                    fallback_model=find_fallback(available, profile),
                    fallback_provider=find_fallback_provider(available, profile),
                )
        return RouterDecision(
            model=context.user_override,
            provider="gemini",
            rationale=f"User override: {context.user_override}.",
            estimated_cost_usd=0,
            estimated_latency_ms=1000,
            used_override=True,
            fallback_model=RELAY_FALLBACK_MODEL,
            fallback_provider=RELAY_FALLBACK_PROVIDER,
        )

    candidates = [
        m
        for m in available
        if (not context.needs_tools or m.supports_tools)
        and (not context.needs_vision or m.supports_vision)
        and (not context.needs_json or m.supports_json)
        and (not context.needs_grounding or m.supports_grounding)
        and (not context.estimated_input_tokens or context.estimated_input_tokens <= m.max_context)
    ]

    if not candidates:
        if not available:
            return RouterDecision(
                model=RELAY_DEFAULT_MODEL,
                provider="gemini",
                rationale="No optimal model found; using default.",
                estimated_cost_usd=0,
                estimated_latency_ms=1000,
                used_override=False,
                fallback_model=RELAY_FALLBACK_MODEL,
                fallback_provider=RELAY_FALLBACK_PROVIDER,
            )
        best = max(available, key=lambda m: m.quality)
        return RouterDecision(
            model=best.id,
            provider=best.provider,
            rationale=f"Best available model given constraints: {best.name}.",
            estimated_cost_usd=estimate_call_cost(best, context),
            estimated_latency_ms=best.typical_latency_ms,
            used_override=False,
            fallback_model=find_fallback(available, best),
            fallback_provider=find_fallback_provider(available, best),
        )

    scored = sorted(
        ((_score_profile(m, context), m) for m in candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )
    winner = scored[0][1]

    return RouterDecision(
        model=winner.id,
        provider=winner.provider,
        rationale=(
            f"Selected {winner.name} for {context.task_type} task "
            f"(priority: {context.priority}): quality={winner.quality}, "
            f"~{winner.typical_latency_ms}ms, ${winner.cost_per_thousand_input}/1K input."
        ),
        estimated_cost_usd=estimate_call_cost(winner, context),
        estimated_latency_ms=winner.typical_latency_ms,
        used_override=False,
        fallback_model=scored[1][1].id if len(scored) > 1 else RELAY_FALLBACK_MODEL,
        fallback_provider=scored[1][1].provider if len(scored) > 1 else RELAY_FALLBACK_PROVIDER,
    )


def _score_profile(profile: ModelProfile, context: RouterContext) -> float:
    """Priority-weighted score: quality, latency and cost trade-offs."""
    score = 0.0

    if context.priority == "latency":
        score += (2000 - profile.typical_latency_ms) / 20
        score += profile.quality * 0.3
        score -= estimate_call_cost(profile, context) * 1000
    elif context.priority == "quality":
        score += profile.quality * 1.5
        score -= estimate_call_cost(profile, context) * 500
    elif context.priority == "cost":
        score += (10 - estimate_call_cost(profile, context)) * 100
        score += profile.quality * 0.5
    else:  # balanced
        score += profile.quality
        score += (2000 - profile.typical_latency_ms) / 10
        score -= estimate_call_cost(profile, context) * 200

    if profile.provider == "ollama":
        score += 5  # small bonus for free/local inference

    if context.task_type in profile.best_for:
        score += 15

    return score


def estimate_call_cost(profile: ModelProfile, context: RouterContext) -> float:
    """Rough USD cost for this call from input/output token estimates."""
    input_tokens = context.estimated_input_tokens if context.estimated_input_tokens is not None else 500
    output_tokens = context.estimated_output_tokens if context.estimated_output_tokens is not None else 500
    return (
        profile.cost_per_thousand_input * (input_tokens / 1000)
        + profile.cost_per_thousand_output * (output_tokens / 1000)
    )


def find_fallback(available: list[ModelProfile], primary: ModelProfile) -> str:
    """First available profile (not the primary) with comparable quality."""
    for profile in available:
        if profile.id != primary.id and profile.quality >= primary.quality * 0.7:
            return profile.id
    return RELAY_FALLBACK_MODEL


def find_fallback_provider(available: list[ModelProfile], primary: ModelProfile) -> ProviderType:
    for profile in available:
        if profile.id != primary.id and profile.quality >= primary.quality * 0.7:
            return profile.provider
    return RELAY_FALLBACK_PROVIDER
