from __future__ import annotations

import pytest

from companion.config import BrainSettings

NO_ENV = {"_env_file": None}

# Every environment variable BrainSettings can read. Tests must not depend on
# the developer's shell / .env leaking in.
_ENV_KEYS = [
    "SUPABASE_URL",
    "SUPABASE_SERVICE_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "RELAI_MODEL",
    "OPENAI_MODEL",
    "ANTHROPIC_MODEL",
    "COMPANION_MODEL",
    "PROVIDER_ORDER",
    "EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "QDRANT_API_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "MEM0_API_KEY",
    "N8N_WEBHOOK_BASE",
    "COMPANION_PORT",
]


@pytest.fixture(autouse=True)
def _wipe_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults_load_without_env() -> None:
    settings = BrainSettings(**NO_ENV)
    assert settings.companion_model == "gemini-2.0-flash"
    assert settings.provider_order == ["gemini", "openai", "ollama", "anthropic"]
    assert settings.qdrant_configured is False
    assert settings.has_ai_provider is True  # ollama base url default
    assert settings.enable_relay_context is True
    assert settings.enable_corpus_search is True


def test_relay_knobs_parse_bool_env_values() -> None:
    assert BrainSettings(enable_relay_context="false", **NO_ENV).enable_relay_context is False
    assert BrainSettings(enable_corpus_search="0", **NO_ENV).enable_corpus_search is False


def test_provider_order_parses_comma_list() -> None:
    settings = BrainSettings(provider_order="openai,gemini", **NO_ENV)
    assert settings.provider_order == ["openai", "gemini"]


def test_supabase_key_resolves_either_env_name() -> None:
    assert BrainSettings(supabase_service_key="a", **NO_ENV).supabase_key == "a"
    assert BrainSettings(supabase_service_role_key="b", **NO_ENV).supabase_key == "b"
    assert BrainSettings(**NO_ENV).supabase_key == ""


def test_ai_provider_detection() -> None:
    assert BrainSettings(gemini_api_key="x", **NO_ENV).has_ai_provider is True
    assert BrainSettings(**NO_ENV).has_ai_provider is True  # ollama fallback always on
