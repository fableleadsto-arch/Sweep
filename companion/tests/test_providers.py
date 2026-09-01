"""Provider chain tests — normalized output from each provider."""

from __future__ import annotations

import asyncio

import httpx

from companion.config import BrainSettings
from companion.providers import GeminiProvider, ProviderChain


def _run(coro):
    return asyncio.run(coro)


def test_gemini_provider_parses_response(settings: BrainSettings) -> None:
    """Gemini must return a normalized result (regression: missing `text` kwarg)."""
    sb = settings.model_copy(update={"gemini_api_key": "test-key"})
    provider = GeminiProvider(sb)
    assert provider.available is True

    def handler(request: httpx.Request) -> httpx.Response:
        assert "generateContent" in str(request.url)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": '{"text": "pong", "tone": "warm"}'}]}}
                ],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 5},
            },
        )

    transport = httpx.MockTransport(handler)

    async def scenario():
        async with httpx.AsyncClient(transport=transport) as client:
            return await provider.generate(
                system="test",
                messages=[{"role": "user", "content": "hi"}],
                json_mode=True,
                client=client,
            )

    result = _run(scenario())
    assert result.provider == "gemini"
    assert result.text == '{"text": "pong", "tone": "warm"}'
    assert result.parsed == {"text": "pong", "tone": "warm"}
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 5


def test_gemini_uses_relai_model_override(settings: BrainSettings) -> None:
    """RELAI_MODEL must win over the companion_model default."""
    sb = settings.model_copy(update={"relai_model": "gemini-flash-latest"})
    assert sb.gemini_model == "gemini-flash-latest"
    sb2 = settings.model_copy(update={"relai_model": ""})
    assert sb2.gemini_model == sb2.companion_model


def test_chain_skips_unavailable_providers(settings: BrainSettings) -> None:
    """Chain reports an empty attempt list when nothing is configured."""
    chain = ProviderChain(settings)
    assert chain.health() == {
        "gemini": False,
        "openai": False,
        "ollama": True,  # default base url
        "anthropic": False,
    }
