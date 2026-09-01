"""RAG service tests — graceful degradation when the knowledge base is down."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from companion.config import BrainSettings
from companion.rag import RagService
from companion.schemas import RagResult


@pytest.fixture()
def sb_settings(settings: BrainSettings) -> BrainSettings:
    """Settings with Supabase configured but a key that will be rejected."""
    return settings.model_copy(
        update={
            "supabase_url": "https://example.supabase.co",
            "supabase_service_role_key": "invalid-key",
        }
    )


def _run(coro) -> RagResult:
    return asyncio.run(coro)


def test_retrieve_degrades_on_unauthorized(sb_settings: BrainSettings) -> None:
    """A 401 from Supabase must yield an empty result, not raise."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/rest/v1/knowledge_chunks")
        return httpx.Response(401, json={"message": "Invalid API key"})

    transport = httpx.MockTransport(handler)

    async def scenario() -> RagResult:
        async with httpx.AsyncClient(transport=transport) as client:
            service = RagService(sb_settings)
            return await service.retrieve("w1", "hello world", client=client)

    result = _run(scenario())
    assert result.found is False
    assert result.context == ""
    assert result.sources == []


def test_retrieve_degrades_on_server_error(sb_settings: BrainSettings) -> None:
    """A 500 from Supabase must yield an empty result, not raise."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(handler)

    async def scenario() -> RagResult:
        async with httpx.AsyncClient(transport=transport) as client:
            service = RagService(sb_settings)
            return await service.retrieve("w1", "hello world", client=client)

    result = _run(scenario())
    assert result.found is False
    assert result.context == ""


def test_retrieve_degrades_on_network_error(sb_settings: BrainSettings) -> None:
    """A dropped connection must yield an empty result, not raise."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(handler)

    async def scenario() -> RagResult:
        async with httpx.AsyncClient(transport=transport) as client:
            service = RagService(sb_settings)
            return await service.retrieve("w1", "hello world", client=client)

    result = _run(scenario())
    assert result.found is False
    assert result.context == ""


def test_corpus_degrades_without_embeddings(sb_settings: BrainSettings) -> None:
    """No embedding provider keys → corpus search yields an empty result."""

    async def scenario() -> RagResult:
        service = RagService(sb_settings)
        return await service.corpus("react hooks")

    result = _run(scenario())
    assert result.found is False
    assert result.context == ""


def test_corpus_respects_disable_knob(sb_settings: BrainSettings) -> None:
    sb_settings.enable_corpus_search = False

    async def scenario() -> RagResult:
        service = RagService(sb_settings)
        return await service.corpus("react hooks")

    result = _run(scenario())
    assert result.found is False
    assert result.context == ""


def test_corpus_formats_rows(sb_settings: BrainSettings) -> None:
    """A valid RPC response is formatted with source citations."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/rest/v1/rpc/search_brain_corpus"):
            return httpx.Response(
                200,
                json=[
                    {
                        "content": "Hooks let you use state without classes.",
                        "chunk_index": 3,
                        "similarity": 0.81,
                        "name": "React Hooks",
                        "source": "react",
                        "source_url": "https://react.dev/reference/react",
                    }
                ],
            )
        raise httpx.ConnectError("unexpected call")

    async def embed_stub(query: str, settings) -> list[float]:
        return [0.1] * 768

    import companion.rag as rag_mod

    original = rag_mod.embed_one
    rag_mod.embed_one = embed_stub
    transport = httpx.MockTransport(handler)
    try:
        async def scenario() -> RagResult:
            service = RagService(sb_settings)
            async with httpx.AsyncClient(transport=transport) as client:
                return await service.corpus("react hooks", client=client)

        result = _run(scenario())
    finally:
        rag_mod.embed_one = original

    assert result.found is True
    assert "React Hooks" in result.context
    assert "react.dev" in result.context
    assert result.sources[0].name == "React Hooks"
    assert result.sources[0].type == "react"
