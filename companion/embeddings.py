"""Embedding pipeline — mirrors `src/RelAI/knowledge/embed.server.ts`.

Tries Gemini's `gemini-embedding-001` first, then falls back to OpenAI's
`text-embedding-3-small`. A failed text yields `None` so callers can degrade
to keyword search exactly like the TypeScript stack does.
"""

from __future__ import annotations

from typing import Optional

import httpx

from .config import BrainSettings

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:embedContent"
)
OPENAI_ENDPOINT = "https://api.openai.com/v1/embeddings"
MAX_TEXT_CHARS = 6000


async def embed_batch(
    texts: list[str],
    settings: BrainSettings,
    client: Optional[httpx.AsyncClient] = None,
) -> list[Optional[list[float]]]:
    """Embed a batch of texts, returning `None` for failures."""
    if not texts:
        return []

    own_client = client is None
    client = client or httpx.AsyncClient(timeout=60.0)
    try:
        if settings.gemini_api_key:
            results = await _embed_via_gemini(texts, settings, client)
            if any(results):
                return results

        if settings.openai_api_key:
            results = await _embed_via_openai(texts, settings, client)
            if any(results):
                return results
    finally:
        if own_client:
            await client.aclose()

    return [None] * len(texts)


async def embed_one(text: str, settings: BrainSettings) -> Optional[list[float]]:
    """Embed a single text, returning `None` on failure."""
    results = await embed_batch([text], settings)
    return results[0]


async def _embed_via_gemini(
    texts: list[str],
    settings: BrainSettings,
    client: httpx.AsyncClient,
) -> list[Optional[list[float]]]:
    results: list[Optional[list[float]]] = []
    for text in texts:
        try:
            resp = await client.post(
                GEMINI_ENDPOINT.format(model=settings.embedding_model),
                params={"key": settings.gemini_api_key},
                json={
                    "content": {"parts": [{"text": text[:MAX_TEXT_CHARS]}]},
                    "outputDimensionality": settings.embedding_dimensions,
                },
            )
            if not resp.is_success:
                results.append(None)
                continue
            data = resp.json()
            values = data.get("embedding", {}).get("values")
            results.append(values if isinstance(values, list) else None)
        except httpx.HTTPError:
            results.append(None)
    return results


async def _embed_via_openai(
    texts: list[str],
    settings: BrainSettings,
    client: httpx.AsyncClient,
) -> list[Optional[list[float]]]:
    results: list[Optional[list[float]]] = []
    batch_size = settings.embedding_batch_size
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            resp = await client.post(
                OPENAI_ENDPOINT,
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": settings.openai_embedding_model,
                    "input": [t[:MAX_TEXT_CHARS] for t in batch],
                    "dimensions": settings.embedding_dimensions,
                },
            )
            if not resp.is_success:
                results.extend([None] * len(batch))
                continue
            data = resp.json()
            items = {item["index"]: item["embedding"] for item in data.get("data", [])}
            results.extend(items.get(i) for i in range(len(batch)))
        except httpx.HTTPError:
            results.extend([None] * len(batch))
    return results
