"""Search provider routing — decides which provider satisfies a search request.

Routing rules (checked in order):
  - technical research  → Exa when configured, else keyless
  - news                → news-capable provider, else keyless
  - deep                → multiple providers, merged + deduped
  - default             → keyless (multi-engine HTML), always available
"""

from __future__ import annotations

from typing import Optional

from ..core.types import SearchOptions, SearchRunResult
from .engine import relai_search


def route_priority(intent: str) -> list[str]:
    """Deterministic provider priority per intent."""
    if intent == "technical":
        return ["exa", "tavily", "keyless"]
    elif intent == "news":
        return ["tavily", "exa", "keyless"]
    elif intent == "deep":
        return ["exa", "tavily", "searxng", "jina", "keyless"]
    elif intent == "platform":
        return ["keyless", "exa", "tavily"]
    else:  # "general"
        return ["tavily", "exa", "keyless"]


async def route_search(
    query: str,
    intent: str = "general",
    options: Optional[SearchOptions] = None,
) -> SearchRunResult:
    """Single-provider search with routing and fallback."""
    opts = options or SearchOptions()
    errors: list[str] = []

    try:
        result = await relai_search(
            query,
            limit=opts.limit,
            site=opts.site,
            page=opts.page,
            aggregate=opts.aggregate,
            mode=opts.time_range,
        )
        hits = result.get("hits", [])
        return SearchRunResult(
            provider=result.get("engine", "unknown"),
            query=query,
            results=[
                {
                    "url": h["url"],
                    "title": h["title"],
                    "snippet": h.get("snippet", ""),
                    "provider": h.get("engine", "unknown"),
                    "access_mode": "public",
                }
                for h in hits
            ],
            blocked=result.get("tried", 0) == 0,
            note=result.get("errors", [None])[0] if result.get("errors") else None,
            errors=result.get("errors", []),
        )
    except Exception as e:
        errors.append(str(e))

    return SearchRunResult(
        provider="none",
        query=query,
        results=[],
        blocked=True,
        note="No provider returned results",
        errors=errors,
    )


async def route_platform_search(
    query: str,
    platform: str,
    limit: int = 5,
) -> SearchRunResult:
    """Route search to a specific platform adapter."""
    from ..platforms import get_adapter_by_platform
    adapter = get_adapter_by_platform(platform)
    if not adapter:
        return SearchRunResult(
            provider="none", query=query, results=[], blocked=True,
            note=f"No adapter for platform: {platform}",
        )
    hits, note, access = await adapter.search(query, limit=limit)
    return SearchRunResult(
        provider=f"platform_{platform}",
        query=query,
        results=[h.model_dump() for h in hits],
        blocked=access.value == "unavailable",
        note=note,
    )


async def run_deep_search(
    query: str,
    options: Optional[SearchOptions] = None,
) -> dict:
    """Merge providers for deep searches; dedupes by canonical URL."""
    opts = options or SearchOptions(limit=8)
    seen: set[str] = set()
    merged: list[dict] = []
    errors: list[str] = []
    used: list[str] = []
    dropped = 0

    try:
        result = await relai_search(
            query,
            limit=opts.limit,
            aggregate=True,  # Fan out across all providers
        )
        for hit in result.get("hits", []):
            key = hit["url"].split("#")[0]
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            merged.append(hit)

        return {
            "provider": result.get("engine", "multiple"),
            "query": query,
            "results": merged[:opts.limit],
            "providers_used": used,
            "deduped": dropped,
            "errors": result.get("errors", []),
        }
    except Exception as e:
        return {
            "provider": "none",
            "query": query,
            "results": [],
            "providers_used": used,
            "deduped": dropped,
            "errors": [str(e)],
        }
