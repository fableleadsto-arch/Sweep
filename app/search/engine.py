"""Multi-engine web search with HTML parsers and API provider fallback.

Provider list (in default routing order):
  keyless  — Multi-engine HTML search (always available, no key needed)
  tavily   — when TAVILY_API_KEY is set
  exa      — when EXA_API_KEY is set
  searxng  — when SEARXNG_BASE_URL is set
  jina     — free-tier reader/search (always available)
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx

from ..core.cache import cache_get, cache_set
from ..core.http import relai_fetch
from ..core.types import SearchRun


# ── Engine Parsers ────────────────────────────────────────────────────


def _first_match(text: str, patterns: list[re.Pattern]) -> str:
    """Try all patterns and return the first non-empty match group."""
    for pat in patterns:
        m = pat.search(text)
        if m and m.group(1):
            return re.sub(r"<[^>]+>", "", m.group(1)).strip()
    return ""


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def _decode_redirect(href: str) -> str:
    """Decode DuckDuckGo redirect URLs."""
    if "duckduckgo.com/l/" in href:
        try:
            u = urlparse(href if href.startswith("http") else f"https:{href}")
            qs = parse_qs(u.query)
            return qs.get("uddg", [href])[0]
        except Exception:
            pass
    if href.startswith("/url?q="):
        try:
            return unquote(href[7].split("&")[0])
        except Exception:
            pass
    return href


def _parse_ddg(html: str) -> list[dict]:
    """DuckDuckGo HTML parser with multiple fallback strategies."""
    results = []
    seen = set()

    # Strategy 1: Result blocks
    blocks = re.findall(
        r'<div[^>]*class="[^"]*result[^"]*results_links[^"]*"[^>]*>([\s\S]*?)</div>\s*</div>',
        html,
        re.IGNORECASE,
    )
    if len(blocks) < 2:
        blocks = re.findall(
            r'<div[^>]*class="[^"]*result\b[^"]*"[^>]*>([\s\S]*?)(?:</div>\s*){2,3}',
            html,
            re.IGNORECASE,
        )

    for block in blocks:
        url_m = re.search(r'<a[^>]+href="(https?://[^"]+)"', block, re.IGNORECASE)
        if not url_m:
            continue
        url = _decode_redirect(url_m.group(1))
        title = _first_match(
            block,
            [
                re.compile(r'class="[^"]*result__a[^"]*"[^>]*>([\s\S]*?)</a>', re.I),
                re.compile(r'<a[^>]+href="https?://[^"]+"[^>]*>([\s\S]*?)</a>', re.I),
            ],
        )
        snippet = _first_match(
            block,
            [
                re.compile(r'class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)</', re.I),
                re.compile(r'class="[^"]*snippet[^"]*"[^>]*>([\s\S]*?)</', re.I),
            ],
        )
        if url.startswith("http") and title and url not in seen:
            seen.add(url)
            results.append({"url": url, "title": _strip_html(title), "snippet": _strip_html(snippet), "engine": "duckduckgo"})

    if results:
        return results

    # Strategy 2: Anchor tags with result classes
    for m in re.finditer(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)</a>', html, re.IGNORECASE):
        url = _decode_redirect(m.group(1))
        title = _strip_html(m.group(2))
        if url.startswith("http") and title and url not in seen:
            seen.add(url)
            results.append({"url": url, "title": title, "snippet": "", "engine": "duckduckgo"})
            if len(results) >= 40:
                break

    return results


def _parse_ddg_lite(html: str) -> list[dict]:
    """DuckDuckGo Lite parser."""
    results = []
    for row in re.findall(r'<tr[^>]*>[\s\S]*?</tr>', html, re.IGNORECASE):
        if "result-link" not in row and "result-snippet" not in row:
            continue
        url_m = re.search(r'<a[^>]+href="([^"]+)"[^>]*class="[^"]*result-link', row, re.I)
        if not url_m:
            continue
        url = url_m.group(1)
        title_m = re.search(r'class="[^"]*result-link[^"]*"[^>]*>([\s\S]*?)</a>', row, re.I)
        title = _strip_html(title_m.group(1)) if title_m else ""
        snippet_m = re.search(r'class="[^"]*result-snippet[^"]*"[^>]*>([\s\S]*?)</t[dD]', row, re.I)
        snippet = _strip_html(snippet_m.group(1)) if snippet_m else ""
        if url and title:
            results.append({"url": url, "title": title, "snippet": snippet, "engine": "duckduckgo-lite"})
    return results


def _parse_bing(html: str) -> list[dict]:
    """Bing parser with multiple fallback patterns."""
    results = []
    seen = set()

    blocks = re.findall(r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>[\s\S]*?</li>', html, re.IGNORECASE)
    if not blocks:
        blocks = re.findall(r'<div[^>]*class="[^"]*b_algo[^"]*"[^>]*>[\s\S]*?</div>', html, re.IGNORECASE)

    for block in blocks:
        cite_m = re.search(r'<cite[^>]*>([\s\S]*?)</cite>', block, re.I)
        cite_text = _strip_html(cite_m.group(1)) if cite_m else ""
        if cite_text.startswith("http"):
            url = cite_text.replace("›", "/").replace("&amp;", "&").strip()
        else:
            url_m = re.search(r'<a[^>]+href="(https?://[^"]+)"', block, re.I)
            url = url_m.group(1) if url_m else ""
            if "/url?q=" in url:
                try:
                    url = unquote(url.split("/url?q=")[1].split("&")[0])
                except Exception:
                    pass

        title_m = re.search(r'<h2[^>]*>([\s\S]*?)</h2>', block, re.I)
        title = _strip_html(title_m.group(1)) if title_m else ""
        snippet_m = re.search(r'<p[^>]*>([\s\S]*?)</p>', block, re.I)
        snippet = _strip_html(snippet_m.group(1)) if snippet_m else ""

        if url.startswith("http") and title and url not in seen:
            seen.add(url)
            results.append({"url": url, "title": title, "snippet": snippet, "engine": "bing"})

    return results


def _parse_brave(html: str) -> list[dict]:
    """Brave Search parser."""
    results = []
    seen = set()

    blocks = re.findall(
        r'<div[^>]*class="[^"]*(?:snippet|result|search-result)[^"]*"[^>]*>[\s\S]*?<a[^>]+href="https?://[^"]+"[^>]*>[\s\S]*?</a>[\s\S]*?</div>',
        html,
        re.IGNORECASE,
    )

    for block in blocks:
        url_m = re.search(r'<a[^>]+href="(https?://[^"]+)"', block, re.I)
        if not url_m:
            continue
        url = url_m.group(1)
        title_m = re.search(r'class="[^"]*title[^"]*"[^>]*>([\s\S]*?)</div>', block, re.I)
        title = _strip_html(title_m.group(1)) if title_m else ""
        snippet_m = re.search(r'class="[^"]*snippet[^"]*"[^>]*>([\s\S]*?)</div>', block, re.I)
        snippet = _strip_html(snippet_m.group(1)) if snippet_m else ""
        if url and title and url not in seen:
            seen.add(url)
            results.append({"url": url, "title": title, "snippet": snippet, "engine": "brave"})

    return results


def _parse_mojeek(html: str) -> list[dict]:
    """Mojeek parser."""
    results = []
    seen = set()

    blocks = re.findall(
        r'<div[^>]*class="[^"]*result[^"]*"[^>]*>[\s\S]*?</div>\s*</div>',
        html,
        re.IGNORECASE,
    )

    for block in blocks:
        url_m = re.search(r'<a[^>]+href="(https?://[^"]+)"', block, re.I)
        if not url_m:
            continue
        url = url_m.group(1)
        title_m = re.search(r'class="[^"]*ob[^"]*"[^>]*>([\s\S]*?)</a>', block, re.I)
        if not title_m:
            title_m = re.search(r'<a[^>]+href="https?://[^"]+"[^>]*>([\s\S]*?)</a>', block, re.I)
        title = _strip_html(title_m.group(1)) if title_m else ""
        snippet_m = re.search(r'<p[^>]*class="[^"]*s[^"]*"[^>]*>([\s\S]*?)</p>', block, re.I)
        snippet = _strip_html(snippet_m.group(1)) if snippet_m else ""
        if url and title and url not in seen:
            seen.add(url)
            results.append({"url": url, "title": title, "snippet": snippet, "engine": "mojeek"})

    return results


def _parse_google(html: str) -> list[dict]:
    """Google parser (non-JS fallback)."""
    results = []
    seen = set()

    blocks = re.findall(
        r'<div[^>]*class="[^"]*g[^"]*"[^>]*>[\s\S]*?</div>\s*</div>',
        html,
        re.IGNORECASE,
    )

    for block in blocks:
        url_m = re.search(r'<a[^>]+href="(/url\?q=[^"&]+)[^"]*"', block, re.I)
        if url_m:
            url = unquote(url_m.group(1)[7].split("&")[0])
        else:
            url_m2 = re.search(r'<a[^>]+href="(https?://[^"]+)"', block, re.I)
            url = url_m2.group(1) if url_m2 else ""

        title_m = re.search(r'<h3[^>]*>([\s\S]*?)</h3>', block, re.I)
        title = _strip_html(title_m.group(1)) if title_m else ""
        snippet_m = re.search(r'class="[^"]*(?:VwiC3b|BNeawe)[^"]*"[^>]*>([\s\S]*?)</div>', block, re.I)
        snippet = _strip_html(snippet_m.group(1)) if snippet_m else ""

        if url and title and "google.com" not in url and url not in seen:
            seen.add(url)
            results.append({"url": url, "title": title, "snippet": snippet, "engine": "google"})

    return results


# ── Engine Registry ───────────────────────────────────────────────────


@dataclass
class Engine:
    name: str
    build_url: callable
    parse: callable


ENGINES: list[Engine] = [
    Engine(
        name="bing",
        build_url=lambda q, page: f"https://www.bing.com/search?q={quote_plus(q)}&setlang=en&cc=US&first={(page - 1) * 10 + 1}",
        parse=_parse_bing,
    ),
    Engine(
        name="duckduckgo",
        build_url=lambda q, page: f"https://html.duckduckgo.com/html/?q={quote_plus(q)}&kl=wt-wt&s={(page - 1) * 10}",
        parse=_parse_ddg,
    ),
    Engine(
        name="duckduckgo-lite",
        build_url=lambda q, _page: f"https://lite.duckduckgo.com/lite/?q={quote_plus(q)}",
        parse=_parse_ddg_lite,
    ),
    Engine(
        name="brave",
        build_url=lambda q, page: f"https://search.brave.com/search?q={quote_plus(q)}&source=web&offset={(page - 1) * 10}",
        parse=_parse_brave,
    ),
    Engine(
        name="mojeek",
        build_url=lambda q, page: f"https://www.mojeek.com/search?q={quote_plus(q)}&page={page}",
        parse=_parse_mojeek,
    ),
    Engine(
        name="google",
        build_url=lambda q, page: f"https://www.google.com/search?q={quote_plus(q)}&hl=en&start={(page - 1) * 10}",
        parse=_parse_google,
    ),
]


# ── API Providers ─────────────────────────────────────────────────────


_circuit_broken_until: dict[str, float] = {}


async def _tavily_search(query: str, limit: int = 10) -> dict:
    from ..config import get_settings

    settings = get_settings()
    if not settings.tavily_api_key:
        return {"hits": [], "error": "TAVILY_API_KEY not set"}

    if _circuit_broken_until.get("tavily", 0) > time.time():
        return {"hits": [], "error": "tavily: circuit broken (rate limited)"}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": settings.tavily_api_key, "query": query, "max_results": limit, "include_answer": False},
            )
            data = resp.json()
            if "results" not in data:
                error = str(data.get("detail", data.get("error", "unknown")))
                if re.search(r"limit|quota|credit|rate|exceeds", error, re.I):
                    _circuit_broken_until["tavily"] = time.time() + 60
                return {"hits": [], "error": error}
            hits = [
                {"url": r["url"], "title": r.get("title", ""), "snippet": r.get("content", "")[:300], "engine": "tavily"}
                for r in data["results"]
            ]
            return {"hits": hits}
    except Exception as e:
        if re.search(r"limit|quota|credit|rate|exceeds", str(e), re.I):
            _circuit_broken_until["tavily"] = time.time() + 60
        return {"hits": [], "error": str(e)}


async def _exa_search(query: str, limit: int = 10) -> dict:
    from ..config import get_settings

    settings = get_settings()
    if not settings.exa_api_key:
        return {"hits": [], "error": "EXA_API_KEY not set"}

    if _circuit_broken_until.get("exa", 0) > time.time():
        return {"hits": [], "error": "exa: circuit broken (rate limited)"}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.exa.ai/search",
                headers={"x-api-key": settings.exa_api_key, "Content-Type": "application/json"},
                json={"query": query, "numResults": limit, "type": "neural", "contents": {"text": True}},
            )
            data = resp.json()
            if "results" not in data:
                error = str(data.get("error", "unknown"))
                if re.search(r"limit|quota|credit|rate|exceeds|429", error, re.I):
                    _circuit_broken_until["exa"] = time.time() + 60
                return {"hits": [], "error": error}
            hits = [
                {"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("text", "")[:300], "engine": "exa"}
                for r in data["results"]
            ]
            return {"hits": hits}
    except Exception as e:
        if re.search(r"limit|quota|credit|rate|exceeds|429", str(e), re.I):
            _circuit_broken_until["exa"] = time.time() + 60
        return {"hits": [], "error": str(e)}


async def _searxng_search(query: str, limit: int = 10) -> dict:
    from ..config import get_settings

    settings = get_settings()
    if not settings.searxng_base_url:
        return {"hits": [], "error": "SEARXNG_BASE_URL not set"}

    try:
        url = f"{settings.searxng_base_url.rstrip('/')}/search?q={quote_plus(query)}&format=json&categories=general"
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            data = resp.json()
            hits = [
                {"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("content", "")[:300], "engine": "searxng"}
                for r in data.get("results", [])[:limit]
            ]
            return {"hits": hits}
    except Exception as e:
        return {"hits": [], "error": str(e)}


async def _jina_search(query: str, limit: int = 10) -> dict:
    from ..config import get_settings

    settings = get_settings()
    try:
        url = f"https://s.jina.ai/{quote_plus(query)}"
        headers = {"Accept": "application/json"}
        if settings.jina_api_key:
            headers["Authorization"] = f"Bearer {settings.jina_api_key}"

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers=headers)
            data = resp.json()
            hits = [
                {"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("content", "")[:300], "engine": "jina"}
                for r in data.get("data", [])[:limit]
            ]
            return {"hits": hits}
    except Exception as e:
        return {"hits": [], "error": str(e)}


# ── Bing Rate Gate ────────────────────────────────────────────────────

BING_MIN_INTERVAL_MS = 1500
_last_bing_call: float = 0


async def _bing_gate() -> None:
    global _last_bing_call
    elapsed = (time.time() - _last_bing_call) * 1000
    if elapsed < BING_MIN_INTERVAL_MS:
        await asyncio.sleep((BING_MIN_INTERVAL_MS - elapsed) / 1000)
    _last_bing_call = time.time()


# ── Main Search Function ──────────────────────────────────────────────


async def relai_search(
    query: str,
    *,
    limit: int = 10,
    site: Optional[str] = None,
    rank: bool = True,
    mode: Optional[str] = None,
    page: int = 1,
    aggregate: bool = False,
) -> dict:
    """Public web search. Tries engines in succession with hard timeout."""
    q = f"site:{site} {query}" if site else query
    limit = min(max(limit, 1), 60)
    page = max(page, 1)

    search_query = q
    if mode == "news" and "after:" not in q and "before:" not in q:
        search_query = q + " after:2025-01-01"

    seen: set[str] = set()
    all_hits: list[dict] = []
    errors: list[str] = []

    budget_ms = 45_000 if aggregate else 20_000
    search_start = time.time()

    # Build provider list: API providers first, then HTML engines
    from ..config import get_settings

    settings = get_settings()
    api_providers: list[tuple[str, callable]] = []
    if settings.tavily_api_key:
        api_providers.append(("tavily", _tavily_search))
    if settings.exa_api_key:
        api_providers.append(("exa", _exa_search))
    if settings.searxng_base_url:
        api_providers.append(("searxng", _searxng_search))
    api_providers.append(("jina", _jina_search))

    # Run API providers first
    for name, search_fn in api_providers:
        if (time.time() - search_start) * 1000 > budget_ms:
            break
        if not aggregate and len(all_hits) >= limit:
            break

        try:
            result = await asyncio.wait_for(search_fn(search_query, limit), timeout=10)
        except asyncio.TimeoutError:
            errors.append(f"{name}: timed out")
            continue
        except Exception as e:
            errors.append(f"{name}: {str(e)[:100]}")
            continue

        hits = result.get("hits", [])
        error = result.get("error")
        if error:
            errors.append(error)
        if not hits and not error:
            continue

        for hit in hits:
            key = hit["url"].split("#")[0]
            if key in seen:
                continue
            seen.add(key)
            all_hits.append(hit)
            if not aggregate and len(all_hits) >= limit:
                break

        if not aggregate and all_hits:
            return {"hits": all_hits[:limit], "engine": name, "query": search_query, "tried": len(all_hits), "errors": errors}

    # Run HTML engines
    html_timeout = 8_000 if aggregate else 10_000
    for engine in ENGINES:
        if (time.time() - search_start) * 1000 > budget_ms:
            break
        if not aggregate and len(all_hits) >= limit:
            break

        if engine.name == "bing":
            await _bing_gate()

        url = engine.build_url(search_query, page)
        try:
            result = await relai_fetch(url, timeout_ms=html_timeout, retries=1 if aggregate else 2, cache=False)
        except Exception as e:
            errors.append(f"{engine.name}: {str(e)[:100]}")
            continue

        if not result.ok or not result.text:
            error_msg = result.error or f"HTTP {result.status or 'no response'}"
            errors.append(f"{engine.name}: {error_msg}")
            continue
        if len(result.text) < 100:
            errors.append(f"{engine.name}: response too short ({len(result.text)} chars)")
            continue

        hits = engine.parse(result.text)
        if not hits:
            errors.append(f"{engine.name}: parsed 0 results from {len(result.text)} chars")
            continue

        for hit in hits:
            key = hit["url"].split("#")[0]
            if key in seen:
                continue
            seen.add(key)
            all_hits.append(hit)
            if not aggregate and len(all_hits) >= limit:
                break

        if not aggregate and all_hits:
            return {"hits": all_hits[:limit], "engine": engine.name, "query": search_query, "tried": len(all_hits), "errors": errors}

    return {
        "hits": all_hits[:limit] if aggregate else all_hits,
        "engine": all_hits[0]["engine"] if all_hits else "none",
        "query": search_query,
        "tried": len(all_hits),
        "errors": errors or ["All engines returned no results"],
    }
