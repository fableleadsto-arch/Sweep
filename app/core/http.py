"""Resilient HTTP fetch layer.

One gateway every crawl, search and extraction call goes through:
  - per-host token-bucket rate limiting + global concurrency ceiling
  - retry with exponential backoff and jitter on 429/5xx/network errors
  - user-agent rotation and optional proxy for blocked hosts
  - CAPTCHA / interstitial detection
  - SSRF guard: no private, link-local or metadata hosts, ever
  - short-lived response cache

Reference architecture: steel (sandboxed, proxied, rate-limited fetching).
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional
from urllib.parse import urlparse

import httpx

from ..config import get_settings
from .guard import is_private_host, assert_safe_url
from .types import FetchResult

# ── User Agents ───────────────────────────────────────────────────────

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

# ── Rate Limiting ─────────────────────────────────────────────────────

HOST_MIN_INTERVAL_MS = 700
MAX_CONCURRENT = 6
CACHE_TTL_MS = 120_000
CACHE_MAX_ENTRIES = 200

_last_hit: dict[str, float] = {}
_host_failures: dict[str, int] = {}
_in_flight = 0
_waiters: list[asyncio.Event] = []
_lock = asyncio.Lock()

# ── Response Cache ────────────────────────────────────────────────────

_cache: dict[str, tuple[float, FetchResult]] = {}


def _cache_get(key: str) -> Optional[FetchResult]:
    entry = _cache.get(key)
    if entry is None:
        return None
    at, value = entry
    if time.time() - at > CACHE_TTL_MS / 1000:
        _cache.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: FetchResult) -> None:
    if len(_cache) >= CACHE_MAX_ENTRIES:
        # Remove oldest entry
        oldest_key = min(_cache, key=lambda k: _cache[k][0])
        _cache.pop(oldest_key, None)
    _cache[key] = (time.time(), value)


# ── Concurrency Control ───────────────────────────────────────────────

async def _acquire_slot() -> None:
    global _in_flight
    async with _lock:
        if _in_flight < MAX_CONCURRENT:
            _in_flight += 1
            return
        event = asyncio.Event()
        _waiters.append(event)
    await event.wait()
    async with _lock:
        _in_flight += 1


def _release_slot() -> None:
    global _in_flight
    _in_flight = max(0, _in_flight - 1)
    if _waiters:
        event = _waiters.pop(0)
        event.set()


async def _polite_delay(host: str) -> None:
    penalty = (_host_failures.get(host, 0)) * 0.4
    gap = HOST_MIN_INTERVAL_MS / 1000 + penalty
    elapsed = time.time() - _last_hit.get(host, 0)
    if elapsed < gap:
        await asyncio.sleep(gap - elapsed)
    _last_hit[host] = time.time()


# ── CAPTCHA Detection ────────────────────────────────────────────────

CAPTCHA_MARKERS = [
    "captcha",
    "are you a robot",
    "unusual traffic",
    "verify you are human",
    "cf-browser-verification",
    "checking your browser before",
    "access denied",
    "enable javascript and cookies to continue",
]


def looks_blocked(status: int, text: str) -> bool:
    """Detect challenge pages that look like a 200 but carry no content."""
    if status in (403, 429, 503):
        return True
    head = text[:4000].lower()
    if len(head) < 200:
        return False
    return any(m in head for m in CAPTCHA_MARKERS)


# ── Proxy ─────────────────────────────────────────────────────────────

def _proxied_url(url: str) -> str:
    settings = get_settings()
    proxy = settings.relai_proxy_url
    if not proxy:
        return url
    if "{url}" in proxy:
        return proxy.replace("{url}", url)
    return f"{proxy}{url}"


def _headers_for(attempt: int) -> dict[str, str]:
    return {
        "User-Agent": USER_AGENTS[attempt % len(USER_AGENTS)],
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }


# ── The Fetch ─────────────────────────────────────────────────────────

async def relai_fetch(
    raw_url: str,
    *,
    timeout_ms: int = 15_000,
    retries: int = 2,
    method: str = "GET",
    body: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    cache: bool = True,
    max_bytes: int = 2_000_000,
) -> FetchResult:
    """Fetch a public URL with rate limiting, retries and block detection.

    Never raises on a network failure — reports it in the result.
    """
    now_iso = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Validate URL
    try:
        parsed_url = assert_safe_url(raw_url)
    except ValueError as e:
        return FetchResult(ok=False, status=0, url=raw_url, error=str(e), fetched_at=now_iso())

    # Check cache
    cache_key = f"{method} {parsed_url}"
    if cache and method == "GET":
        cached = _cache_get(cache_key)
        if cached:
            return cached

    retries = min(max(retries, 0), 4)
    timeout_s = timeout_ms / 1000
    host = urlparse(parsed_url).host or ""

    last = FetchResult(
        ok=False, status=0, url=parsed_url, error="not attempted", fetched_at=now_iso()
    )

    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(timeout_s),
        limits=httpx.Limits(max_connections=MAX_CONCURRENT),
    ) as client:
        for attempt in range(retries + 1):
            await _acquire_slot()
            try:
                await _polite_delay(host)

                request_url = parsed_url
                response = None

                # Manual redirect following (max 4)
                for _redirect in range(5):
                    target = _proxied_url(request_url) if attempt > 0 else request_url
                    req_headers = {**_headers_for(attempt), **(headers or {})}

                    try:
                        if method == "GET":
                            response = await client.get(target, headers=req_headers)
                        else:
                            response = await client.post(target, headers=req_headers, content=body)
                    except httpx.TimeoutException:
                        last = FetchResult(
                            ok=False, status=0, url=parsed_url,
                            error=f"timed out after {timeout_ms}ms",
                            attempts=attempt + 1, fetched_at=now_iso(),
                        )
                        _host_failures[host] = _host_failures.get(host, 0) + 1
                        break
                    except Exception as e:
                        last = FetchResult(
                            ok=False, status=0, url=parsed_url,
                            error=str(e)[:200],
                            attempts=attempt + 1, fetched_at=now_iso(),
                        )
                        _host_failures[host] = _host_failures.get(host, 0) + 1
                        break

                    if response is None:
                        break

                    # Handle redirects
                    if 300 <= response.status_code < 400:
                        location = response.headers.get("location", "")
                        if not location:
                            break
                        try:
                            from urllib.parse import urljoin
                            next_url = urljoin(request_url, location)
                            assert_safe_url(next_url)
                            request_url = next_url
                            continue
                        except ValueError:
                            break

                    break  # Non-redirect response

                if response is None:
                    continue

                # Read body with byte cap
                content = response.content[:max_bytes]
                text = content.decode("utf-8", errors="replace")
                content_type = response.headers.get("content-type", "")
                blocked = looks_blocked(response.status_code, text)

                last = FetchResult(
                    ok=response.is_success and not blocked,
                    status=response.status_code,
                    url=request_url,
                    text=text,
                    content_type=content_type,
                    blocked=blocked,
                    attempts=attempt + 1,
                    error="blocked or challenged by the host" if blocked else None,
                    fetched_at=now_iso(),
                )

                if last.ok:
                    _host_failures.pop(host, None)
                    if cache and method == "GET":
                        _cache_set(cache_key, last)
                    return last

                _host_failures[host] = _host_failures.get(host, 0) + 1
                # 4xx that is not a throttle will not change on retry
                if not blocked and 400 <= response.status_code < 500 and response.status_code != 429:
                    return last

            finally:
                _release_slot()

            # Exponential backoff
            if attempt < retries:
                import random
                backoff = 0.5 * (2 ** attempt) + random.random() * 0.4
                await asyncio.sleep(backoff)

    return last


# ── Pooled Execution ──────────────────────────────────────────────────

async def pooled(
    items: list,
    limit: int,
    worker,
) -> list:
    """Run tasks with a bounded worker pool."""
    size = min(max(limit, 1), MAX_CONCURRENT)
    results = [None] * len(items)
    cursor = 0
    lock = asyncio.Lock()

    async def _worker():
        nonlocal cursor
        while True:
            async with lock:
                if cursor >= len(items):
                    break
                i = cursor
                cursor += 1
            results[i] = await worker(items[i], i)

    await asyncio.gather(*[_worker() for _ in range(min(size, len(items)))])
    return results
