"""X (Twitter) platform adapter — search posts, extract threads + metadata."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

import httpx

from .base import PlatformAdapter
from ..core.types import PageData, SearchResult, SearchAccessMode, SurfPlatform
from ..core.http import relai_fetch

# ── URL patterns ──────────────────────────────────────────────────────
_X_HOSTS = {"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"}


def _is_tweet_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.hostname in _X_HOSTS if parsed.hostname else False


def _tweet_id(url: str) -> Optional[str]:
    """Extract tweet ID from URL like /user/status/1234567890."""
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    # /user/status/ID or /user/status/ID/photo/1
    if "status" in parts:
        idx = parts.index("status")
        if idx + 1 < len(parts):
            return parts[idx + 1].split("/")[0]
    return None


def _author_from_url(url: str) -> str:
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    return parts[0] if parts else ""


class XAdapter(PlatformAdapter):
    @property
    def platform(self) -> SurfPlatform:
        return SurfPlatform.X

    def can_handle(self, url: str) -> bool:
        return _is_tweet_url(url)

    # ── Search ────────────────────────────────────────────────────────

    async def search(
        self, query: str, *, limit: int = 5, subreddit: Optional[str] = None,
    ) -> tuple[list[SearchResult], Optional[str], SearchAccessMode]:
        # X API requires OAuth2 — use web search as public fallback
        try:
            from ..search.engine import relai_search
            result = await relai_search(f"site:x.com OR site:twitter.com {query}", limit=limit)
            hits = result.get("hits", [])
            filtered = [h for h in hits if any(
                host in h.get("url", "") for host in ("x.com", "twitter.com")
            )]
            if not filtered:
                return [], "No X/Twitter results", SearchAccessMode.PUBLIC
            return [
                SearchResult(
                    url=h["url"], title=h["title"], snippet=h.get("snippet", ""),
                    provider="x_search", access_mode=SearchAccessMode.PUBLIC,
                    platform="x",
                )
                for h in filtered
            ], None, SearchAccessMode.PUBLIC
        except Exception as e:
            return [], str(e), SearchAccessMode.UNAVAILABLE

    # ── Extraction ────────────────────────────────────────────────────

    async def extract_page(self, url: str) -> Optional[PageData]:
        if not _is_tweet_url(url):
            return None

        # Attempt nitter mirror or syndication API first
        syndication = await self._syndication_extract(url)
        if syndication:
            return syndication

        # Fallback: fetch the page and extract what we can
        return await self._page_extract(url)

    async def _syndication_extract(self, url: str) -> Optional[PageData]:
        """Try Twitter syndication/embed endpoint for tweet data."""
        tid = _tweet_id(url)
        if not tid:
            return None
        try:
            async with httpx.AsyncClient(timeout=12, follow_redirects=True) as client:
                resp = await client.get(
                    f"https://cdn.syndication.twimg.com/tweet-result",
                    params={"id": tid, "token": "0"},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if resp.status_code != 200:
                    return None
                data = resp.json()
                author = data.get("author", {})
                author_name = author.get("name", "")
                author_handle = author.get("screen_name", "")
                text = data.get("text", "")
                created = data.get("created_at", "")
                retweets = data.get("retweet_count", 0)
                likes = data.get("favorite_count", 0)
                replies = data.get("reply_count", 0)

                parts = [
                    f"# @{author_handle}" + (f" ({author_name})" if author_name else ""),
                    f"Posted: {created}" if created else "",
                    f"💬 {replies}  🔁 {retweets}  ❤️ {likes}",
                    "",
                    text,
                ]

                # Thread context
                if data.get("thread"):
                    parts.append("\n--- Thread ---")
                    for tweet in data["thread"][:10]:
                        ta = tweet.get("author", {})
                        parts.append(f"\n**@{ta.get('screen_name', '')}**: {tweet.get('text', '')}")

                # Media
                photos = data.get("photos", [])
                if photos:
                    parts.append(f"\n📸 {len(photos)} image(s)")
                if data.get("video"):
                    parts.append("🎥 Video attached")

                markdown = "\n".join(p for p in parts if p is not None)
                metadata: dict[str, str] = {"tweet_id": tid}
                if author_handle:
                    metadata["author"] = f"@{author_handle}"
                if created:
                    metadata["posted_at"] = created
                metadata["retweets"] = str(retweets)
                metadata["likes"] = str(likes)

                return PageData(
                    url=url,
                    title=f"@{author_handle}: {text[:80]}",
                    text=markdown,
                    markdown=markdown,
                    links=[],
                    headings=[],
                    metadata=metadata,
                    truncated=False,
                    status=200,
                    content_type="text/markdown",
                    access_mode=SearchAccessMode.PUBLIC,
                )
        except Exception:
            pass
        return None

    async def _page_extract(self, url: str) -> Optional[PageData]:
        """Fallback: fetch the actual X page."""
        try:
            result = await relai_fetch(url, timeout_ms=12_000, retries=1)
            if not result.ok:
                return None
            from ..extraction.page_data import extract_page_data
            page = extract_page_data(
                html=result.text, url=result.url, status=result.status,
                content_type=result.content_type, max_chars=15_000,
            )
            # Enrich metadata
            tid = _tweet_id(url)
            author = _author_from_url(url)
            if tid:
                page.metadata["tweet_id"] = tid
            if author:
                page.metadata["author"] = f"@{author}"
            return page
        except Exception:
            return None
