"""Instagram platform adapter — search profiles/posts, extract public metadata."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

import httpx

from .base import PlatformAdapter
from ..core.types import PageData, SearchResult, SearchAccessMode, SurfPlatform
from ..core.http import relai_fetch

# ── URL patterns ──────────────────────────────────────────────────────
_IG_HOSTS = {"instagram.com", "www.instagram.com", "m.instagram.com"}


def _ig_type(url: str) -> Optional[str]:
    """Classify Instagram URL: profile, post, reel, story, tv."""
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) == 1 and parts[0]:
        return "profile"
    if parts[0] == "p":
        return "post"
    if parts[0] == "reel":
        return "reel"
    if parts[0] == "stories":
        return "story"
    if parts[0] == "tv":
        return "tv"
    if parts[0] == "explore":
        return "explore"
    return "page"


class InstagramAdapter(PlatformAdapter):
    @property
    def platform(self) -> SurfPlatform:
        return SurfPlatform.INSTAGRAM

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.hostname in _IG_HOSTS if parsed.hostname else False

    # ── Search ────────────────────────────────────────────────────────

    async def search(
        self, query: str, *, limit: int = 5, subreddit: Optional[str] = None,
    ) -> tuple[list[SearchResult], Optional[str], SearchAccessMode]:
        # Instagram API requires auth — use web search as public fallback
        try:
            from ..search.engine import relai_search
            result = await relai_search(f"site:instagram.com {query}", limit=limit)
            hits = result.get("hits", [])
            filtered = [h for h in hits if "instagram.com" in h.get("url", "")]
            if not filtered:
                return [], "No Instagram results", SearchAccessMode.PUBLIC
            return [
                SearchResult(
                    url=h["url"], title=h["title"], snippet=h.get("snippet", ""),
                    provider="instagram_search", access_mode=SearchAccessMode.PUBLIC,
                    platform="instagram",
                )
                for h in filtered
            ], None, SearchAccessMode.PUBLIC
        except Exception as e:
            return [], str(e), SearchAccessMode.UNAVAILABLE

    # ── Extraction ────────────────────────────────────────────────────

    async def extract_page(self, url: str) -> Optional[PageData]:
        if not self.can_handle(url):
            return None

        ig_type = _ig_type(url)
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")

        # Try embed/oembed endpoint
        oembed_data = await self._oembed(url)

        # Fallback: page fetch
        page = await self._page_extract(url)

        parts_out: list[str] = []
        title = ""
        metadata: dict[str, str] = {"instagram_type": ig_type}

        if oembed_data:
            title = oembed_data.get("title", "")
            author = oembed_data.get("author_name", "")
            if title:
                parts_out.append(f"# {title}")
            if author:
                parts_out.append(f"Author: @{author}")
                metadata["author"] = f"@{author}"
            if oembed_data.get("thumbnail_url"):
                parts_out.append(f"Thumbnail: {oembed_data['thumbnail_url']}")

        # Extract username from URL path
        if ig_type == "profile" and parts:
            username = parts[0]
            parts_out.append(f"Profile: @{username}")
            metadata["username"] = f"@{username}"
        elif ig_type in ("post", "reel") and len(parts) >= 2:
            metadata["shortcode"] = parts[1]

        parts_out.append(f"Type: {ig_type.title()}")

        if page and page.text:
            parts_out.append("")
            # Only include the meaningful portion
            text_lines = page.text.split("\n")
            useful = []
            for line in text_lines:
                stripped = line.strip()
                if any(kw in stripped.lower() for kw in (
                    "followers", "following", "posts", "bio", "description",
                    "likes", "comments", "views",
                )):
                    useful.append(stripped)
                if len(useful) >= 30:
                    break
            if useful:
                parts_out.extend(useful)
            elif page.text[:2000]:
                parts_out.append(page.text[:2000])

        markdown = "\n".join(parts_out)

        return PageData(
            url=url,
            title=title or f"Instagram {ig_type.title()}",
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

    async def _oembed(self, url: str) -> Optional[dict]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://api.instagram.com/oembed/",
                    params={"url": url},
                )
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass
        return None

    async def _page_extract(self, url: str) -> Optional[PageData]:
        try:
            result = await relai_fetch(url, timeout_ms=12_000, retries=1)
            if not result.ok:
                return None
            from ..extraction.page_data import extract_page_data
            return extract_page_data(
                html=result.text, url=result.url, status=result.status,
                content_type=result.content_type, max_chars=12_000,
            )
        except Exception:
            return None
