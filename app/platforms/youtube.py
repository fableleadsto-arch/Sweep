"""YouTube platform adapter — search videos, extract metadata, chapters, transcripts."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse, parse_qs

import httpx

from .base import PlatformAdapter
from ..core.types import PageData, SearchResult, SearchAccessMode, SurfPlatform
from ..core.http import relai_fetch
from ..config import get_settings

# ── URL patterns ──────────────────────────────────────────────────────
_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


def _video_id(url: str) -> Optional[str]:
    """Extract video ID from any YouTube URL format."""
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.strip("/").split("/")[0]
    if parsed.hostname in _YT_HOSTS:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
        # /shorts/VIDEO_ID or /embed/VIDEO_ID
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] in ("shorts", "embed", "v"):
            return parts[1]
    return None


def _parse_duration(iso: str) -> Optional[int]:
    """Convert ISO 8601 duration (PT1H2M3S) to seconds."""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not m:
        return None
    return (int(m.group(1) or 0) * 3600) + (int(m.group(2) or 0) * 60) + int(m.group(3) or 0)


class YouTubeAdapter(PlatformAdapter):
    @property
    def platform(self) -> SurfPlatform:
        return SurfPlatform.YOUTUBE

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.hostname in _YT_HOSTS

    # ── Search ────────────────────────────────────────────────────────

    async def search(
        self, query: str, *, limit: int = 5, subreddit: Optional[str] = None,
    ) -> tuple[list[SearchResult], Optional[str], SearchAccessMode]:
        settings = get_settings()
        yt_key = getattr(settings, "youtube_api_key", None)

        # Try YouTube Data API v3 if key available
        if yt_key:
            return await self._search_api(query, limit, yt_key)

        # Fallback: site-scoped web search
        return await self._search_web(query, limit)

    async def _search_api(
        self, query: str, limit: int, api_key: str,
    ) -> tuple[list[SearchResult], Optional[str], SearchAccessMode]:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://www.googleapis.com/youtube/v3/search",
                    params={
                        "part": "snippet",
                        "q": query,
                        "type": "video",
                        "maxResults": min(limit, 25),
                        "key": api_key,
                    },
                )
                if resp.status_code != 200:
                    return [], f"YouTube API {resp.status_code}", SearchAccessMode.UNAVAILABLE

                data = resp.json()
                results = []
                for item in data.get("items", [])[:limit]:
                    vid = item["id"].get("videoId", "")
                    snippet = item.get("snippet", {})
                    results.append(SearchResult(
                        url=f"https://www.youtube.com/watch?v={vid}",
                        title=snippet.get("title", ""),
                        snippet=snippet.get("description", "")[:300],
                        provider="youtube_api",
                        access_mode=SearchAccessMode.PUBLIC,
                        platform="youtube",
                        published_at=snippet.get("publishedAt", ""),
                    ))
                return results, None, SearchAccessMode.PUBLIC
        except Exception as e:
            return [], str(e), SearchAccessMode.UNAVAILABLE

    async def _search_web(
        self, query: str, limit: int,
    ) -> tuple[list[SearchResult], Optional[str], SearchAccessMode]:
        try:
            from ..search.engine import relai_search
            result = await relai_search(f"site:youtube.com {query}", limit=limit)
            hits = result.get("hits", [])
            filtered = [h for h in hits if any(
                host in h.get("url", "")
                for host in ("youtube.com", "youtu.be")
            )]
            if not filtered:
                return [], "No YouTube results", SearchAccessMode.PUBLIC
            return [
                SearchResult(
                    url=h["url"], title=h["title"], snippet=h.get("snippet", ""),
                    provider="youtube_web", access_mode=SearchAccessMode.PUBLIC,
                    platform="youtube",
                )
                for h in filtered
            ], None, SearchAccessMode.PUBLIC
        except Exception as e:
            return [], str(e), SearchAccessMode.UNAVAILABLE

    # ── Extraction ────────────────────────────────────────────────────

    async def extract_page(self, url: str) -> Optional[PageData]:
        vid = _video_id(url)
        if not vid:
            return None

        # Try oEmbed for structured metadata
        oembed = await self._oembed(vid)
        # Try page fetch for additional data
        page = await self._fetch_page(url)

        title = ""
        author = ""
        text_parts: list[str] = []

        if oembed:
            title = oembed.get("title", "")
            author = oembed.get("author_name", "")
            text_parts.append(f"# {title}")
            text_parts.append(f"Channel: {author}")
            if oembed.get("thumbnail_url"):
                text_parts.append(f"Thumbnail: {oembed['thumbnail_url']}")
            text_parts.append("")

        if page and page.text:
            # Extract useful sections from page text
            lines = page.text.split("\n")
            for line in lines:
                stripped = line.strip()
                # Grab description, chapters, metadata
                if any(kw in stripped.lower() for kw in (
                    "description", "chapter", "published", "views",
                    "likes", "subscribe", "about",
                )):
                    text_parts.append(stripped)
                if len("\n".join(text_parts)) > 8000:
                    break

        text = "\n".join(text_parts) if text_parts else f"Video: {vid}"
        metadata: dict[str, str] = {"video_id": vid}
        if author:
            metadata["channel"] = author
        if oembed:
            metadata["type"] = oembed.get("type", "video")

        return PageData(
            url=url,
            title=title or f"YouTube Video {vid}",
            text=text,
            markdown=text,
            links=[],
            headings=[],
            metadata=metadata,
            truncated=False,
            status=200,
            content_type="text/markdown",
            access_mode=SearchAccessMode.PUBLIC,
        )

    async def _oembed(self, vid: str) -> Optional[dict]:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://www.youtube.com/oembed",
                    params={"url": f"https://www.youtube.com/watch?v={vid}", "format": "json"},
                )
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass
        return None

    async def _fetch_page(self, url: str) -> Optional[PageData]:
        try:
            result = await relai_fetch(url, timeout_ms=12_000, retries=1)
            if not result.ok:
                return None
            from ..extraction.page_data import extract_page_data
            return extract_page_data(
                html=result.text, url=result.url, status=result.status,
                content_type=result.content_type, max_chars=15_000,
            )
        except Exception:
            return None
