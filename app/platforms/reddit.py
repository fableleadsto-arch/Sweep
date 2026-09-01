"""Reddit platform adapter — search posts via public JSON API."""

from __future__ import annotations

import re
from typing import Optional

import httpx

from .base import PlatformAdapter
from ..core.types import PageData, SearchResult, SearchAccessMode, SurfPlatform
from ..core.http import relai_fetch
from ..scraping.markdown import html_to_markdown


class RedditAdapter(PlatformAdapter):
    @property
    def platform(self) -> SurfPlatform:
        return SurfPlatform.REDDIT

    def can_handle(self, url: str) -> bool:
        return "reddit.com" in url or "redd.it" in url

    async def search(
        self, query: str, *, limit: int = 5, subreddit: Optional[str] = None,
    ) -> tuple[list[SearchResult], Optional[str], SearchAccessMode]:
        try:
            search_query = f"site:reddit.com {query}"
            if subreddit:
                search_query = f"site:reddit.com/r/{subreddit} {query}"
            # Use web search for Reddit (their API requires OAuth)
            from ..search.engine import relai_search
            result = await relai_search(search_query, limit=limit)
            hits = result.get("hits", [])
            filtered = [h for h in hits if "reddit.com" in h.get("url", "")]
            if not filtered:
                return [], "No Reddit results found", SearchAccessMode.PUBLIC
            return [
                SearchResult(
                    url=h["url"], title=h["title"], snippet=h.get("snippet", ""),
                    provider="reddit_search", access_mode=SearchAccessMode.PUBLIC, platform="reddit",
                )
                for h in filtered
            ], None, SearchAccessMode.PUBLIC
        except Exception as e:
            return [], str(e), SearchAccessMode.UNAVAILABLE

    async def extract_page(self, url: str) -> Optional[PageData]:
        # Reddit .json endpoint
        json_url = url.rstrip("/") + ".json"
        try:
            result = await relai_fetch(json_url, timeout_ms=15_000, retries=2)
            if not result.ok:
                return None
            import json
            data = json.loads(result.text)
            post = data[0]["data"]["children"][0]["data"]
            comments = data[1]["data"]["children"] if len(data) > 1 else []

            text_parts = [
                f"# {post.get('title', '')}",
                f"by u/{post.get('author', '')} in r/{post.get('subreddit', '')}",
                f"Score: {post.get('score', 0)} | Comments: {post.get('num_comments', 0)}",
                "",
                post.get("selftext", "")[:10_000],
            ]
            for c in comments[:20]:
                if c.get("kind") == "t3":
                    cd = c["data"]
                    text_parts.append(f"\n---\n**u/{cd.get('author', '')}** ({cd.get('score', 0)} pts):\n{cd.get('body', '')[:1000]}")

            text = "\n".join(text_parts)
            return PageData(
                url=url, title=post.get("title", ""), text=text, markdown=text,
                links=[], headings=[], metadata={"subreddit": post.get("subreddit", "")},
                truncated=False, status=200, content_type="text/markdown",
                access_mode=SearchAccessMode.PUBLIC,
            )
        except Exception:
            return None
