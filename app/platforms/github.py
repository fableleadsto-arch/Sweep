"""GitHub platform adapter — search repos/issues/users, extract READMEs."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

import httpx

from .base import PlatformAdapter
from ..core.types import PageData, SearchResult, SearchAccessMode, SurfPlatform
from ..extraction.page_data import extract_page_data


class GitHubAdapter(PlatformAdapter):
    @property
    def platform(self) -> SurfPlatform:
        return SurfPlatform.GITHUB

    def can_handle(self, url: str) -> bool:
        return "github.com" in url

    async def search(
        self, query: str, *, limit: int = 5, subreddit: Optional[str] = None,
    ) -> tuple[list[SearchResult], Optional[str], SearchAccessMode]:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://api.github.com/search/repositories",
                    params={"q": query, "per_page": limit, "sort": "stars"},
                    headers={"Accept": "application/vnd.github.v3+json"},
                )
                if resp.status_code != 200:
                    return [], f"GitHub API returned {resp.status_code}", SearchAccessMode.UNAVAILABLE

                data = resp.json()
                results = []
                for repo in data.get("items", [])[:limit]:
                    results.append(SearchResult(
                        url=repo["html_url"],
                        title=repo["full_name"],
                        snippet=repo.get("description", "")[:300],
                        provider="github",
                        access_mode=SearchAccessMode.PUBLIC,
                        platform="github",
                    ))
                return results, None, SearchAccessMode.PUBLIC
        except Exception as e:
            return [], str(e), SearchAccessMode.UNAVAILABLE

    async def extract_page(self, url: str) -> Optional[PageData]:
        try:
            parsed = urlparse(url)
            parts = parsed.path.strip("/").split("/")
            if len(parts) < 2:
                return None
            owner, repo = parts[0], parts[1]

            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"https://api.github.com/repos/{owner}/{repo}/readme",
                    headers={"Accept": "application/vnd.github.v3.raw"},
                )
                if resp.status_code != 200:
                    return None
                content = resp.text[:20_000]
                return PageData(
                    url=url,
                    title=f"{owner}/{repo}",
                    text=content,
                    markdown=content,
                    links=[],
                    headings=[],
                    metadata={"type": "github_readme"},
                    truncated=len(resp.text) > 20_000,
                    status=200,
                    content_type="text/markdown",
                    access_mode=SearchAccessMode.PUBLIC,
                )
        except Exception:
            return None
