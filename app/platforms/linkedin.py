"""LinkedIn platform adapter — search profiles/companies/posts, extract public data."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

import httpx

from .base import PlatformAdapter
from ..core.types import PageData, SearchResult, SearchAccessMode, SurfPlatform
from ..core.http import relai_fetch

# ── URL patterns ──────────────────────────────────────────────────────
_LI_HOSTS = {"linkedin.com", "www.linkedin.com"}


def _linkedin_type(url: str) -> Optional[str]:
    """Classify LinkedIn URL: profile, company, post, article, jobs."""
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if not parts or parts[0] == "in":
        return "profile"
    if parts[0] == "company":
        return "company"
    if parts[0] == "posts":
        return "post"
    if parts[0] == "pulse":
        return "article"
    if parts[0] == "jobs":
        return "jobs"
    if parts[0] == "schools":
        return "school"
    return "page"


class LinkedInAdapter(PlatformAdapter):
    @property
    def platform(self) -> SurfPlatform:
        return SurfPlatform.LINKEDIN

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.hostname in _LI_HOSTS if parsed.hostname else False

    # ── Search ────────────────────────────────────────────────────────

    async def search(
        self, query: str, *, limit: int = 5, subreddit: Optional[str] = None,
    ) -> tuple[list[SearchResult], Optional[str], SearchAccessMode]:
        # LinkedIn blocks automated access — use web search as public fallback
        try:
            from ..search.engine import relai_search
            result = await relai_search(f"site:linkedin.com {query}", limit=limit)
            hits = result.get("hits", [])
            filtered = [h for h in hits if "linkedin.com" in h.get("url", "")]
            if not filtered:
                return [], "No LinkedIn results", SearchAccessMode.PUBLIC
            return [
                SearchResult(
                    url=h["url"], title=h["title"], snippet=h.get("snippet", ""),
                    provider="linkedin_search", access_mode=SearchAccessMode.PUBLIC,
                    platform="linkedin",
                )
                for h in filtered
            ], None, SearchAccessMode.PUBLIC
        except Exception as e:
            return [], str(e), SearchAccessMode.UNAVAILABLE

    # ── Extraction ────────────────────────────────────────────────────

    async def extract_page(self, url: str) -> Optional[PageData]:
        if not self.can_handle(url):
            return None
        return await self._page_extract(url)

    async def _page_extract(self, url: str) -> Optional[PageData]:
        """Fetch LinkedIn page and extract available structured data."""
        li_type = _linkedin_type(url)
        try:
            result = await relai_fetch(url, timeout_ms=12_000, retries=1)
            if not result.ok:
                return None

            from ..extraction.page_data import extract_page_data
            page = extract_page_data(
                html=result.text, url=result.url, status=result.status,
                content_type=result.content_type, max_chars=15_000,
            )

            # LinkedIn pages have og:meta — extract what we can
            parts: list[str] = []
            title = page.title or ""
            if title:
                parts.append(f"# {title}")
            parts.append(f"Type: {li_type.title()}")

            # Try to pull person/org name from title patterns
            if li_type == "profile" and " - " in title:
                name = title.split(" - ")[0].strip()
                parts.append(f"Name: {name}")
                page.metadata["person_name"] = name
            elif li_type == "company" and " - " in title:
                company = title.split(" - ")[0].strip()
                parts.append(f"Company: {company}")
                page.metadata["company_name"] = company

            # Add page text
            if page.text:
                parts.append("")
                parts.append(page.text[:8_000])

            page.metadata["linkedin_type"] = li_type
            page.text = "\n".join(parts)
            page.markdown = page.text

            return page
        except Exception:
            return None
