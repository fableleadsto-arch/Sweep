"""Wikipedia connector — REST summary + MediaWiki search (no API key)."""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import quote, urlparse

from ..models import IngestSource, RawItem, SourceKind
from .base import IngestConnector, http_json

_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_SEARCH = "https://en.wikipedia.org/w/api.php"


def _article_title(source: IngestSource) -> Optional[str]:
    parsed = urlparse(source.url)
    if "wikipedia.org" in (parsed.netloc or "") and parsed.path.startswith("/wiki/"):
        title = parsed.path[len("/wiki/"):]
        return quote(title.replace("_", " "), safe=" ")
    return None


class WikipediaConnector(IngestConnector):
    kind = SourceKind.WIKIPEDIA

    async def fetch(
        self, source: IngestSource, client: Any
    ) -> list[RawItem]:
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        title = _article_title(source)
        if title is not None:
            return await self._summaries([title], source, client, headers)
        search = source.config.get("query") or source.name or source.url
        data = await http_json(
            client,
            _SEARCH,
            params={
                "action": "query",
                "list": "search",
                "srsearch": search,
                "srlimit": "10",
                "format": "json",
            },
            headers=headers,
        )
        hits = (((data or {}).get("query") or {}).get("search")) if isinstance(data, dict) else None
        titles = [str(h.get("title")) for h in hits if isinstance(h, dict) and h.get("title")] if isinstance(hits, list) else []
        return await self._summaries(titles, source, client, headers)

    async def _summaries(
        self,
        titles: list[str],
        source: IngestSource,
        client: Any,
        headers: dict[str, str],
    ) -> list[RawItem]:
        items: list[RawItem] = []
        for title in titles[:8]:
            data = await http_json(client, _SUMMARY.format(title=quote(title, safe=" ")), headers=headers)
            if not isinstance(data, dict) or not data.get("title"):
                continue
            content_urls = (data.get("content_urls") or {}).get("desktop") if isinstance(data.get("content_urls"), dict) else None
            page_url = (content_urls or {}).get("page") if isinstance(content_urls, dict) else None
            page_url = page_url or source.url
            extract = data.get("extract") or ""
            description = data.get("description") or ""
            content = (
                f"{data.get('title', title)}\n"
                f"{('Description: ' + description) if description else ''}\n\n{extract}"
            ).strip()
            published = None
            timestamp = data.get("timestamp")
            if timestamp:
                published = _parse_timestamp(timestamp)
            items.append(
                RawItem(
                    title=data.get("title", title),
                    url=page_url,
                    content=content,
                    summary=extract,
                    published_at=published,
                    external_id=data.get("pageid") and str(data.get("pageid")),
                )
            )
        return items


def _parse_timestamp(value: str):
    from .base import parse_dt

    return parse_dt(str(value))
