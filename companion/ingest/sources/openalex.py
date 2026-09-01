"""OpenAlex connector — scholarly works from the open OpenAlex API.

``source.url`` is either an ``api.openalex.org`` query URL or a free-text
search; an optional ``OPENALEX_API_KEY`` raises the polite rate limit.
"""

from __future__ import annotations

from typing import Any, Optional

from ..models import IngestSource, RawItem, SourceKind
from .base import IngestConnector, first, http_json, parse_dt

_API = "https://api.openalex.org/works"


def _abstract(data: Optional[dict]) -> str:
    if not data:
        return ""
    inv = data.get("abstract_inverted_index")
    if not isinstance(inv, dict):
        return ""
    positions: dict[int, str] = {}
    for word, indexes in inv.items():
        for pos in indexes:
            positions[int(pos)] = word
    return " ".join(positions[i] for i in sorted(positions))


class OpenAlexConnector(IngestConnector):
    kind = SourceKind.OPENALEX

    async def fetch(
        self, source: IngestSource, client: Any
    ) -> list[RawItem]:
        if source.url.startswith(_API) or "api.openalex.org" in source.url:
            url = source.url
            params: dict[str, Any] = {}
        else:
            url = _API
            params = {"search": source.config.get("query") or source.name or source.url, "per-page": "25"}
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        if self.settings.openalex_api_key:
            headers["Authorization"] = f"Bearer {self.settings.openalex_api_key}"
        data = await http_json(client, url, params=params, headers=headers)
        if not isinstance(data, dict):
            return []
        works = data.get("results") if isinstance(data.get("results"), list) else [data]
        items: list[RawItem] = []
        for work in works:
            if not isinstance(work, dict) or not work.get("title"):
                continue
            title = str(work["title"])
            url_value = work.get("doi") or work.get("id") or source.url
            authors = ", ".join(
                (a.get("author", {}).get("display_name") or "") for a in (work.get("authorships") or [])
                if isinstance(a, dict)
            )
            journal = first(work.get("primary_location") or {}, "display_name", "source")
            content = (
                f"{title}\nAuthors: {authors}\n"
                f"Source: {journal or ''}\n"
                f"Year: {work.get('publication_year', '')}\n"
                f"Citations: {work.get('cited_by_count', 0)}\n\n{_abstract(work.get('abstract_inverted_index'))}"
            )
            published = None
            date_str = work.get("publication_date")
            if date_str:
                published = parse_dt(date_str)
            items.append(
                RawItem(
                    title=title,
                    url=url_value,
                    content=content.strip(),
                    summary=_abstract(work.get("abstract_inverted_index")),
                    author=authors,
                    published_at=published,
                    external_id=str(work.get("id") or ""),
                )
            )
        return items
