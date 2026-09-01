"""Crossref connector — bibliographic records from the open Crossref REST API."""

from __future__ import annotations

from typing import Any, Optional

from ..models import IngestSource, RawItem, SourceKind
from .base import IngestConnector, first, http_json, parse_dt

_API = "https://api.crossref.org/works"


class CrossrefConnector(IngestConnector):
    kind = SourceKind.CROSSREF

    async def fetch(
        self, source: IngestSource, client: Any
    ) -> list[RawItem]:
        if "api.crossref.org" in source.url:
            url = source.url
            params: dict[str, Any] = {}
        else:
            url = _API
            params = {
                "query.bibliographic": source.config.get("query") or source.name or source.url,
                "rows": "20",
            }
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        data = await http_json(client, url, params=params, headers=headers)
        message = (data or {}).get("message") if isinstance(data, dict) else None
        items_container = (message or {}).get("items") if isinstance(message, dict) else None
        if isinstance(items_container, list):
            works = items_container
        elif isinstance(message, dict) and message.get("title"):
            works = [message]
        else:
            return []
        items: list[RawItem] = []
        for work in works:
            if not isinstance(work, dict):
                continue
            titles = work.get("title") or []
            title = titles[0] if titles else ""
            if not title:
                continue
            authors = ", ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip() for a in (work.get("author") or [])
                if isinstance(a, dict)
            )
            doi = work.get("DOI")
            journal = first(work, "container-title")
            if isinstance(journal, list):
                journal = journal[0] if journal else ""
            year = ""
            issued = work.get("issued", {}).get("date-parts")
            if issued and issued[0] and issued[0][0]:
                year = str(issued[0][0])
            abstract = work.get("abstract") or ""
            import re as _re

            abstract = _re.sub(r"<[^>]+>", " ", abstract)
            content = (
                f"{title}\nAuthors: {authors}\n"
                f"Journal: {journal or ''}\nYear: {year}\nDOI: {doi}\n\n{abstract.strip()}"
            )
            published = None
            if issued and issued[0]:
                parts = issued[0]
                if len(parts) >= 3:
                    published = parse_dt(f"{parts[0]}-{parts[1]:02d}-{parts[2]:02d}")
                elif len(parts) == 2:
                    published = parse_dt(f"{parts[0]}-{parts[1]:02d}")
            items.append(
                RawItem(
                    title=title,
                    url=f"https://doi.org/{doi}" if doi else source.url,
                    content=content.strip(),
                    summary=abstract.strip(),
                    author=authors,
                    published_at=published,
                    external_id=doi or str(work.get("URL") or ""),
                )
            )
        return items
