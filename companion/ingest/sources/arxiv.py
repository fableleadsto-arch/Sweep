"""arXiv connector — Atom API over submitted papers (no API key)."""

from __future__ import annotations

import re
from typing import Any
from xml.etree import ElementTree

from ..models import IngestSource, RawItem, SourceKind
from .base import IngestConnector, http_text, parse_dt

_API = "https://export.arxiv.org/api/query"


def _local(tag: str) -> str:
    return tag.split("}")[-1] if tag else ""


class ArxivConnector(IngestConnector):
    kind = SourceKind.ARXIV

    async def fetch(
        self, source: IngestSource, client: Any
    ) -> list[RawItem]:
        if "arxiv.org" in source.url and "/abs/" in source.url:
            query = f"{_API}?id_list={source.url.split('/abs/')[-1].strip('/').split('?')[0]}"
        elif source.url.startswith(_API):
            query = source.url
        else:
            search = source.config.get("query") or source.name.strip() or source.url.strip()
            query = f"{_API}?search_query=all:{search}&sortBy=submittedDate&max_results=20"
        headers = {"User-Agent": self.user_agent, "Accept": "application/atom+xml"}
        text = await http_text(client, query, headers=headers)
        if not text:
            return []
        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError:
            return []
        items: list[RawItem] = []
        for entry in root.iter():
            if _local(entry.tag) != "entry":
                continue
            item = self._entry(entry, source)
            if item:
                items.append(item)
        return items

    def _entry(self, entry: Any, source: IngestSource) -> RawItem | None:
        def text(name: str) -> str:
            for child in entry.iter():
                if _local(child.tag) == name and child.text and child.text.strip():
                    return child.text.strip()
            return ""

        title = text("title") or source.name
        link = ""
        for child in entry.iter():
            if _local(child.tag) == "id" and child.text:
                link = child.text.strip()
                break
        summary = re.sub(r"\s+", " ", text("summary"))
        published = parse_dt(text("published") or text("updated"))
        author = ""
        for child in entry.iter():
            if _local(child.tag) == "name" and child.text:
                author = child.text.strip()
                break
        if not title and not summary:
            return None
        return RawItem(
            title=title,
            url=link or source.url,
            content=f"{title}\n\n{summary}" if summary else title,
            summary=summary,
            author=author,
            published_at=published,
            external_id=link.rstrip("/").split("/abs/")[-1] if "/abs/" in link else "",
        )
