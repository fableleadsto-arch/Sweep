"""RSS / Atom feed connector (stdlib XML parsing, no external dependency)."""

from __future__ import annotations

from typing import Any, Optional
from xml.etree import ElementTree

from ..models import IngestSource, RawItem, SourceKind
from .base import IngestConnector, http_text, parse_dt


def _local(tag: str) -> str:
    return tag.split("}")[-1] if tag else ""


def _child_text(node: Any, *names: str) -> str:
    wanted = {_local(name) for name in names}
    for child in node.iter():
        if _local(child.tag) in wanted and child.text and child.text.strip():
            return child.text.strip()
    return ""


def _child_link(node: Any) -> str:
    links: list[str] = []
    alternate = ""
    for child in node.iter():
        if _local(child.tag) != "link":
            continue
        href = child.get("href")
        if not href:
            continue
        rel = child.get("rel")
        if rel == "alternate":
            alternate = href
        elif rel in (None, ""):
            links.append(href)
        elif rel == "self" and not links and not alternate:
            links.append(href)
    return alternate or (links[0] if links else _child_text(node, "guid", "id", "link"))


class RSSConnector(IngestConnector):
    """Fetch RSS 2.0 / RDF / Atom entries from ``source.url``."""

    kind = SourceKind.RSS

    async def fetch(
        self, source: IngestSource, client: Any
    ) -> list[RawItem]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.9, */*;q=0.5",
        }
        text = await http_text(client, source.url, headers=headers)
        if not text:
            return []
        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError:
            return []
        return self._parse(root, source)

    def _parse(self, root: Any, source: IngestSource) -> list[RawItem]:
        items: list[RawItem] = []
        top = _local(root.tag)
        if top in ("rss", "rdf", "RDF"):
            for node in root.iter():
                if _local(node.tag) != "item":
                    continue
                item = self._rss_item(node, source)
                if item:
                    items.append(item)
        else:
            for node in root.iter():
                if _local(node.tag) != "entry":
                    continue
                item = self._atom_entry(node, source)
                if item:
                    items.append(item)
        return items

    def _rss_item(self, node: Any, source: IngestSource) -> Optional[RawItem]:
        title = _child_text(node, "title")
        link = _child_link(node) or _child_text(node, "guid")
        description = _child_text(node, "description", "encoded", "summary")
        published = parse_dt(_child_text(node, "pubDate", "date", "dc:date"))
        author = _child_text(node, "author", "dc:creator")
        if not title and not description:
            return None
        return RawItem(
            title=title or (description[:120] + "…" if description else source.name),
            url=link or source.url,
            content=description,
            summary=description,
            author=author,
            published_at=published,
            external_id=_child_text(node, "guid", "id") or link,
        )

    def _atom_entry(self, node: Any, source: IngestSource) -> Optional[RawItem]:
        title = _child_text(node, "title")
        link = _child_link(node)
        content = _child_text(node, "content", "summary")
        published = parse_dt(_child_text(node, "published", "updated"))
        author = _child_text(node, "name", "author")
        if not title and not content:
            return None
        return RawItem(
            title=title or (content[:120] + "…" if content else source.name),
            url=link or source.url,
            content=content,
            summary=content,
            author=author,
            published_at=published,
            external_id=_child_text(node, "id") or link,
        )
