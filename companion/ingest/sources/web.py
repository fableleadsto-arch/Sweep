"""Generic web-page connector — single page → clean text via BeautifulSoup.

Politeness: a minimal robots.txt parser honors ``User-agent`` / ``Disallow``
rules before fetching a page, and per-domain results are cached for the run.
"""

from __future__ import annotations

import re
from typing import Any, Optional
from urllib.parse import urlparse

from ..models import IngestSource, RawItem, SourceKind
from .base import IngestConnector, http_text

_TEXT_MAX = 60_000


class _RobotsPolicy:
    """Tiny robots.txt checker (User-agent / Disallow only, per-run cache)."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[str, list[str]]] = {}

    async def allowed(self, client: Any, url: str, user_agent: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._cache:
            text = await http_text(
                client, f"{base}/robots.txt", headers={"User-Agent": user_agent}, timeout=10.0
            )
            self._cache[base] = (user_agent, _parse_disallows(text, user_agent))
        _, disallows = self._cache[base]
        path = parsed.path or "/"
        for rule in disallows:
            if path.startswith(rule):
                return False
        return True


def _parse_disallows(text: str, user_agent: str) -> list[str]:
    if not text:
        return []
    lines = text.splitlines()
    groups: list[tuple[list[str], list[str]]] = []
    agents: list[str] = []
    rules: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("user-agent:"):
            if agents:
                groups.append((agents, rules))
            agents = [line.split(":", 1)[1].strip().lower()]
            rules = []
        elif line.lower().startswith("disallow:"):
            value = line.split(":", 1)[1].strip()
            if value:
                rules.append(value)
    if agents:
        groups.append((agents, rules))
    ua = user_agent.lower()
    matched: list[str] = []
    for agents_list, rules_list in groups:
        if any(ua.startswith(a) for a in agents_list) or "*" in agents_list:
            matched.extend(rules_list)
    return matched or []


class WebConnector(IngestConnector):
    kind = SourceKind.WEB
    _robots = _RobotsPolicy()

    async def fetch(
        self, source: IngestSource, client: Any
    ) -> list[RawItem]:
        if not await self._robots.allowed(client, source.url, self.user_agent):
            return []
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        }
        html = await http_text(client, source.url, headers=headers)
        if not html:
            return []
        page = self._extract(html, source.url)
        if not page["text"]:
            return []
        return [
            RawItem(
                title=page["title"] or source.name,
                url=page["canonical"] or source.url,
                content=page["text"][:_TEXT_MAX],
                summary=page["description"],
                external_id=page["canonical"] or source.url,
            )
        ]

    def _extract(self, html: str, fallback_url: str) -> dict[str, str]:
        from bs4 import BeautifulSoup  # lazy: only when a web source is fetched

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:  # noqa: BLE001 - fall back to the stdlib parser
            soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe", "nav", "footer", "header", "form"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else ""
        description = ""
        meta = soup.find("meta", attrs={"name": re.compile("description", re.IGNORECASE)})
        if meta and meta.get("content"):
            description = str(meta["content"]).strip()
        canonical = ""
        link = soup.find("link", attrs={"rel": re.compile("canonical", re.IGNORECASE)})
        if link and link.get("href"):
            canonical = str(link["href"])
        container = soup.find("article") or soup.find("main") or soup.body
        text = container.get_text(separator="\n", strip=True) if container else ""
        text = re.sub(r"\n{3,}", "\n\n", text)
        return {"title": title, "description": description, "canonical": canonical, "text": text}
