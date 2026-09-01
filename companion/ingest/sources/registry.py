"""Source-connector dispatch + default source templates.

``build_connector`` maps a source kind to the matching connector. Default
templates seed the registry with sensible trust scores / cadences so a source
created through the API or dashboard starts with honest, conservative values.
"""

from __future__ import annotations

import uuid
from datetime import timezone
from typing import Any, Optional

from ...config import BrainSettings
from ..models import (
    FREQUENCY_SECONDS,
    CrawlFrequency,
    IngestSource,
    SourceKind,
    utcnow,
)
from .arxiv import ArxivConnector
from .base import IngestConnector
from .crossref import CrossrefConnector
from .github import GitHubConnector
from .huggingface import HuggingFaceConnector
from .openalex import OpenAlexConnector
from .rss import RSSConnector
from .web import WebConnector
from .wikipedia import WikipediaConnector

CONNECTORS: dict[SourceKind, type[IngestConnector]] = {
    SourceKind.RSS: RSSConnector,
    SourceKind.GITHUB: GitHubConnector,
    SourceKind.ARXIV: ArxivConnector,
    SourceKind.OPENALEX: OpenAlexConnector,
    SourceKind.CROSSREF: CrossrefConnector,
    SourceKind.WIKIPEDIA: WikipediaConnector,
    SourceKind.HUGGINGFACE: HuggingFaceConnector,
    SourceKind.WEB: WebConnector,
}


def build_connector(source: IngestSource, settings: BrainSettings) -> Optional[IngestConnector]:
    """Return the connector for a source, or None for unknown kinds."""
    connector_type = CONNECTORS.get(source.kind)
    if connector_type is None:
        return None
    connector = connector_type(settings)
    if not connector.can_handle(source):
        return None
    return connector


def list_connector_kinds() -> list[dict[str, Any]]:
    """Catalog of supported source kinds for the dashboard/API."""
    return [
        {
            "kind": kind.value,
            "label": _KIND_LABELS.get(kind, kind.value),
            "default_url": _DEFAULTS.get(kind, {}).get("url", ""),
            "default_frequency": _DEFAULTS.get(kind, {}).get("frequency", "daily"),
            "trust": _DEFAULTS.get(kind, {}).get("trust", 0.5),
            "description": _KIND_DESCRIPTIONS.get(kind, ""),
        }
        for kind in (
            SourceKind.RSS,
            SourceKind.GITHUB,
            SourceKind.ARXIV,
            SourceKind.OPENALEX,
            SourceKind.CROSSREF,
            SourceKind.WIKIPEDIA,
            SourceKind.HUGGINGFACE,
            SourceKind.WEB,
        )
    ]


_KIND_LABELS: dict[SourceKind, str] = {
    SourceKind.RSS: "RSS / Atom feed",
    SourceKind.GITHUB: "GitHub repository",
    SourceKind.ARXIV: "arXiv",
    SourceKind.OPENALEX: "OpenAlex",
    SourceKind.CROSSREF: "Crossref",
    SourceKind.WIKIPEDIA: "Wikipedia",
    SourceKind.HUGGINGFACE: "Hugging Face",
    SourceKind.WEB: "Web page",
}

_KIND_DESCRIPTIONS: dict[SourceKind, str] = {
    SourceKind.RSS: "Any RSS 2.0 / RDF / Atom feed URL.",
    SourceKind.GITHUB: "A GitHub repo (metadata, README, releases, changelog).",
    SourceKind.ARXIV: "An arXiv listing URL or a free-text paper search.",
    SourceKind.OPENALEX: "An OpenAlex works query or free-text search.",
    SourceKind.CROSSREF: "A Crossref works query or bibliographic search.",
    SourceKind.WIKIPEDIA: "A Wikipedia article URL or a topic search.",
    SourceKind.HUGGINGFACE: "A model/dataset page or a search query.",
    SourceKind.WEB: "Any public web page (robots.txt respected).",
}

# Conservative defaults: peer-reviewed sources get higher trust and slower
# cadence; volatile feeds get fast cadence and lower trust.
_DEFAULTS: dict[SourceKind, dict[str, Any]] = {
    SourceKind.RSS: {"trust": 0.5, "frequency": "15m", "url": "https://example.com/feed.xml"},
    SourceKind.GITHUB: {"trust": 0.8, "frequency": "6h", "url": "https://github.com/owner/repo"},
    SourceKind.ARXIV: {"trust": 0.9, "frequency": "daily", "url": "https://arxiv.org/list/cs.AI/recent"},
    SourceKind.OPENALEX: {"trust": 0.85, "frequency": "weekly", "url": "machine learning"},
    SourceKind.CROSSREF: {"trust": 0.85, "frequency": "daily", "url": "https://api.crossref.org/works"},
    SourceKind.WIKIPEDIA: {"trust": 0.75, "frequency": "daily", "url": "https://en.wikipedia.org/wiki/Artificial_intelligence"},
    SourceKind.HUGGINGFACE: {"trust": 0.8, "frequency": "daily", "url": "https://huggingface.co/models"},
    SourceKind.WEB: {"trust": 0.4, "frequency": "daily", "url": "https://example.com/article"},
}

TRUST_FLOOR: dict[SourceKind, float] = {kind: d["trust"] for kind, d in _DEFAULTS.items()}


def default_source(
    *,
    kind: SourceKind,
    name: str,
    url: str,
    topics: Optional[list[str]] = None,
    category: str = "",
    priority: int = 5,
    trust_score: Optional[float] = None,
    frequency: Optional[CrawlFrequency] = None,
    config: Optional[dict[str, Any]] = None,
) -> IngestSource:
    """Build an IngestSource with kind-appropriate conservative defaults."""
    defaults = _DEFAULTS.get(kind, {})
    trust = trust_score if trust_score is not None else float(defaults.get("trust", 0.5))
    freq = frequency or CrawlFrequency(defaults.get("frequency", "daily"))
    now = utcnow()
    return IngestSource(
        id=uuid.uuid4().hex,
        kind=kind,
        name=name.strip(),
        url=url.strip(),
        topics=[t.strip() for t in (topics or []) if t and t.strip()],
        category=category,
        priority=max(0, min(10, int(priority))),
        trust_score=max(0.0, min(1.0, trust)),
        crawl_frequency=freq,
        enabled=True,
        created_at=now,
        updated_at=now,
        config=config or {},
    )
