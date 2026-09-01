"""Content normalization — standardize documents from diverse sources."""

from __future__ import annotations

import hashlib
import re
from typing import Optional
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse


def canonicalize_url(url: str) -> str:
    """Normalize a URL for deduplication — strip tracking params, fragments, etc."""
    try:
        parsed = urlparse(url)
    except Exception:
        return url

    # Strip common tracking parameters
    TRACKING_PARAMS = {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "ref", "source", "fbclid", "gclid", "mc_cid", "mc_eid",
        "_ga", "_gl", "yclid", "msclkid",
    }

    params = parse_qs(parsed.query, keep_blank_values=False)
    cleaned = {k: v for k, v in params.items() if k.lower() not in TRACKING_PARAMS}
    new_query = urlencode(cleaned, doseq=True) if cleaned else ""

    # Strip trailing slash and fragment
    path = parsed.path.rstrip("/") or "/"

    return urlunparse((
        parsed.scheme,
        parsed.netloc.lower(),
        path,
        parsed.params,
        new_query,
        "",  # Strip fragment
    ))


def content_hash(text: str) -> str:
    """Short hash for deduplication."""
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def clean_content(text: str) -> str:
    """Clean extracted text — collapse whitespace, remove zero-width chars."""
    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class NormalizedDoc:
    """A normalized document from any source."""

    __slots__ = (
        "url", "title", "description", "markdown", "text",
        "source", "fetched_at", "content_hash",
    )

    def __init__(
        self,
        url: str,
        title: str = "",
        description: str = "",
        markdown: str = "",
        text: str = "",
        source: str = "",
        fetched_at: str = "",
    ):
        self.url = canonicalize_url(url)
        self.title = title
        self.description = description
        self.markdown = markdown
        self.text = text
        self.source = source
        self.fetched_at = fetched_at
        self.content_hash = content_hash(text or markdown)


def normalize_batch(docs: list[dict]) -> list[NormalizedDoc]:
    """Normalize a batch of documents."""
    results = []
    for doc in docs:
        results.append(NormalizedDoc(
            url=doc.get("url", ""),
            title=doc.get("title", ""),
            description=doc.get("description", ""),
            markdown=doc.get("markdown", ""),
            text=doc.get("text", ""),
            source=doc.get("source", ""),
            fetched_at=doc.get("fetched_at", ""),
        ))
    return results
