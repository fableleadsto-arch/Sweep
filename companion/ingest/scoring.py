"""Deterministic scoring for ingested documents.

Every score is a pure function of the content/attributes — no LLM required —
so the pipeline can cheaply filter before any AI call happens (cost control,
spec: "cheap deterministic filtering first").

Score semantics (all 0.0–1.0, higher is better):

* **relevance** — how many of the source's topics the document actually covers.
* **quality** — structural signal: reasonable length, not shouty, has an
  author/title/date, no link-spam density.
* **freshness** — exponential recency decay over a half-life.
* **authority** — base trust of the source kind × source trust_score.
* **confidence** — weighted combination used to rank claims/contexts.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

from .models import SourceKind, utcnow

# Base authority for each connector kind (0.0–1.0). Peer-reviewed and
# first-party sources outrank personal feeds and arbitrary web pages.
KIND_TRUST: dict[SourceKind, float] = {
    SourceKind.RSS: 0.5,
    SourceKind.GITHUB: 0.8,
    SourceKind.ARXIV: 0.9,
    SourceKind.OPENALEX: 0.85,
    SourceKind.CROSSREF: 0.85,
    SourceKind.WIKIPEDIA: 0.75,
    SourceKind.HUGGINGFACE: 0.8,
    SourceKind.WEB: 0.4,
    SourceKind.GENERIC_API: 0.5,
}

_WORD = re.compile(r"[A-Za-z0-9]+")
_UPPER_RUN = re.compile(r"[A-Z]{4,}")
_EXCLAMATION = re.compile(r"!")
_HTTP = re.compile(r"https?://")


def _words(text: str) -> int:
    return len(_WORD.findall(text or ""))


def topic_overlap(text: str, topics: list[str]) -> float:
    """Fraction of the source's topics mentioned in the document (0.0–1.0)."""
    if not topics:
        return 0.0
    lowered = (text or "").lower()
    hits = sum(1 for t in topics if t and t.lower() in lowered)
    return hits / len(topics)


def score_relevance(text: str, topics: list[str], *, boost: float = 0.5) -> float:
    """Relevance is topic coverage blended with a keyword-density boost."""
    coverage = topic_overlap(text, topics)
    if not topics or coverage <= 0:
        return 0.0
    density = min(1.0, _words(text) / 8000.0)
    return min(1.0, coverage * (1.0 - boost) + density * boost)


def score_quality(
    document: object,
    *,
    min_chars: int = 200,
    max_chars: int = 60_000,
) -> float:
    """Structural quality: length band, no shouting, low link-spam ratio."""
    content = getattr(document, "content", "") or ""
    chars = len(content)
    if chars < min_chars:
        return 0.0
    if chars > max_chars:
        return 0.6
    length_score = 0.3 + 0.5 * (min(chars, 6000) / 6000.0)
    words = max(1, _words(content))
    upper_runs = len(_UPPER_RUN.findall(content))
    shout_penalty = min(0.2, upper_runs * 0.02)
    links = len(_HTTP.findall(content))
    spam_penalty = min(0.25, (links / words) * 40.0)
    if getattr(document, "title", None):
        length_score += 0.1
    if getattr(document, "published_at", None):
        length_score += 0.1
    return max(0.0, min(1.0, length_score - shout_penalty - spam_penalty))


def score_freshness(
    published_at: Optional[datetime],
    *,
    now: Optional[datetime] = None,
    half_life_days: float = 120.0,
) -> float:
    """Recency decay: 1.0 at publish, 0.5 after one half-life."""
    if published_at is None:
        return 0.5  # no date → neutral, never 1.0
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    now = now or utcnow()
    age = max(0.0, (now - published_at).total_seconds())
    half_life_s = half_life_days * 24 * 3600
    return math.exp(-age / half_life_s * math.log(2))


def authority(kind: SourceKind, trust_score: float = 0.5) -> float:
    """Combined source authority for a document of a given kind."""
    base = KIND_TRUST.get(kind, 0.5)
    return min(1.0, base * (0.5 + 0.5 * trust_score))


def compute_confidence(
    relevance: float,
    quality: float,
    source_authority: float,
    freshness: float,
    *,
    weights: Optional[tuple[float, float, float, float]] = None,
) -> float:
    """Weighted confidence for a document / claim derived from it."""
    w_r, w_q, w_a, w_f = weights or (0.35, 0.25, 0.25, 0.15)
    return min(1.0, max(0.0, (w_r * relevance + w_q * quality + w_a * source_authority + w_f * freshness)))


def minutes_since(value: Optional[datetime], now: Optional[datetime] = None) -> float:
    """Whole minutes since a timestamp (0 when missing) — used by the scheduler."""
    if value is None:
        return float("inf")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    now = now or utcnow()
    return (now - value).total_seconds() / 60.0
