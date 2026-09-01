"""Source relevance scoring — scores how relevant/authoritative a source is."""

from __future__ import annotations

import re
from datetime import datetime

from ..core.types import Source, SourceScore


STOPWORDS = frozenset([
    "the", "and", "for", "with", "that", "this", "from", "have", "what", "when",
    "where", "about", "into", "over", "their", "they", "them", "there", "than",
    "then", "will", "would", "should", "could", "also", "were", "been", "being",
    "which", "while", "your", "our", "some", "more", "most", "other", "only",
])


def _objective_keywords(objective: str) -> list[str]:
    """Extract meaningful keywords from an objective."""
    return [
        word for word in re.split(r"[^a-z0-9]+", objective.lower())
        if len(word) > 3 and word not in STOPWORDS
    ][:12]


def _score_relevance(text: str, keywords: list[str]) -> float:
    """Score how relevant text is to the objective keywords."""
    if not keywords:
        return 0.5
    lower = text.lower()
    hits = sum(1 for kw in keywords if kw in lower)
    return min(1.0, hits / max(len(keywords) * 0.3, 1))


def _score_authority(url: str, title: str) -> float:
    """Score source authority based on URL patterns and title signals."""
    score = 0.5
    url_lower = url.lower()

    # High-authority patterns
    if any(d in url_lower for d in [".gov", ".edu", "wikipedia.org", "github.com"]):
        score += 0.3
    elif any(d in url_lower for d in [".org", "arxiv.org", "scholar.google"]):
        score += 0.2
    elif any(d in url_lower for d in [".com", ".io", ".co"]):
        score += 0.1

    # Authority signals in title
    title_lower = title.lower()
    if any(w in title_lower for w in ["official", "documentation", "docs", "reference"]):
        score += 0.15

    return min(1.0, score)


def _score_freshness(url: str, published_at: str | None) -> float:
    """Score how fresh/recent the source is."""
    if published_at:
        try:
            pub = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            age_days = (datetime.now(pub.tzinfo) - pub).days
            if age_days < 30:
                return 1.0
            elif age_days < 180:
                return 0.8
            elif age_days < 365:
                return 0.6
            elif age_days < 730:
                return 0.4
            return 0.2
        except Exception:
            pass
    return 0.5  # Unknown freshness


def _score_directness(url: str, title: str, objective: str) -> float:
    """Score how directly the source relates to the objective."""
    score = 0.3
    obj_lower = objective.lower()
    title_lower = title.lower()

    # Direct mention in title
    for word in obj_lower.split():
        if len(word) > 3 and word in title_lower:
            score += 0.2

    # Platform relevance
    if any(p in url.lower() for p in ["linkedin.com", "crunchbase.com", "pitchbook.com"]):
        score += 0.15

    return min(1.0, score)


def score_source(source: Source, objective: str) -> SourceScore:
    """Compute a full source score against the research objective."""
    keywords = _objective_keywords(objective)
    text = f"{source.title} {source.url}"

    relevance = _score_relevance(text, keywords)
    authority = _score_authority(source.url, source.title)
    freshness = _score_freshness(source.url, None)
    directness = _score_directness(source.url, source.title, objective)

    overall = relevance * 0.35 + authority * 0.25 + freshness * 0.15 + directness * 0.25

    return SourceScore(
        relevance=round(relevance, 3),
        authority=round(authority, 3),
        freshness=round(freshness, 3),
        directness=round(directness, 3),
        overall=round(overall, 3),
    )


def attach_scores(hits: list[dict], objective: str) -> list[dict]:
    """Attach scores to search result hits."""
    scored = []
    for hit in hits:
        source = Source(
            title=hit.get("title", ""),
            url=hit.get("url", ""),
            access_mode="public",
        )
        score = score_source(source, objective)
        hit["score"] = score.model_dump()
        scored.append(hit)
    return scored
