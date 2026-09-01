"""Brain integration — make ingested knowledge immediately usable.

Two hooks:

* ``build_ingest_knowledge_context`` — a prompt-ready block for the brain turn
  combining vector-retrieved ingest chunks (Supabase) with structured claims /
  entities matched by simple term overlap (works even on the local store).
* ``maybe_refresh`` — on-demand refresh: when a query mentions a source's
  topic and that source is stale, schedule a background re-ingest (cooldown
  enforced by the scheduler).

External content is labeled and must be treated as untrusted data by the model
(consistent with the agent loop's ``[UNTRUSTED DATA]`` convention).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from ..config import BrainSettings
from ..rag import RagService
from .scheduler import get_scheduler
from .store import KnowledgeStore

logger = logging.getLogger(__name__)

TERM_SPLIT = re.compile(r"[a-z0-9]+")

# Words that never name an entity — skipped before claim lookups.
_STOPWORDS = {
    "what", "when", "where", "who", "which", "why", "how", "the", "and", "are",
    "was", "were", "for", "with", "that", "this", "is", "of", "latest",
    "current", "version", "versions", "release", "releases", "tell", "about",
}


def _query_terms(query: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for token in TERM_SPLIT.findall(query.lower()):
        if len(token) < 3 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms[:5]


async def build_ingest_knowledge_context(
    query: str,
    settings: BrainSettings,
    store: Optional[KnowledgeStore] = None,
    rag: Optional[RagService] = None,
) -> str:
    """Assemble the continuous-knowledge context block for one query."""
    terms = _query_terms(query)
    blocks: list[str] = []

    if rag is not None:
        try:
            result = await rag.ingest_retrieve(query, max_chunks=3)
            if result.found and result.context:
                blocks.append(result.context)
        except Exception as exc:  # noqa: BLE001 - best-effort
            logger.debug("ingest vector retrieval failed: %s", exc)

    if store is not None and terms:
        try:
            claims = []
            for term in terms:
                for claim in await store.list_claims(entity=term, limit=6):
                    if claim not in claims:
                        claims.append(claim)
                    if len(claims) >= 10:
                        break
                if len(claims) >= 10:
                    break
            if claims:
                lines = ["Ingested knowledge claims (source-attributed):"]
                for claim in claims[:8]:
                    status = "" if claim.status.value == "active" else f" [{claim.status.value}]"
                    lines.append(
                        f"- {claim.entity}: {claim.property} = {claim.value}"
                        f"{status} (confidence {claim.confidence:.2f})"
                    )
                    if claim.source_url:
                        lines.append(f"  Source: {claim.source_url}")
                blocks.append("\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            logger.debug("ingest claim lookup failed: %s", exc)

    if not blocks:
        return ""
    header = "Continuous knowledge (ingested from external sources):"
    return "\n\n".join([header, *blocks])


async def maybe_refresh(query: str, settings: BrainSettings) -> bool:
    """Trigger a stale source whose configured topic the query mentions.

    Cooldown-guarded by the scheduler (one manual trigger per source per
    minute) so ordinary chat can't hammer a source. Returns True when a
    refresh was scheduled. Never raises — a failed trigger simply means the
    next scheduled run will pick it up.
    """
    if not settings.enable_knowledge_ingestion:
        return False
    try:
        store = KnowledgeStore(settings)
        sources = await store.list_sources()
        query_lower = query.lower()
        scheduler = get_scheduler(settings)
        for source in sources:
            if not source.enabled:
                continue
            if any(t and t in query_lower for t in source.topics):
                triggered = await scheduler.trigger(source.id)
                if triggered:
                    return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("on-demand refresh skipped: %s", exc)
    return False
