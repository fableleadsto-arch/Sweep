"""The ingestion pipeline: fetch → sanitize → dedupe → score → extract → store.

Runs one source at a time, per-source isolated (a failure in one source never
affects another). Flow for each fetched item:

1. sanitize + size cap (+ prompt-injection screening)
2. exact-hash dedupe (global) + text near-dup (same source) + optional
   embedding near-dup
3. relevance filter (cheap deterministic scoring *before* any AI work)
4. store document + chunks
5. extract entities / claims / edges; resolve contradictions
6. embed chunks when an embedding provider is configured (cost control:
   deterministic pipeline first, embeddings only when available)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from ..config import BrainSettings
from ..embeddings import embed_batch
from . import dedupe
from .contradictions import resolve_contradictions
from .models import (
    IngestedChunk,
    IngestedDocument,
    IngestionError,
    IngestionRun,
    IngestSource,
    RawItem,
    RunStatus,
    SourceKind,
    utcnow,
)
from .scoring import authority, compute_confidence, score_freshness, score_quality, score_relevance
from .security import injection_signals, sanitize_content, strip_credentials
from .sources import build_connector
from .store import KnowledgeStore
from .text import chunk_text, count_tokens

logger = logging.getLogger(__name__)

NEAR_DUP_EMBED_THRESHOLD = 0.95


class IngestionPipeline:
    """Executes the full per-source ingestion flow against a store."""

    def __init__(self, store: KnowledgeStore, settings: BrainSettings) -> None:
        self.store = store
        self.settings = settings

    @property
    def _embed_enabled(self) -> bool:
        return bool(
            self.settings.ingest_embed_when_available
            and (self.settings.gemini_api_key or self.settings.openai_api_key)
        )

    async def run_source(
        self,
        source_id: str,
        *,
        client: Optional[httpx.AsyncClient] = None,
        embed: bool = True,
    ) -> Optional[IngestionRun]:
        """Ingest every new item from one source. Returns the run, or None."""
        source = await self.store.get_source(source_id)
        if source is None:
            return None
        if not source.enabled:
            return None

        run = IngestionRun(id=uuid.uuid4().hex, source_id=source.id)
        await self.store.record_run(run)
        source.last_checked = utcnow()
        await self.store.upsert_source(source)

        own_client = client is None
        client = client or httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        try:
            from .security import SSRFBlockedError, validate_outbound_url

            validate_outbound_url(source.url)  # SSRF guard at fetch time
            connector = build_connector(source, self.settings)
            if connector is None:
                run.status = RunStatus.FAILED
                run.error_message = "No connector available for this source kind"
                run.error_count = 1
                await self._record_failure(source, run, stage="registry", error_type="unknown_kind", message=run.error_message)
                return run
            items = await connector.fetch(source, client)
            run.items_found = len(items)
            await self.store.record_run(run)

            max_items = self.settings.ingest_max_documents_per_source
            for item in items[:max_items]:
                try:
                    accepted = await self._ingest_item(source, item, run, client, embed=embed)
                    if accepted == "added":
                        run.added += 1
                    elif accepted == "duplicate":
                        run.duplicates += 1
                    elif accepted == "updated":
                        run.updated += 1
                    elif accepted == "rejected":
                        run.rejected += 1
                except Exception as exc:  # noqa: BLE001 - one bad item never kills the run
                    run.error_count += 1
                    logger.warning("ingest item failed for %s: %s", source.id, exc)

            run.status = RunStatus.PARTIAL if run.error_count else RunStatus.SUCCESS
            source.last_successful_sync = utcnow()
            source.consecutive_failures = 0
            await self.store.upsert_source(source)
        except Exception as exc:  # noqa: BLE001
            run.status = RunStatus.FAILED
            run.error_count += 1
            run.error_message = str(exc)[:400]
            await self._record_failure(source, run, stage="fetch", error_type=type(exc).__name__, message=str(exc))
        finally:
            run.finished_at = utcnow()
            await self.store.record_run(run)
            if own_client:
                await client.aclose()
        return run

    async def _record_failure(
        self,
        source: IngestSource,
        run: IngestionRun,
        *,
        stage: str,
        error_type: str,
        message: str,
    ) -> None:
        source.error_count += 1
        source.consecutive_failures += 1
        await self.store.upsert_source(source)
        await self.store.record_error(
            IngestionError(
                id=uuid.uuid4().hex,
                source_id=source.id,
                stage=stage,
                error_type=error_type,
                message=str(message)[:400],
                retry_count=source.consecutive_failures,
            )
        )

    # ── per-item flow ───────────────────────────────────────────────────

    async def _ingest_item(
        self,
        source: IngestSource,
        item: RawItem,
        run: IngestionRun,
        client: httpx.AsyncClient,
        *,
        embed: bool,
    ) -> str:
        content = sanitize_content(item.content or item.summary or item.title)
        if not content.strip():
            return "rejected"

        signals = injection_signals(content)
        if signals:
            logger.info("rejecting %s — prompt-injection signals: %s", item.url, signals)
            return "rejected"

        content_hash = dedupe.content_hash(content)
        existing = await self.store.find_document_by_hash(content_hash, source_id=source.id)
        if existing is not None:
            return "duplicate"

        # Text near-duplicate against recent same-source documents.
        recent = await self.store.list_documents(source_id=source.id, limit=20)
        for prior in recent:
            if dedupe.is_near_duplicate(content, prior.content):
                return "duplicate"

        topics = [t for t in source.topics if t]
        relevance = score_relevance(content, topics)
        if topics and relevance < self.settings.ingest_min_relevance:
            return "rejected"

        quality = score_quality(item)
        freshness = score_freshness(item.published_at)
        source_authority = authority(source.kind, source.trust_score)
        confidence = compute_confidence(relevance, quality, source_authority, freshness)

        doc = IngestedDocument(
            id=uuid.uuid4().hex,
            source_id=source.id,
            source_kind=source.kind,
            name=item.title or source.name,
            content=content,
            url=strip_credentials(item.url),
            published_at=item.published_at,
            content_hash=content_hash,
            external_id=item.external_id,
            topics=topics,
            relevance=relevance,
            quality=quality,
            confidence=confidence,
            metadata={
                "author": item.author,
                "summary": item.summary[:500],
                "signals_clean": True,
            },
        )
        await self.store.save_document(doc)

        chunks = [self._chunk(doc, i, part) for i, part in enumerate(chunk_text(content))]
        await self.store.save_chunks(chunks)
        doc.chunk_count = len(chunks)
        await self.store.save_document(doc)

        await self._extract_and_store(doc, source_authority, confidence)

        if embed and self._embed_enabled and chunks:
            try:
                embeddings = await embed_batch([c.content[:6000] for c in chunks], self.settings, client)
                for chunk, vector in zip(chunks, embeddings):
                    if vector is not None:
                        await self.store.update_chunk_embedding(chunk.id, vector)
            except Exception as exc:  # noqa: BLE001 - embeddings must never fail a run
                logger.warning("embedding failed for %s: %s", doc.id, exc)

        return "added"

    def _chunk(self, doc: IngestedDocument, index: int, content: str) -> IngestedChunk:
        return IngestedChunk(
            id=uuid.uuid4().hex,
            document_id=doc.id,
            source_id=doc.source_id,
            chunk_index=index,
            content=content,
            tokens=count_tokens(content),
        )

    async def _extract_and_store(self, doc: IngestedDocument, source_authority: float, confidence: float) -> None:
        from .extract import extract_claims, extract_edges, extract_entities, extract_topics

        # Entities
        for name, kind, _count in extract_entities(doc.content):
            await self.store.upsert_entity(name, kind=kind, description=f"mentioned in {doc.name}")

        # Claims + contradictions
        claims = extract_claims(doc, source_authority=source_authority, confidence=confidence)
        stored: list[Any] = []
        for claim in claims:
            resolution = resolve_contradictions(claim, await self._existing_candidates(claim.entity, claim.property))
            claim.status = resolution.status
            await self.store.save_claims([claim])
            if resolution.superseded_ids:
                for sid in resolution.superseded_ids:
                    await self.store.set_claim_status(sid, resolution.status if resolution.status.value == "contradicted" else "superseded")
                    await self.store.increment_claim_contradictions(sid)
            if resolution.update:
                await self.store.record_update(resolution.update)
            stored.append(claim)

        # Edges
        edges = extract_edges(doc, confidence=confidence)
        if edges:
            await self.store.save_edges(edges)

    async def _existing_candidates(self, entity: str, prop: str) -> list[Any]:
        """Claims with the same entity+property (broad match for resolution)."""
        claims = await self.store.list_claims(entity=entity, limit=50)
        return [c for c in claims if c.property == prop]

    # ── convenience wrappers used by routes/brain ───────────────────────

    async def refresh_source(self, source_id: str) -> IngestionRun | None:
        """Synchronous one-shot run (on-demand refresh / trigger button)."""
        return await self.run_source(source_id)

    async def refresh_all(self) -> list[IngestionRun]:
        sources = await self.store.list_sources()
        runs: list[IngestionRun] = []
        for source in sources:
            if source.enabled:
                run = await self.run_source(source.id)
                if run is not None:
                    runs.append(run)
        return runs
