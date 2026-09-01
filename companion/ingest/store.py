"""Knowledge store for the ingestion engine.

Two backends behind one async facade:

* **``local``** — JSON files under ``.relayhub/ingest/``. The dev fallback,
  mirroring the FileMemoryStore pattern: fully functional without Supabase,
  used when Supabase is not configured.
* **``supabase``** — PostgREST writes into the global ingestion tables
  (``ingest_documents``, ``ingest_chunks``, ``knowledge_sources``,
  ``ingestion_runs``, ``ingestion_errors``, ``knowledge_claims``,
  ``knowledge_entities``, ``knowledge_entity_edges``, ``knowledge_updates``).
  pgvector embeddings are stored as JSON arrays cast to ``vector(768)``.

Mode is chosen from ``ingest_store``: ``"auto"`` uses Supabase when the
service key is configured, otherwise local.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx

from ..config import BrainSettings
from .models import (
    CrawlFrequency,
    EntityEdge,
    IngestStats,
    IngestedChunk,
    IngestedDocument,
    IngestionError,
    IngestionRun,
    IngestSource,
    KnowledgeClaim,
    KnowledgeEntity,
    KnowledgeUpdate,
    RunStatus,
    SourceKind,
    from_dict,
    to_dict,
)

# ─────────────────────────────────────────────────────────────────────────
#  Local file store
# ─────────────────────────────────────────────────────────────────────────


class LocalFileStore:
    """JSON-file persistence (dev/fallback). Thread-safe via an asyncio lock."""

    FILES = (
        "sources.json",
        "documents.json",
        "chunks.json",
        "claims.json",
        "entities.json",
        "edges.json",
        "updates.json",
        "runs.json",
        "errors.json",
    )

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._cache: dict[str, list[dict[str, Any]]] = {}

    # ── low-level IO ────────────────────────────────────────────────────

    def _path(self, name: str) -> Path:
        return self.data_dir / name

    async def _load(self, name: str) -> list[dict[str, Any]]:
        async with self._lock:
            if name in self._cache:
                return self._cache[name]
            path = self._path(name)
            if not path.exists():
                return []
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rows = data if isinstance(data, list) else []
            except (json.JSONDecodeError, OSError):
                rows = []
            self._cache[name] = rows
            return rows

    async def _save(self, name: str, rows: list[dict[str, Any]]) -> None:
        async with self._lock:
            self._cache[name] = rows
            path = self._path(name)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
            os.replace(tmp, path)

    async def _add(self, name: str, row: dict[str, Any]) -> None:
        rows = await self._load(name)
        rows.append(row)
        await self._save(name, rows)

    async def _replace(self, name: str, match_key: str, row: dict[str, Any]) -> None:
        rows = await self._load(name)
        rows = [row if item.get(match_key) == row.get(match_key) else item for item in rows]
        if not any(item.get(match_key) == row.get(match_key) for item in rows):
            rows.append(row)
        await self._save(name, rows)

    async def _remove(self, name: str, match_key: str, value: Any) -> None:
        rows = await self._load(name)
        await self._save(name, [item for item in rows if item.get(match_key) != value])

    # ── sources ─────────────────────────────────────────────────────────

    async def list_sources(self) -> list[IngestSource]:
        return [from_dict(IngestSource, r) for r in await self._load("sources.json")]

    async def get_source(self, source_id: str) -> Optional[IngestSource]:
        for row in await self._load("sources.json"):
            if row.get("id") == source_id:
                return from_dict(IngestSource, row)
        return None

    async def upsert_source(self, source: IngestSource) -> IngestSource:
        await self._replace("sources.json", "id", to_dict(source))
        return source

    async def delete_source(self, source_id: str) -> bool:
        before = await self.get_source(source_id)
        await self._remove("sources.json", "id", source_id)
        return before is not None

    # ── runs / errors ───────────────────────────────────────────────────

    async def record_run(self, run: IngestionRun) -> IngestionRun:
        await self._replace("runs.json", "id", to_dict(run))
        return run

    async def list_runs(self, source_id: Optional[str] = None, limit: int = 20) -> list[IngestionRun]:
        rows = await self._load("runs.json")
        if source_id:
            rows = [r for r in rows if r.get("source_id") == source_id]
        rows = sorted(rows, key=lambda r: r.get("started_at", ""), reverse=True)[:limit]
        return [from_dict(IngestionRun, r) for r in rows]

    async def record_error(self, error: IngestionError) -> IngestionError:
        await self._add("errors.json", to_dict(error))
        return error

    async def list_errors(self, source_id: Optional[str] = None, limit: int = 20) -> list[IngestionError]:
        rows = await self._load("errors.json")
        if source_id:
            rows = [r for r in rows if r.get("source_id") == source_id]
        rows = sorted(rows, key=lambda r: r.get("occurred_at", ""), reverse=True)[:limit]
        return [from_dict(IngestionError, r) for r in rows]

    # ── documents ───────────────────────────────────────────────────────

    async def find_document_by_hash(self, content_hash: str, source_id: Optional[str] = None) -> Optional[IngestedDocument]:
        for row in await self._load("documents.json"):
            if row.get("content_hash") != content_hash:
                continue
            if source_id and row.get("source_id") != source_id:
                continue
            return from_dict(IngestedDocument, row)
        return None

    async def find_document_by_url(self, url: str) -> Optional[IngestedDocument]:
        for row in await self._load("documents.json"):
            if row.get("url") == url:
                return from_dict(IngestedDocument, row)
        return None

    async def save_document(self, doc: IngestedDocument) -> IngestedDocument:
        await self._replace("documents.json", "id", to_dict(doc))
        return doc

    async def list_documents(
        self,
        source_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[IngestedDocument]:
        rows = await self._load("documents.json")
        if source_id:
            rows = [r for r in rows if r.get("source_id") == source_id]
        rows = sorted(rows, key=lambda r: r.get("ingested_at", ""), reverse=True)
        return [from_dict(IngestedDocument, r) for r in rows[offset : offset + limit]]

    async def get_document(self, doc_id: str) -> Optional[IngestedDocument]:
        for row in await self._load("documents.json"):
            if row.get("id") == doc_id:
                return from_dict(IngestedDocument, row)
        return None

    async def count_documents(self) -> int:
        return len(await self._load("documents.json"))

    async def delete_document(self, doc_id: str) -> bool:
        before = await self.get_document(doc_id)
        if before is None:
            return False
        await self._remove("documents.json", "id", doc_id)
        await self._remove("chunks.json", "document_id", doc_id)
        await self._remove("claims.json", "document_id", doc_id)
        return True

    async def recent_content_hashes(self, limit: int = 300) -> set[str]:
        rows = await self._load("documents.json")
        return {str(r.get("content_hash", "")) for r in rows[:limit] if r.get("content_hash")}

    # ── chunks ──────────────────────────────────────────────────────────

    async def save_chunk(self, chunk: IngestedChunk) -> IngestedChunk:
        await self._replace("chunks.json", "id", to_dict(chunk))
        return chunk

    async def save_chunks(self, chunks: Iterable[IngestedChunk]) -> int:
        saved = 0
        for chunk in chunks:
            await self.save_chunk(chunk)
            saved += 1
        return saved

    async def update_chunk_embedding(self, chunk_id: str, embedding: list[float]) -> bool:
        rows = await self._load("chunks.json")
        for row in rows:
            if row.get("id") == chunk_id:
                row["embedding"] = embedding
                await self._save("chunks.json", rows)
                return True
        return False

    async def chunks_for_document(self, doc_id: str) -> list[IngestedChunk]:
        rows = await self._load("chunks.json")
        rows = [r for r in rows if r.get("document_id") == doc_id]
        rows = sorted(rows, key=lambda r: r.get("chunk_index", 0))
        return [from_dict(IngestedChunk, r) for r in rows]

    async def chunk_samples(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = await self._load("chunks.json")
        return [r for r in rows[:limit]]

    async def count_chunks(self) -> int:
        return len(await self._load("chunks.json"))

    # ── claims / entities / edges / updates ─────────────────────────────

    async def save_claims(self, claims: Iterable[KnowledgeClaim]) -> int:
        saved = 0
        for claim in claims:
            await self._replace("claims.json", "id", to_dict(claim))
            saved += 1
        return saved

    async def list_claims(self, entity: Optional[str] = None, limit: int = 50) -> list[KnowledgeClaim]:
        rows = await self._load("claims.json")
        if entity:
            rows = [r for r in rows if r.get("entity", "").lower() == entity.lower()]
        rows = sorted(rows, key=lambda r: r.get("collected_at", ""), reverse=True)
        return [from_dict(KnowledgeClaim, r) for r in rows[:limit]]

    async def claims_for_document(self, doc_id: str) -> list[KnowledgeClaim]:
        rows = await self._load("claims.json")
        rows = [r for r in rows if r.get("document_id") == doc_id]
        return [from_dict(KnowledgeClaim, r) for r in rows]

    async def set_claim_status(self, claim_id: str, status: Any) -> bool:
        rows = await self._load("claims.json")
        changed = False
        for row in rows:
            if row.get("id") == claim_id:
                row["status"] = status.value if hasattr(status, "value") else str(status)
                changed = True
        if changed:
            await self._save("claims.json", rows)
        return changed

    async def increment_claim_contradictions(self, claim_id: str) -> bool:
        rows = await self._load("claims.json")
        changed = False
        for row in rows:
            if row.get("id") == claim_id:
                row["contradictions"] = int(row.get("contradictions", 0)) + 1
                changed = True
        if changed:
            await self._save("claims.json", rows)
        return changed

    async def upsert_entity(self, name: str, kind: str = "", description: str = "") -> KnowledgeEntity:
        now = datetime.now(timezone.utc)
        rows = await self._load("entities.json")
        for row in rows:
            if row.get("name", "").lower() == name.lower():
                row["mention_count"] = int(row.get("mention_count", 1)) + 1
                row["last_seen_at"] = now.isoformat()
                if kind and not row.get("kind"):
                    row["kind"] = kind
                await self._save("entities.json", rows)
                return from_dict(KnowledgeEntity, row)
        entity = KnowledgeEntity(id=uuid.uuid4().hex, name=name, kind=kind, description=description)
        rows.append(to_dict(entity))
        await self._save("entities.json", rows)
        return entity

    async def list_entities(self, limit: int = 20) -> list[KnowledgeEntity]:
        rows = await self._load("entities.json")
        rows = sorted(rows, key=lambda r: r.get("mention_count", 0), reverse=True)
        return [from_dict(KnowledgeEntity, r) for r in rows[:limit]]

    async def save_edges(self, edges: Iterable[EntityEdge]) -> int:
        saved = 0
        for edge in edges:
            await self._replace("edges.json", "id", to_dict(edge))
            saved += 1
        return saved

    async def record_update(self, update: KnowledgeUpdate) -> KnowledgeUpdate:
        await self._add("updates.json", to_dict(update))
        return update

    async def list_updates(self, limit: int = 20) -> list[KnowledgeUpdate]:
        rows = await self._load("updates.json")
        rows = sorted(rows, key=lambda r: r.get("detected_at", ""), reverse=True)
        return [from_dict(KnowledgeUpdate, r) for r in rows[:limit]]

    # ── stats / health ──────────────────────────────────────────────────

    async def stats(self) -> IngestStats:
        now = datetime.now(timezone.utc)
        sources = await self.list_sources()
        runs = await self._load("runs.json")
        errors = await self._load("errors.json")
        added_24h = sum(
            1
            for r in runs
            if r.get("status") in (RunStatus.SUCCESS.value, RunStatus.PARTIAL.value)
            and r.get("started_at", "")
            and _dt(r.get("started_at")) and _dt(r.get("started_at")) >= now - timedelta(hours=24)
            for _ in [None]
        )
        last_success = max(
            (
                _dt(r.get("started_at"))
                for r in runs
                if r.get("status") in (RunStatus.SUCCESS.value, RunStatus.PARTIAL.value)
            ),
            default=None,
        )
        return IngestStats(
            source_count=len(sources),
            enabled_count=sum(1 for s in sources if s.enabled),
            document_count=await self.count_documents(),
            chunk_count=await self.count_chunks(),
            claim_count=len(await self._load("claims.json")),
            entity_count=len(await self._load("entities.json")),
            update_count=len(await self._load("updates.json")),
            run_count=len(runs),
            error_count=len(errors),
            added_24h=added_24h,
            last_successful_sync=last_success,
            store="local",
        )

    async def health(self) -> dict[str, Any]:
        return {"store": "local", "data_dir": str(self.data_dir), "ok": True}


def _dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# ─────────────────────────────────────────────────────────────────────────
#  Supabase REST store
# ─────────────────────────────────────────────────────────────────────────


class SupabaseStore:
    """PostgREST-backed store over the global ingestion tables."""

    TABLE_SOURCES = "knowledge_sources"
    TABLE_DOCUMENTS = "ingest_documents"
    TABLE_CHUNKS = "ingest_chunks"
    TABLE_RUNS = "ingestion_runs"
    TABLE_ERRORS = "ingestion_errors"
    TABLE_CLAIMS = "knowledge_claims"
    TABLE_ENTITIES = "knowledge_entities"
    TABLE_EDGES = "knowledge_entity_edges"
    TABLE_UPDATES = "knowledge_updates"

    def __init__(self, settings: BrainSettings) -> None:
        self.settings = settings

    @property
    def _base(self) -> str:
        return f"{self.settings.supabase_url}/rest/v1"

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self.settings.supabase_key,
            "Authorization": f"Bearer {self.settings.supabase_key}",
            "Content-Type": "application/json",
        }

    def _row(self, model: Any) -> dict[str, Any]:
        data = to_dict(model)
        if isinstance(data.get("crawl_frequency"), str) and hasattr(model, "crawl_frequency"):
            data["crawl_frequency"] = model.crawl_frequency.value
        if isinstance(data.get("kind"), str) and hasattr(model, "kind"):
            data["kind"] = model.kind.value
        return data

    async def _select(self, table: str, *, filters: dict[str, Any] | None = None, limit: int = 50, order: Optional[str] = None) -> list[dict[str, Any]]:
        params: list[tuple[str, str]] = [("select", "*"), ("limit", str(limit))]
        if order:
            params.append(("order", order))
        if filters:
            for key, value in filters.items():
                params.append((key, f"eq.{value}"))
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{self._base}/{table}", headers=self._headers, params=params)
            if resp.is_success:
                data = resp.json()
                return data if isinstance(data, list) else []
            return []

    async def _insert(self, table: str, rows: list[dict[str, Any]]) -> bool:
        if not rows:
            return True
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base}/{table}",
                headers={
                    **self._headers,
                    "Prefer": "resolution=merge-duplicates,return=minimal",
                },
                json=rows,
            )
            return resp.is_success

    async def _update_row(self, table: str, match: dict[str, Any], fields: dict[str, Any]) -> bool:
        filters = "&".join(f"{k}=eq.{v}" for k, v in match.items())
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.patch(
                f"{self._base}/{table}?{filters}",
                headers=self._headers,
                json=fields,
            )
            return resp.is_success

    async def _count(self, table: str) -> int:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self._base}/{table}",
                headers={**self._headers, "Prefer": "count=exact", "Range": "0-0"},
                params={"select": "id"},
            )
            if resp.is_success:
                content_range = resp.headers.get("content-range", "")
                if "/" in content_range:
                    try:
                        return int(content_range.split("/")[1])
                    except ValueError:
                        pass
            return 0

    # ── sources ─────────────────────────────────────────────────────────

    async def list_sources(self) -> list[IngestSource]:
        rows = await self._select(self.TABLE_SOURCES, limit=500, order="priority.desc")
        return [_source_from_row(r) for r in rows]

    async def get_source(self, source_id: str) -> Optional[IngestSource]:
        rows = await self._select(self.TABLE_SOURCES, filters={"id": source_id}, limit=1)
        return _source_from_row(rows[0]) if rows else None

    async def upsert_source(self, source: IngestSource) -> IngestSource:
        row = self._row(source)
        row["crawl_frequency"] = source.crawl_frequency.value
        row["kind"] = source.kind.value
        await self._insert(self.TABLE_SOURCES, [row])
        return source

    async def delete_source(self, source_id: str) -> bool:
        existing = await self.get_source(source_id)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.delete(
                f"{self._base}/{self.TABLE_SOURCES}?id=eq.{source_id}",
                headers=self._headers,
            )
        return existing is not None

    # ── runs / errors ───────────────────────────────────────────────────

    async def record_run(self, run: IngestionRun) -> IngestionRun:
        await self._insert(self.TABLE_RUNS, [self._row(run)])
        return run

    async def list_runs(self, source_id: Optional[str] = None, limit: int = 20) -> list[IngestionRun]:
        filters = {"source_id": source_id} if source_id else None
        rows = await self._select(self.TABLE_RUNS, filters=filters, limit=limit, order="started_at.desc")
        return [from_dict(IngestionRun, _lower_keys(r)) for r in rows]

    async def record_error(self, error: IngestionError) -> IngestionError:
        await self._insert(self.TABLE_ERRORS, [self._row(error)])
        return error

    async def list_errors(self, source_id: Optional[str] = None, limit: int = 20) -> list[IngestionError]:
        filters = {"source_id": source_id} if source_id else None
        rows = await self._select(self.TABLE_ERRORS, filters=filters, limit=limit, order="occurred_at.desc")
        return [from_dict(IngestionError, _lower_keys(r)) for r in rows]

    # ── documents ───────────────────────────────────────────────────────

    async def find_document_by_hash(self, content_hash: str, source_id: Optional[str] = None) -> Optional[IngestedDocument]:
        filters: dict[str, Any] = {"content_hash": content_hash}
        rows = await self._select(self.TABLE_DOCUMENTS, filters=filters, limit=1)
        if rows and source_id and rows[0].get("source_id") != source_id:
            return None
        return from_dict(IngestedDocument, _lower_keys(rows[0])) if rows else None

    async def find_document_by_url(self, url: str) -> Optional[IngestedDocument]:
        rows = await self._select(self.TABLE_DOCUMENTS, filters={"url": url}, limit=1)
        return from_dict(IngestedDocument, _lower_keys(rows[0])) if rows else None

    async def save_document(self, doc: IngestedDocument) -> IngestedDocument:
        await self._insert(self.TABLE_DOCUMENTS, [self._row(doc)])
        return doc

    async def list_documents(self, source_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> list[IngestedDocument]:
        params = "&".join(
            [
                "select=*",
                f"limit={limit}",
                f"offset={offset}",
                "order=ingested_at.desc",
                *(f"source_id=eq.{source_id}" if source_id else []),
            ]
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{self._base}/{self.TABLE_DOCUMENTS}?{params}", headers=self._headers)
        if not resp.is_success:
            return []
        rows = resp.json()
        return [from_dict(IngestedDocument, _lower_keys(r)) for r in rows]

    async def get_document(self, doc_id: str) -> Optional[IngestedDocument]:
        rows = await self._select(self.TABLE_DOCUMENTS, filters={"id": doc_id}, limit=1)
        return from_dict(IngestedDocument, _lower_keys(rows[0])) if rows else None

    async def count_documents(self) -> int:
        return await self._count(self.TABLE_DOCUMENTS)

    async def delete_document(self, doc_id: str) -> bool:
        existing = await self.get_document(doc_id)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.delete(f"{self._base}/{self.TABLE_DOCUMENTS}?id=eq.{doc_id}", headers=self._headers)
        return existing is not None and resp.is_success

    async def recent_content_hashes(self, limit: int = 300) -> set[str]:
        rows = await self._select(self.TABLE_DOCUMENTS, limit=limit, order="ingested_at.desc")
        return {str(r.get("content_hash", "")) for r in rows if r.get("content_hash")}

    # ── chunks ──────────────────────────────────────────────────────────

    async def save_chunk(self, chunk: IngestedChunk) -> IngestedChunk:
        await self._insert(self.TABLE_CHUNKS, [self._row(chunk)])
        return chunk

    async def save_chunks(self, chunks: Iterable[IngestedChunk]) -> int:
        rows = [self._row(c) for c in chunks]
        await self._insert(self.TABLE_CHUNKS, rows)
        return len(rows)

    async def update_chunk_embedding(self, chunk_id: str, embedding: list[float]) -> bool:
        return await self._update_row(self.TABLE_CHUNKS, {"id": chunk_id}, {"embedding": embedding})

    async def chunks_for_document(self, doc_id: str) -> list[IngestedChunk]:
        rows = await self._select(self.TABLE_CHUNKS, filters={"document_id": doc_id}, limit=500, order="chunk_index.asc")
        return [from_dict(IngestedChunk, _lower_keys(r)) for r in rows]

    async def chunk_samples(self, limit: int = 100) -> list[dict[str, Any]]:
        return await self._select(self.TABLE_CHUNKS, limit=limit, order="created_at.desc")

    async def count_chunks(self) -> int:
        return await self._count(self.TABLE_CHUNKS)

    # ── claims / entities / edges / updates ─────────────────────────────

    async def save_claims(self, claims: Iterable[KnowledgeClaim]) -> int:
        rows = [self._row(c) for c in claims]
        await self._insert(self.TABLE_CLAIMS, rows)
        return len(rows)

    async def list_claims(self, entity: Optional[str] = None, limit: int = 50) -> list[KnowledgeClaim]:
        rows = await self._select(self.TABLE_CLAIMS, limit=limit, order="collected_at.desc")
        if entity:
            rows = [r for r in rows if r.get("entity", "").lower() == entity.lower()]
        return [from_dict(KnowledgeClaim, _lower_keys(r)) for r in rows]

    async def claims_for_document(self, doc_id: str) -> list[KnowledgeClaim]:
        rows = await self._select(self.TABLE_CLAIMS, filters={"document_id": doc_id}, limit=200, order="collected_at.desc")
        return [from_dict(KnowledgeClaim, _lower_keys(r)) for r in rows]

    async def set_claim_status(self, claim_id: str, status: Any) -> bool:
        value = status.value if hasattr(status, "value") else str(status)
        return await self._update_row(self.TABLE_CLAIMS, {"id": claim_id}, {"status": value})

    async def increment_claim_contradictions(self, claim_id: str) -> bool:
        claim = await self._claim(claim_id)
        if claim is None:
            return False
        return await self._update_row(self.TABLE_CLAIMS, {"id": claim_id}, {"contradictions": claim.contradictions + 1})

    async def _claim(self, claim_id: str) -> Optional[KnowledgeClaim]:
        rows = await self._select(self.TABLE_CLAIMS, filters={"id": claim_id}, limit=1)
        return from_dict(KnowledgeClaim, _lower_keys(rows[0])) if rows else None

    async def upsert_entity(self, name: str, kind: str = "", description: str = "") -> KnowledgeEntity:
        # PostgREST can't express ON CONFLICT — fetch, bump, or insert.
        rows = await self._select(self.TABLE_ENTITIES, filters={"name": name}, limit=1)
        if rows:
            row = rows[0]
            await self._update_row(
                self.TABLE_ENTITIES,
                {"name": name},
                {"mention_count": int(row.get("mention_count", 1)) + 1, "last_seen_at": datetime.now(timezone.utc).isoformat()},
            )
            return from_dict(KnowledgeEntity, _lower_keys(row))
        entity = KnowledgeEntity(id=uuid.uuid4().hex, name=name, kind=kind, description=description)
        await self._insert(self.TABLE_ENTITIES, [self._row(entity)])
        return entity

    async def list_entities(self, limit: int = 20) -> list[KnowledgeEntity]:
        rows = await self._select(self.TABLE_ENTITIES, limit=limit, order="mention_count.desc")
        return [from_dict(KnowledgeEntity, _lower_keys(r)) for r in rows]

    async def save_edges(self, edges: Iterable[EntityEdge]) -> int:
        rows = [self._row(e) for e in edges]
        await self._insert(self.TABLE_EDGES, rows)
        return len(rows)

    async def record_update(self, update: KnowledgeUpdate) -> KnowledgeUpdate:
        await self._insert(self.TABLE_UPDATES, [self._row(update)])
        return update

    async def list_updates(self, limit: int = 20) -> list[KnowledgeUpdate]:
        rows = await self._select(self.TABLE_UPDATES, limit=limit, order="detected_at.desc")
        return [from_dict(KnowledgeUpdate, _lower_keys(r)) for r in rows]

    # ── stats / health ──────────────────────────────────────────────────

    async def stats(self) -> IngestStats:
        sources = await self.list_sources()
        runs = await self._select(self.TABLE_RUNS, limit=500, order="started_at.desc")
        errors = await self._select(self.TABLE_ERRORS, limit=500, order="occurred_at.desc")
        now = datetime.now(timezone.utc)
        added_24h = 0
        last_success = None
        for run in runs:
            started = _dt(run.get("started_at"))
            if started and started >= now - timedelta(hours=24) and run.get("status") in ("success", "partial"):
                added_24h += 1
            if run.get("status") in ("success", "partial") and last_success is None:
                last_success = started or last_success
        return IngestStats(
            source_count=len(sources),
            enabled_count=sum(1 for s in sources if s.enabled),
            document_count=await self.count_documents(),
            chunk_count=await self.count_chunks(),
            claim_count=await self._count(self.TABLE_CLAIMS),
            entity_count=await self._count(self.TABLE_ENTITIES),
            update_count=await self._count(self.TABLE_UPDATES),
            run_count=len(runs),
            error_count=len(errors),
            added_24h=added_24h,
            last_successful_sync=last_success,
            store="supabase",
        )

    async def health(self) -> dict[str, Any]:
        return {"store": "supabase", "ok": True, "url": self.settings.supabase_url}


def _source_from_row(row: dict[str, Any]) -> IngestSource:
    data = _lower_keys(row)
    kind = data.get("kind", "web")
    try:
        data["kind"] = SourceKind(kind)
    except ValueError:
        data["kind"] = SourceKind.WEB
    freq = data.get("crawl_frequency", "daily")
    try:
        data["crawl_frequency"] = CrawlFrequency(freq)
    except ValueError:
        data["crawl_frequency"] = CrawlFrequency.DAILY
    return from_dict(IngestSource, data)


def _lower_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {str(k).lower(): v for k, v in row.items()}


# ─────────────────────────────────────────────────────────────────────────
#  Facade
# ─────────────────────────────────────────────────────────────────────────


class KnowledgeStore:
    """Picks the backend and delegates every method to it."""

    def __init__(self, settings: BrainSettings) -> None:
        self.settings = settings
        mode = (settings.ingest_store or "auto").lower()
        use_supabase = mode == "supabase" or (
            mode == "auto" and bool(settings.supabase_url and settings.supabase_key)
        )
        if use_supabase:
            self._backend = SupabaseStore(settings)
            self.backend_name = "supabase"
        else:
            self._backend = LocalFileStore(Path(settings.ingest_data_dir))
            self.backend_name = "local"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    async def stats(self) -> IngestStats:
        return await self._backend.stats()

    async def health(self) -> dict[str, Any]:
        base = await self._backend.health()
        base["configured_store"] = self.backend_name
        return base
