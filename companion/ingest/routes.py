"""FastAPI routes for the continuous knowledge-ingestion engine.

Everything under ``/api/brain/ingest/*``. Handlers lazily import the ingestion
modules so importing this module stays light; the heavy work (connectors, bs4,
dateutil, embeddings) only loads when a source is actually ingested.

All operations are authenticated by the standard ``require_token`` dependency.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..config import BrainSettings, get_settings
from ..main import require_token
from .models import (
    ClaimStatus,
    CrawlFrequency,
    IngestStats,
    IngestSource,
    SourceKind,
    to_dict,
)
from .security import SSRFBlockedError, validate_outbound_url

router = APIRouter(prefix="/api/brain/ingest", tags=["ingest"])


# ─────────────────────────────────────────────────────────────────────────
#  Request models
# ─────────────────────────────────────────────────────────────────────────


class SourcePayload(BaseModel):
    kind: SourceKind
    name: str = Field(min_length=1, max_length=200)
    url: str = Field(min_length=1, max_length=2000)
    topics: list[str] = Field(default_factory=list)
    category: str = Field(default="", max_length=100)
    priority: int = Field(default=5, ge=0, le=10)
    trust_score: float = Field(default=None, ge=0.0, le=1.0)  # type: ignore[assignment]
    crawl_frequency: CrawlFrequency = CrawlFrequency.DAILY
    config: dict[str, Any] = Field(default_factory=dict)


class SourcePatch(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    url: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    topics: Optional[list[str]] = None
    category: Optional[str] = Field(default=None, max_length=100)
    priority: Optional[int] = Field(default=None, ge=0, le=10)
    trust_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    crawl_frequency: Optional[CrawlFrequency] = None
    config: Optional[dict[str, Any]] = None
    enabled: Optional[bool] = None


class TogglePayload(BaseModel):
    enabled: bool


# ─────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────


def _store(settings: BrainSettings):
    from .store import KnowledgeStore

    return KnowledgeStore(settings)


def _source_out(source: IngestSource) -> dict[str, Any]:
    return to_dict(source)


# ─────────────────────────────────────────────────────────────────────────
#  Sources
# ─────────────────────────────────────────────────────────────────────────


@router.get("/sources", dependencies=[Depends(require_token)])
async def list_sources(settings: BrainSettings = Depends(get_settings)) -> list[dict[str, Any]]:
    sources = await _store(settings).list_sources()
    return [_source_out(s) for s in sources]


@router.get("/sources/kinds", dependencies=[Depends(require_token)])
async def list_source_kinds() -> list[dict[str, Any]]:
    from .sources import list_connector_kinds

    return list_connector_kinds()


@router.post("/sources", dependencies=[Depends(require_token)])
async def create_source(
    payload: SourcePayload,
    settings: BrainSettings = Depends(get_settings),
) -> dict[str, Any]:
    from .sources import default_source

    try:
        validate_outbound_url(payload.url, resolve=False)
    except SSRFBlockedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    source = default_source(
        kind=payload.kind,
        name=payload.name,
        url=payload.url,
        topics=payload.topics,
        category=payload.category,
        priority=payload.priority,
        trust_score=payload.trust_score,
        frequency=payload.crawl_frequency,
        config=payload.config,
    )
    await _store(settings).upsert_source(source)
    return _source_out(source)


@router.post("/sources/{source_id}/toggle", dependencies=[Depends(require_token)])
async def toggle_source(
    source_id: str,
    payload: TogglePayload,
    settings: BrainSettings = Depends(get_settings),
) -> dict[str, Any]:
    store = _store(settings)
    source = await store.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    source.enabled = payload.enabled
    await store.upsert_source(source)
    return _source_out(source)


@router.post("/sources/{source_id}", dependencies=[Depends(require_token)])
async def update_source(
    source_id: str,
    payload: SourcePatch,
    settings: BrainSettings = Depends(get_settings),
) -> dict[str, Any]:
    store = _store(settings)
    source = await store.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    if payload.name is not None:
        source.name = payload.name.strip()
    if payload.url is not None:
        try:
            validate_outbound_url(payload.url, resolve=False)
        except SSRFBlockedError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        source.url = payload.url.strip()
    if payload.topics is not None:
        source.topics = [t.strip() for t in payload.topics if t and t.strip()]
    if payload.category is not None:
        source.category = payload.category
    if payload.priority is not None:
        source.priority = max(0, min(10, payload.priority))
    if payload.trust_score is not None:
        source.trust_score = max(0.0, min(1.0, payload.trust_score))
    if payload.crawl_frequency is not None:
        source.crawl_frequency = payload.crawl_frequency
    if payload.config is not None:
        source.config = payload.config
    if payload.enabled is not None:
        source.enabled = payload.enabled
    await store.upsert_source(source)
    return _source_out(source)


@router.delete("/sources/{source_id}", dependencies=[Depends(require_token)])
async def delete_source(source_id: str, settings: BrainSettings = Depends(get_settings)) -> dict[str, Any]:
    deleted = await _store(settings).delete_source(source_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Source not found")
    return {"deleted": True, "id": source_id}


@router.post("/sources/{source_id}/sync", dependencies=[Depends(require_token)])
async def sync_source(
    source_id: str,
    settings: BrainSettings = Depends(get_settings),
) -> dict[str, Any]:
    """Trigger a one-shot ingestion run for one source (returns the run)."""
    from .pipeline import IngestionPipeline

    store = _store(settings)
    if await store.get_source(source_id) is None:
        raise HTTPException(status_code=404, detail="Source not found")
    pipeline = IngestionPipeline(store, settings)
    run = await pipeline.run_source(source_id)
    return to_dict(run) if run else {"status": "skipped"}


@router.post("/sync/all", dependencies=[Depends(require_token)])
async def sync_all(settings: BrainSettings = Depends(get_settings)) -> dict[str, Any]:
    from .pipeline import IngestionPipeline

    pipeline = IngestionPipeline(_store(settings), settings)
    runs = await pipeline.refresh_all()
    return {
        "runs": [to_dict(r) for r in runs],
        "count": len(runs),
        "summary": {
            "added": sum(r.added for r in runs),
            "duplicates": sum(r.duplicates for r in runs),
            "rejected": sum(r.rejected for r in runs),
            "errors": sum(r.error_count for r in runs),
        },
    }


# ─────────────────────────────────────────────────────────────────────────
#  Statistics / history
# ─────────────────────────────────────────────────────────────────────────


@router.get("/stats", dependencies=[Depends(require_token)])
async def ingest_stats(settings: BrainSettings = Depends(get_settings)) -> IngestStats:
    return await _store(settings).stats()


@router.get("/runs", dependencies=[Depends(require_token)])
async def list_runs(
    source_id: Optional[str] = None,
    limit: int = 20,
    settings: BrainSettings = Depends(get_settings),
) -> list[dict[str, Any]]:
    runs = await _store(settings).list_runs(source_id=source_id, limit=max(1, min(100, limit)))
    return [to_dict(r) for r in runs]


@router.get("/errors", dependencies=[Depends(require_token)])
async def list_errors(
    source_id: Optional[str] = None,
    limit: int = 20,
    settings: BrainSettings = Depends(get_settings),
) -> list[dict[str, Any]]:
    errors = await _store(settings).list_errors(source_id=source_id, limit=max(1, min(100, limit)))
    return [to_dict(e) for e in errors]


@router.get("/documents", dependencies=[Depends(require_token)])
async def list_documents(
    source_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    settings: BrainSettings = Depends(get_settings),
) -> list[dict[str, Any]]:
    docs = await _store(settings).list_documents(
        source_id=source_id, limit=max(1, min(200, limit)), offset=max(0, offset)
    )
    return [to_dict(d) for d in docs]


@router.get("/documents/{document_id}", dependencies=[Depends(require_token)])
async def document_detail(
    document_id: str,
    settings: BrainSettings = Depends(get_settings),
) -> dict[str, Any]:
    store = _store(settings)
    doc = await store.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = await store.chunks_for_document(document_id)
    claims = await store.claims_for_document(document_id)
    return {
        "document": to_dict(doc),
        "chunks": [to_dict(c) for c in chunks],
        "claims": [to_dict(c) for c in claims],
    }


@router.get("/claims", dependencies=[Depends(require_token)])
async def list_claims(
    entity: Optional[str] = None,
    limit: int = 50,
    settings: BrainSettings = Depends(get_settings),
) -> list[dict[str, Any]]:
    claims = await _store(settings).list_claims(entity=entity, limit=max(1, min(200, limit)))
    return [to_dict(c) for c in claims]


@router.get("/updates", dependencies=[Depends(require_token)])
async def list_updates(
    limit: int = 20,
    settings: BrainSettings = Depends(get_settings),
) -> list[dict[str, Any]]:
    updates = await _store(settings).list_updates(limit=max(1, min(100, limit)))
    return [to_dict(u) for u in updates]


@router.get("/diagnostics", dependencies=[Depends(require_token)])
async def ingest_diagnostics(settings: BrainSettings = Depends(get_settings)) -> dict[str, Any]:
    from datetime import datetime, timezone

    store = _store(settings)
    health = await store.health()
    stats = await store.stats()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "enabled": settings.enable_knowledge_ingestion,
        "scheduler_enabled": settings.ingest_scheduler_enabled,
        "embedding_available": bool(settings.gemini_api_key or settings.openai_api_key),
        "github_token": bool(settings.github_token),
        "openalex_api_key": bool(settings.openalex_api_key),
        "store": health,
        "stats": to_dict(stats),
    }


# ─────────────────────────────────────────────────────────────────────────
#  Install (mirrors companion.brain_agent._install)
# ─────────────────────────────────────────────────────────────────────────


def _install() -> None:
    from ..main import app

    existing = {getattr(route, "path", None) for route in app.routes}
    if not any(path and path.startswith("/api/brain/ingest") for path in existing):
        app.include_router(router)


_install()
