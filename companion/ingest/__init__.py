"""Continuous knowledge-ingestion engine for RelayAI.

Importing this package never loads heavy dependencies (connectors pull in
httpx/dateutil/bs4 lazily only when ingestion actually runs).

Public surface (lazy via PEP 562): :class:`KnowledgeStore`,
:class:`IngestionPipeline`, :class:`IngestionScheduler`, ``due_sources``,
``build_ingest_knowledge_context``, ``maybe_refresh``.
"""

from __future__ import annotations

__all__ = [
    "KnowledgeStore",
    "IngestionPipeline",
    "IngestionScheduler",
    "due_sources",
    "get_scheduler",
    "start_background_scheduler",
    "build_ingest_knowledge_context",
    "maybe_refresh",
]


def __getattr__(name: str):
    if name == "KnowledgeStore":
        from .store import KnowledgeStore

        return KnowledgeStore
    if name in ("IngestionPipeline",):
        from .pipeline import IngestionPipeline

        return IngestionPipeline
    if name in ("IngestionScheduler", "due_sources", "get_scheduler", "start_background_scheduler"):
        from .scheduler import IngestionScheduler, due_sources, get_scheduler, start_background_scheduler

        return {"IngestionScheduler": IngestionScheduler, "due_sources": due_sources, "get_scheduler": get_scheduler, "start_background_scheduler": start_background_scheduler}[name]
    if name in ("build_ingest_knowledge_context", "maybe_refresh"):
        from .brain import build_ingest_knowledge_context, maybe_refresh

        return {"build_ingest_knowledge_context": build_ingest_knowledge_context, "maybe_refresh": maybe_refresh}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
