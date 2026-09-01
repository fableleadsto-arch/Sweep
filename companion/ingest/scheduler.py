"""Background scheduler for continuous ingestion.

The first time-based scheduler in the companion service. Each enabled source
runs on its own cadence (15m / hourly / 6h / daily / weekly) with:

* **per-source isolation** — a failing source never blocks or kills others;
* **exponential backoff** — consecutive failures push the next attempt out
  (300s, 600s, 1200s, … capped at 24h), disabled after N failures;
* **no overlap** — a source already being ingested is never started twice;
* **manual triggers** with a cooldown so the dashboard can't hammer a source.

``due_sources`` is a pure function so the cadence/backoff math is unit-testable
without any I/O.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from ..config import BrainSettings
from .models import FREQUENCY_SECONDS, CrawlFrequency, IngestSource, utcnow
from .pipeline import IngestionPipeline
from .store import KnowledgeStore

logger = logging.getLogger(__name__)

BACKOFF_BASE_SECONDS = 300
BACKOFF_MAX_SECONDS = 24 * 3600
MAX_CONSECUTIVE_FAILURES = 8
MANUAL_TRIGGER_COOLDOWN_SECONDS = 60


def backoff_seconds(source: IngestSource) -> int:
    """How long to wait before retrying a source that keeps failing."""
    failures = max(0, source.consecutive_failures)
    if failures == 0:
        return 0
    seconds = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** (failures - 1)))
    return int(seconds)


def due_sources(
    sources: list[IngestSource],
    now: Optional[datetime] = None,
    *,
    disabled_after: int = MAX_CONSECUTIVE_FAILURES,
) -> list[IngestSource]:
    """Sources whose cadence has elapsed and are not backing off."""
    now = now or utcnow()
    due: list[IngestSource] = []
    for source in sources:
        if not source.enabled:
            continue
        if source.consecutive_failures >= disabled_after:
            continue
        last = source.last_checked or source.last_successful_sync
        if last is None:
            due.append(source)
            continue
        elapsed = (now - _aware(last)).total_seconds()
        # A failing source waits out its exponential backoff, then is due again.
        if source.consecutive_failures > 0:
            if elapsed < backoff_seconds(source):
                continue
            due.append(source)
            continue
        interval = FREQUENCY_SECONDS.get(source.crawl_frequency, FREQUENCY_SECONDS[CrawlFrequency.DAILY])
        if elapsed >= interval:
            due.append(source)
    return due


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=utcnow().tzinfo)
    return value


class IngestionScheduler:
    """Asyncio background loop driving the ingestion pipeline."""

    def __init__(self, store: KnowledgeStore, pipeline: IngestionPipeline, settings: BrainSettings) -> None:
        self.store = store
        self.pipeline = pipeline
        self.settings = settings
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._in_flight: set[str] = set()
        self._last_manual: dict[str, datetime] = {}

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="ingest-scheduler")
        logger.info("ingestion scheduler started (tick %ss)", self.settings.ingest_scheduler_tick_seconds)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.tick()
            except Exception as exc:  # noqa: BLE001 - the loop must survive errors
                logger.warning("ingestion scheduler tick failed: %s", exc)
            await asyncio.sleep(self.settings.ingest_scheduler_tick_seconds)

    async def tick(self) -> None:
        """Start ingestion for every due source (non-blocking, isolated)."""
        sources = await self.store.list_sources()
        for source in due_sources(sources):
            if source.id in self._in_flight:
                continue
            await self._enqueue(source.id)

    async def _enqueue(self, source_id: str) -> None:
        self._in_flight.add(source_id)

        async def _run() -> None:
            try:
                await self.pipeline.run_source(source_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("scheduled ingest failed for %s: %s", source_id, exc)
            finally:
                self._in_flight.discard(source_id)

        asyncio.create_task(_run(), name=f"ingest-{source_id[:8]}")

    async def trigger(self, source_id: str, *, force: bool = False) -> bool:
        """Manually start a source now (dashboard/on-demand refresh).

        Returns False when a manual trigger is within the cooldown window or
        the source is already running.
        """
        now = utcnow()
        if not force:
            last = self._last_manual.get(source_id)
            if last is not None and (now - last).total_seconds() < MANUAL_TRIGGER_COOLDOWN_SECONDS:
                return False
        if source_id in self._in_flight:
            return False
        self._last_manual[source_id] = now
        await self._enqueue(source_id)
        return True


# ─────────────────────────────────────────────────────────────────────────
#  Process-wide scheduler singleton (lazily started)
# ─────────────────────────────────────────────────────────────────────────

_scheduler: Optional[IngestionScheduler] = None


def get_scheduler(settings: Optional[BrainSettings] = None) -> IngestionScheduler:
    """Return (and lazily start) the process-wide scheduler singleton."""
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    from ..config import get_settings as _get_settings

    settings = settings or _get_settings()
    store = KnowledgeStore(settings)
    pipeline = IngestionPipeline(store, settings)
    _scheduler = IngestionScheduler(store, pipeline, settings)
    return _scheduler


async def start_background_scheduler() -> Optional[IngestionScheduler]:
    """Start the background loop when enabled (called from the FastAPI lifespan)."""
    from ..config import get_settings

    settings = get_settings()
    if not settings.enable_knowledge_ingestion or not settings.ingest_scheduler_enabled:
        return None
    scheduler = get_scheduler(settings)
    await scheduler.start()
    return scheduler


async def stop_background_scheduler() -> None:
    if _scheduler is not None:
        await _scheduler.stop()
