"""Connector base + shared HTTP helpers for the ingestion sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

import httpx

from ...config import BrainSettings
from ..models import IngestSource, RawItem, SourceKind


class IngestConnector(ABC):
    """Fetch raw items from one external source type.

    Every connector returns pre-normalized :class:`RawItem` objects; the
    pipeline handles sanitization, deduplication, scoring and extraction.
    Connectors must be read-only and never require an API key (tokens such as
    ``GITHUB_TOKEN`` are optional rate-limit upgrades only).
    """

    kind: SourceKind = SourceKind.GENERIC_API

    def __init__(self, settings: BrainSettings) -> None:
        self.settings = settings

    @property
    def user_agent(self) -> str:
        return self.settings.ingest_user_agent

    @abstractmethod
    async def fetch(
        self, source: IngestSource, client: httpx.AsyncClient
    ) -> list[RawItem]:
        """Fetch + normalize items for ``source``. Never raises."""

    def can_handle(self, source: IngestSource) -> bool:  # noqa: ARG002
        return True


# ─────────────────────────────────────────────────────────────────────────
#  HTTP helpers (best-effort: a failed fetch yields [] / "", never throws)
# ─────────────────────────────────────────────────────────────────────────


async def http_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 20.0,
) -> Any:
    try:
        resp = await client.get(url, params=params, headers=headers, timeout=timeout)
        if not resp.is_success:
            return None
        return resp.json()
    except (httpx.HTTPError, ValueError):
        return None


async def http_text(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 20.0,
) -> str:
    try:
        resp = await client.get(url, params=params, headers=headers, timeout=timeout)
        if not resp.is_success:
            return ""
        return resp.text
    except httpx.HTTPError:
        return ""


def parse_dt(value: Any, default: Optional[datetime] = None) -> Optional[datetime]:
    """Parse a date from any string-ish value; never raises."""
    if not value:
        return default
    from dateutil import parser

    try:
        return parser.parse(str(value))
    except (ValueError, TypeError, OverflowError):
        return default


def first(iterable: Any, *keys: str) -> Any:
    """First non-empty value among ``keys`` across a dict-like object."""
    if not isinstance(iterable, dict):
        return None
    for key in keys:
        value = iterable.get(key)
        if value:
            return value
    return None
