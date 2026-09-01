"""Data models for the continuous knowledge-ingestion engine.

Pure dataclasses + enums with JSON round-tripping helpers so both the local
file store and the Supabase REST store serialize exactly the same shape.

Nothing here imports a heavy framework — this module must stay import-safe for
boot (`import companion` should not pull in torch, bs4, or an HTTP client).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, get_type_hints

# ─────────────────────────────────────────────────────────────────────────
#  Enums
# ─────────────────────────────────────────────────────────────────────────


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SourceKind(str, Enum):
    """Connector types the registry can dispatch on."""

    RSS = "rss"
    GITHUB = "github"
    ARXIV = "arxiv"
    OPENALEX = "openalex"
    CROSSREF = "crossref"
    WIKIPEDIA = "wikipedia"
    HUGGINGFACE = "huggingface"
    WEB = "web"
    GENERIC_API = "generic_api"


SOURCE_KIND_LABELS: dict[SourceKind, str] = {
    SourceKind.RSS: "RSS / Atom feed",
    SourceKind.GITHUB: "GitHub repository",
    SourceKind.ARXIV: "arXiv",
    SourceKind.OPENALEX: "OpenAlex",
    SourceKind.CROSSREF: "Crossref",
    SourceKind.WIKIPEDIA: "Wikipedia",
    SourceKind.HUGGINGFACE: "Hugging Face",
    SourceKind.WEB: "Web page",
    SourceKind.GENERIC_API: "Generic API",
}


class CrawlFrequency(str, Enum):
    """How often a source should be re-checked (scheduler cadence)."""

    FIFTEEN_MIN = "15m"
    HOURLY = "hourly"
    EVERY_6H = "6h"
    DAILY = "daily"
    WEEKLY = "weekly"


FREQUENCY_SECONDS: dict[CrawlFrequency, int] = {
    CrawlFrequency.FIFTEEN_MIN: 15 * 60,
    CrawlFrequency.HOURLY: 60 * 60,
    CrawlFrequency.EVERY_6H: 6 * 60 * 60,
    CrawlFrequency.DAILY: 24 * 60 * 60,
    CrawlFrequency.WEEKLY: 7 * 24 * 60 * 60,
}


class RunStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class ClaimStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONTRADICTED = "contradicted"


# ─────────────────────────────────────────────────────────────────────────
#  Serialization helpers
# ─────────────────────────────────────────────────────────────────────────


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _dt(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def to_dict(model: Any) -> dict[str, Any]:
    """Dataclass → JSON-safe dict (datetimes → ISO strings, enums → values)."""
    data = asdict(model)
    for key, value in data.items():
        if isinstance(value, datetime):
            data[key] = value.isoformat()
        elif isinstance(value, Enum):
            data[key] = value.value
    return data


def from_dict(cls: type, data: dict[str, Any]) -> Any:
    """JSON dict → dataclass (ISO strings → datetimes, values → enums).

    Uses :func:`typing.get_type_hints` so fields annotated under
    ``from __future__ import annotations`` (string annotations) still convert
    enums and datetimes correctly.
    """
    if not isinstance(data, dict):
        return cls()
    try:
        hints = get_type_hints(cls)
    except Exception:  # noqa: BLE001 - fall back to raw (possibly string) hints
        hints = getattr(cls, "__annotations__", {})
    values: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            values[key] = None
            continue
        field_type = hints.get(key)
        if isinstance(value, datetime):
            values[key] = value
        elif isinstance(value, str) and "datetime" in str(field_type):
            values[key] = _dt(value)
        elif field_type is not None and isinstance(field_type, type) and issubclass(field_type, Enum):
            try:
                values[key] = field_type(value)
            except ValueError:
                values[key] = value
        else:
            values[key] = value
    return cls(**values)


# ─────────────────────────────────────────────────────────────────────────
#  Models
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class IngestSource:
    """A persistent, configurable ingestion source (source registry entry)."""

    id: str
    kind: SourceKind
    name: str
    url: str
    topics: list[str] = field(default_factory=list)
    category: str = ""
    priority: int = 5
    trust_score: float = 0.5
    crawl_frequency: CrawlFrequency = CrawlFrequency.DAILY
    enabled: bool = True
    last_checked: Optional[datetime] = None
    last_successful_sync: Optional[datetime] = None
    error_count: int = 0
    consecutive_failures: int = 0
    rate_limit_hint: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)


@dataclass
class RawItem:
    """One pre-normalization item produced by a connector."""

    title: str
    url: str
    content: str
    summary: str = ""
    author: str = ""
    published_at: Optional[datetime] = None
    external_id: str = ""
    etag: Optional[str] = None
    last_modified: Optional[str] = None


@dataclass
class IngestedDocument:
    """A cleaned, scored document in the continuous-knowledge store."""

    id: str
    source_id: str
    source_kind: SourceKind
    name: str
    content: str
    url: str = ""
    published_at: Optional[datetime] = None
    ingested_at: datetime = field(default_factory=utcnow)
    content_hash: str = ""
    external_id: str = ""
    topics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    relevance: float = 0.0
    quality: float = 0.0
    confidence: float = 0.0
    chunk_count: int = 0
    status: str = "ready"


@dataclass
class IngestedChunk:
    """One embeddable chunk of an ingested document."""

    id: str
    document_id: str
    source_id: str
    chunk_index: int
    content: str
    heading: str = ""
    tokens: int = 0
    embedding: Optional[list[float]] = None


@dataclass
class KnowledgeClaim:
    """A structured fact extracted from an ingested document.

    ``subject`` disambiguates same-entity-property claims (e.g. a library
    ``supports`` a language, a company ``uses`` a stack).
    """

    id: str
    entity: str
    property: str
    value: str
    subject: str = ""
    document_id: str = ""
    source_id: str = ""
    source_url: str = ""
    collected_at: datetime = field(default_factory=utcnow)
    confidence: float = 0.0
    authority: float = 0.0
    status: ClaimStatus = ClaimStatus.ACTIVE
    contradictions: int = 0


@dataclass
class KnowledgeEntity:
    """A canonical entity observed across ingested documents."""

    id: str
    name: str
    kind: str = ""
    description: str = ""
    mention_count: int = 1
    first_seen_at: datetime = field(default_factory=utcnow)
    last_seen_at: datetime = field(default_factory=utcnow)


@dataclass
class EntityEdge:
    """A directed relation between two entities (knowledge graph edge)."""

    id: str
    from_entity: str
    to_entity: str
    relation: str
    source_url: str = ""
    confidence: float = 0.0
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class KnowledgeUpdate:
    """A detected change/contradiction in previously stored knowledge."""

    id: str
    entity: str
    property: str
    old_value: Optional[str]
    new_value: Optional[str]
    source_id: str = ""
    source_url: str = ""
    detected_at: datetime = field(default_factory=utcnow)
    reason: str = ""
    confidence: float = 0.0


@dataclass
class IngestionRun:
    """One execution of a source's ingest pipeline."""

    id: str
    source_id: str
    started_at: datetime = field(default_factory=utcnow)
    finished_at: Optional[datetime] = None
    status: RunStatus = RunStatus.RUNNING
    items_found: int = 0
    added: int = 0
    duplicates: int = 0
    rejected: int = 0
    updated: int = 0
    error_count: int = 0
    error_message: str = ""


@dataclass
class IngestionError:
    """A recorded per-source failure with retry bookkeeping."""

    id: str
    source_id: str
    occurred_at: datetime = field(default_factory=utcnow)
    stage: str = "fetch"
    http_status: int = 0
    error_type: str = ""
    message: str = ""
    retry_count: int = 0


@dataclass
class IngestStats:
    """Aggregate counters shown in the dashboard."""

    source_count: int = 0
    enabled_count: int = 0
    document_count: int = 0
    chunk_count: int = 0
    claim_count: int = 0
    entity_count: int = 0
    update_count: int = 0
    run_count: int = 0
    error_count: int = 0
    added_24h: int = 0
    last_successful_sync: Optional[datetime] = None
    store: str = "local"
