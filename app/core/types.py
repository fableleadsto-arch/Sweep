"""Core type contracts — Pydantic models replacing TypeScript interfaces.

Every type that flows between modules is defined here so the API surface,
search layer, extraction layer, and research engine all share a single
source of truth.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Search ────────────────────────────────────────────────────────────

class SearchAccessMode(str, Enum):
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    UNAVAILABLE = "unavailable"


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str = ""
    provider: str
    access_mode: SearchAccessMode = SearchAccessMode.PUBLIC
    platform: Optional[str] = None
    published_at: Optional[str] = None
    relevance: Optional[float] = None
    metadata: dict[str, str] = Field(default_factory=dict)


class SearchOptions(BaseModel):
    limit: int = 10
    site: Optional[str] = None
    time_range: Optional[str] = None  # "day" | "week" | "month" | "year"
    after: Optional[str] = None
    before: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None
    page: int = 1
    aggregate: bool = False


class SearchRun(BaseModel):
    provider: str
    results: list[SearchResult] = Field(default_factory=list)
    blocked: bool = False
    note: Optional[str] = None
    errors: list[str] = Field(default_factory=list)


class SearchRunResult(BaseModel):
    provider: str
    query: str
    results: list[SearchResult] = Field(default_factory=list)
    blocked: bool = False
    note: Optional[str] = None
    errors: list[str] = Field(default_factory=list)


# ── Page / Browse ─────────────────────────────────────────────────────

class LinkData(BaseModel):
    url: str
    text: str
    intent: Optional[str] = None
    external: bool = False


class HeadingData(BaseModel):
    level: int
    text: str
    id: Optional[str] = None


class InjectionAssessment(BaseModel):
    suspect: bool = False
    signals: list[str] = Field(default_factory=list)


class PageData(BaseModel):
    url: str
    title: str = ""
    description: Optional[str] = None
    text: str = ""
    markdown: Optional[str] = None
    links: list[LinkData] = Field(default_factory=list)
    headings: list[HeadingData] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    structured_data: Optional[Any] = None
    truncated: bool = False
    fetched_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: int = 200
    content_type: str = "text/html"
    access_mode: SearchAccessMode = SearchAccessMode.PUBLIC
    blocked: bool = False
    injection: Optional[InjectionAssessment] = None


class NavigationCommand(BaseModel):
    kind: str  # "goto" | "click" | "fill" | "scroll" | "back" | "forward" | "reload"
    target: Optional[str] = None
    value: Optional[str] = None
    selector: Optional[str] = None
    direction: Optional[str] = None  # "down" | "up" | "bottom" | "top"


class NavigationResult(BaseModel):
    ok: bool
    url: str
    title: Optional[str] = None
    text: Optional[str] = None
    links: Optional[list[LinkData]] = None
    headings: Optional[list[HeadingData]] = None
    error: Optional[str] = None
    blocked: bool = False


# ── Platforms ─────────────────────────────────────────────────────────

class SurfPlatform(str, Enum):
    REDDIT = "reddit"
    X = "x"
    INSTAGRAM = "instagram"
    YOUTUBE = "youtube"
    GITHUB = "github"
    LINKEDIN = "linkedin"
    GENERIC = "generic"


# ── Evidence ──────────────────────────────────────────────────────────

class SourceType(str, Enum):
    OFFICIAL = "official"
    PRIMARY = "primary"
    COMMUNITY = "community"
    SECONDARY = "secondary"
    NEWS = "news"
    DOCUMENTATION = "documentation"
    REPOSITORY = "repository"
    SOCIAL = "social"
    UNKNOWN = "unknown"


class SourceScore(BaseModel):
    relevance: float = 0.0
    authority: float = 0.0
    freshness: float = 0.0
    directness: float = 0.0
    overall: float = 0.0


class Source(BaseModel):
    title: str
    url: str
    platform: Optional[SurfPlatform] = None
    type: Optional[SourceType] = None
    access_mode: SearchAccessMode = SearchAccessMode.PUBLIC
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    score: Optional[SourceScore] = None


class Evidence(BaseModel):
    id: str = ""
    source_url: str
    source_title: str
    platform: Optional[SurfPlatform] = None
    excerpt: str
    claim: str
    timestamp: Optional[str] = None
    access_mode: SearchAccessMode = SearchAccessMode.PUBLIC
    confidence: float = 0.6


# ── Research ──────────────────────────────────────────────────────────

class SurfSessionStatus(str, Enum):
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SurfAction(BaseModel):
    id: str
    kind: str  # "plan" | "search" | "open" | "navigate" | "extract" | etc.
    description: str
    status: str = "running"  # "running" | "done" | "error"
    provider: Optional[str] = None
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at: Optional[str] = None
    ms: Optional[int] = None
    detail: Optional[str] = None


class SurfPlan(BaseModel):
    objective: str
    queries: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    required_information: list[str] = Field(default_factory=list)
    verification_requirements: list[str] = Field(default_factory=list)
    depth: str = "standard"  # "quick" | "standard" | "deep" | "exhaustive"


class SurfSession(BaseModel):
    id: str
    user_id: str
    workspace_id: Optional[str] = None
    objective: str
    plan: Optional[SurfPlan] = None
    sources: list[Source] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    actions: list[SurfAction] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    status: SurfSessionStatus = SurfSessionStatus.RUNNING
    error: Optional[str] = None


class SurfLimits(BaseModel):
    max_searches: int = 8
    max_pages: int = 12
    max_navigation_depth: int = 3
    max_runtime_ms: int = 120_000
    max_tokens: int = 24_000
    max_concurrent_pages: int = 2


DEPTH_LIMITS: dict[str, dict[str, int]] = {
    "quick": {"max_searches": 2, "max_pages": 3, "max_navigation_depth": 1, "max_runtime_ms": 30_000, "max_tokens": 8_000},
    "standard": {"max_searches": 5, "max_pages": 8, "max_navigation_depth": 2, "max_runtime_ms": 75_000, "max_tokens": 16_000},
    "deep": {"max_searches": 10, "max_pages": 16, "max_navigation_depth": 3, "max_runtime_ms": 180_000, "max_tokens": 32_000},
    "exhaustive": {"max_searches": 20, "max_pages": 30, "max_navigation_depth": 5, "max_runtime_ms": 420_000, "max_tokens": 64_000},
}


# ── Fetch ─────────────────────────────────────────────────────────────

class FetchResult(BaseModel):
    ok: bool
    status: int
    url: str
    text: str = ""
    content_type: str = ""
    blocked: bool = False
    attempts: int = 0
    error: Optional[str] = None
    fetched_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── Browser ───────────────────────────────────────────────────────────

class BrowserActionResult(BaseModel):
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
