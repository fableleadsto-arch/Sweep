"""Request/response models for the brain service API.

These are the public wire contract. Keeping them explicit makes the API
self-documenting (FastAPI generates OpenAPI from them) and keeps clients and
the Python service honest about shapes.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TurnRequest(BaseModel):
    """One turn of the companion conversation loop."""

    user_id: str
    workspace_id: Optional[str] = None
    message: str = Field(min_length=1)
    history: list[dict[str, Any]] = Field(default_factory=list)


class GenerativeUI(BaseModel):
    """Optional structured UI payload injected into the chat response."""

    type: Literal["chart", "card", "table", "metric"]
    component: str


class N8nPayload(BaseModel):
    """Automation dispatch payload (requires human approval by default)."""

    webhook: str
    body: dict[str, Any] = Field(default_factory=dict)


class SourceRef(BaseModel):
    """A citation pointing at a knowledge source."""

    name: str
    type: str = "text"
    chunk_index: int = 0


class TurnResponse(BaseModel):
    """The brain's output for a single turn."""

    text: str
    tone: str = "warm"
    generative_ui: Optional[GenerativeUI] = None
    n8n_payload: Optional[N8nPayload] = None
    requires_approval: bool = True
    approval_id: Optional[str] = None
    agent_delegations: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    model: str = ""
    provider: str = ""
    cost_estimate: float = 0.0
    context_sources: int = 0


class MemoryEntry(BaseModel):
    """A single persistent memory."""

    id: str
    user_id: str
    workspace_id: Optional[str] = None
    content: str
    kind: str = "fact"
    tags: list[str] = Field(default_factory=list)
    source: str = "relai"
    confidence: float = 0.7
    created_at: str = ""
    updated_at: str = ""
    expires_at: Optional[str] = None


class MemorySearchRequest(BaseModel):
    user_id: str
    workspace_id: Optional[str] = None
    query: str = ""
    limit: int = Field(default=8, ge=1, le=50)


class MemoryRememberRequest(BaseModel):
    user_id: str
    workspace_id: Optional[str] = None
    content: str = Field(min_length=1)
    kind: str = "fact"
    tags: list[str] = Field(default_factory=list)
    source: str = "relai"
    confidence: float = Field(default=0.7, ge=0, le=1)
    expires_at: Optional[str] = None


class RagRequest(BaseModel):
    """Retrieve relevant knowledge-base context for a query."""

    workspace_id: str
    query: str = Field(min_length=1)
    max_chunks: int = Field(default=6, ge=1, le=20)
    min_score: float = Field(default=0.45, ge=0, le=1)


class RagResult(BaseModel):
    context: str
    found: bool
    source_count: int = 0
    sources: list[SourceRef] = Field(default_factory=list)


class OverviewRequest(BaseModel):
    """Assemble the daily overview briefing for a user's workspace."""

    user_id: str
    workspace_id: Optional[str] = None
    briefing_type: Literal["morning", "evening"] = "morning"


class OverviewResponse(BaseModel):
    briefing: str


class ContextRequest(BaseModel):
    """Gather the live relay workspace context a brain turn would use."""

    user_id: str
    workspace_id: Optional[str] = None
    message: str = ""


class ContextResponse(BaseModel):
    """The relay context bundle, exposed for debugging and client prefetch."""

    profile: dict[str, Any]
    overview_requested: bool
    overview: str
    workspace: dict[str, Any]
    graph_facts: list[str]
    companion_tasks: list[dict[str, Any]]


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)


class EmbedResult(BaseModel):
    embeddings: list[Optional[list[float]]]


class ComputeRequest(BaseModel):
    """One capability-engine task — Relay picks the right framework for it.

    ``capability`` optionally overrides auto-selection. ``data`` accepts a CSV
    string, rows of dicts, a list of numbers, a text string, or a dict with
    ``rows``/``columns``; ``image_base64`` carries a data URL/raw base64 image
    for the vision capability.
    """

    task: str = Field(min_length=1, max_length=4000)
    capability: str = ""
    data: Any = None
    image_base64: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


class CapabilityInfo(BaseModel):
    """One entry in the discoverable capability catalog."""

    id: str
    label: str
    description: str
    available: bool
    libraries: list[str] = Field(default_factory=list)
    available_libraries: list[str] = Field(default_factory=list)
    missing_libraries: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)


class ComputeResult(BaseModel):
    """The outcome of a capability-engine run."""

    capability: str
    ok: bool
    summary: str = ""
    result: Any = None
    libraries_used: list[str] = Field(default_factory=list)
    error: str = ""


class ExecuteRequest(BaseModel):
    """A generated Python script to run in the sandbox (spec §35).

    The script may import the scientific/data stack (NumPy, Pandas, SymPy,
    SciPy, scikit-learn, Matplotlib, NetworkX, Faker, ...) and may define a
    module-level ``result`` that is JSON-serialized back to the caller. ``env``
    is the JSON-safe input the script reads from the ``env`` variable. The
    sandbox refuses system/framework imports, dynamic loading and object-graph
    escape patterns, and enforces CPU/RAM/time/output caps.
    """

    code: str = Field(min_length=1, max_length=50_000)
    timeout_ms: int = Field(default=30_000, ge=1_000, le=120_000)
    env: dict[str, Any] = Field(default_factory=dict)


class ExecuteResult(BaseModel):
    """Structured outcome of one sandboxed execution."""

    ok: bool
    duration_ms: int = 0
    result: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    violations: list[str] = Field(default_factory=list)
    sandboxed: bool = True


class ProviderHealth(BaseModel):
    gemini: bool = False
    openai: bool = False
    ollama: bool = False
    anthropic: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = ""
    providers: ProviderHealth
    python: str = ""
    version: str = ""


class ErrorResponse(BaseModel):
    error: str


# ── Compute backends / diagnostics ──────────────────────────────────────

class ComputeBackendStatus(BaseModel):
    """One backend's live status for Settings -> Compute."""

    id: str
    label: str
    kind: str
    available: bool
    enabled: bool
    required_libraries: list[str] = Field(default_factory=list)
    missing_libraries: list[str] = Field(default_factory=list)
    version: str = ""
    reason: str = ""
    devices: list[dict[str, Any]] = Field(default_factory=list)
    install_hint: str = ""


class ComputeDiagnosticsResponse(BaseModel):
    """Full compute-environment report (diagnostics page + CLI)."""

    generated_at: str
    python: dict[str, Any]
    os: dict[str, str]
    cpu: dict[str, Any]
    memory: dict[str, Any]
    disk: dict[str, Any]
    gpus: list[dict[str, Any]]
    compute_devices: list[str] = Field(default_factory=list)
    frameworks: list[dict[str, Any]] = Field(default_factory=list)
    backends: list[dict[str, Any]] = Field(default_factory=list)
    smoke_tests: dict[str, dict[str, Any]] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    status: str = ""
    native_models: list[dict[str, Any]] = Field(default_factory=list)


class ComputeBackendToggleRequest(BaseModel):
    enabled: bool


class ComputeBackendInstallRequest(BaseModel):
    dry_run: bool = True


class ComputeInstallResult(BaseModel):
    ok: bool
    backend: str
    method: str = ""
    dry_run: bool = True
    wheel: str = ""
    profile: str = ""
    already_available: bool = False
    error: str = ""
    summary: str = ""


class ComputeScheduleRequest(BaseModel):
    """A compute task for the scheduler (dry-run: returns the chosen backend).

    Mirrors ComputeRequest but adds compute-layer hints (framework, device,
    model format, memory estimate) so the scheduler can reason hardware-aware.
    """

    task: str = Field(min_length=1, max_length=4000)
    capability: str = ""
    data: Any = None
    image_base64: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)
    framework: str = ""
    device_preference: str = ""
    model_format: str = ""
    estimated_memory_mb: Optional[float] = None
    precision: str = ""


class ComputeScheduleResponse(BaseModel):
    """The scheduler's verdict for a task."""

    task: str
    capability: str = ""
    backend: Optional[str] = None
    device: str = ""
    score: float = 0.0
    reason: str = ""
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)
    ok: bool = True


# ── Relay Native neural stack ───────────────────────────────────────────


class NativeModelRecord(BaseModel):
    """One registered native model with its real (computed) parameter count."""

    name: str
    version: str = ""
    architecture: str = "transformer"
    framework: str = "pytorch"
    parameters: int = 0
    context_length: int = 0
    training_dataset: str = ""
    status: str = "experimental"
    precision: str = "fp32"
    created_at: str = ""
    hardware: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    verified: bool = False
    path: str = ""
    scale: str = ""


class NativeRecommendResponse(BaseModel):
    """Hardware-aware recommendation: the largest scale that truly fits."""

    scale: str
    parameters: int = 0
    bytes_per_param: int = 4
    footprint_bytes: int = 0
    kv_cache_bytes: int = 0
    activations_bytes: int = 0
    total_bytes: int = 0
    available_bytes: int = 0
    fits: bool = False
    reason: str = ""
    mode: str = "train"


class NativeGenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    model: Optional[str] = None
    max_new_tokens: int = Field(default=64, ge=1, le=2048)
    temperature: float = Field(default=0.8, ge=0.0, le=3.0)
    top_k: int = Field(default=40, ge=0, le=500)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=3.0)
    stop_sequences: list[str] = Field(default_factory=list)
    seed: Optional[int] = None


class NativeGenerateResult(BaseModel):
    ok: bool = True
    error: str = ""
    text: str = ""
    prompt: str = ""
    input_tokens: int = 0
    generated_tokens: int = 0
    tokens_per_second: float = 0.0
    duration_s: float = 0.0
    model: str = ""
    source: str = "native"


class NativeRouteRequest(BaseModel):
    task: str = Field(min_length=1, max_length=120)
    model: Optional[str] = None
    text_length: int = Field(default=0, ge=0)


class NativeRouteResult(BaseModel):
    task: str
    source: str = "external"
    model: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
