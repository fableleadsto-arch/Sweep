"""Versioned wire contracts for the Relay cross-service boundary.

These are the machine-readable contracts shared by the Python Brain, the Java
Runtime and the Node API gateway. They describe how intelligence (Brain),
execution (Runtime) and the gateway exchange work:

    Node gateway  ── plan/agent turn ──▶  Python Brain (orchestrator)
    Python Brain ── tool_execution ─────▶  Tool registry / compute engine
    Python Brain ── task_envelope ──────▶  Java Runtime (jobs, events)
    Java Runtime ── runtime_event ──────▶  Node gateway (SSE) / frontend

Rules:
  * Every shape is JSON-serializable and versioned (``PROTOCOL_VERSION``).
  * Unknown fields are tolerated on read (forward-compatible) but new required
    fields bump the version.
  * The Zod mirror lives at ``src/lib/relay-schema/runtime.ts`` — keep in sync.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

PROTOCOL_VERSION = "1"

# ─────────────────────────────────────────────────────────────
#  Task lifecycle (the atomic unit of execution)
# ─────────────────────────────────────────────────────────────


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskEnvelope(BaseModel):
    """A unit of work submitted to the Java Runtime (or the Brain).

    ``type`` discriminates the workload: ``tool_execution`` for a single tool
    call, ``workflow_run`` for a Relay Flow execution, ``job`` for scheduled
    background work.
    """

    task_id: str = Field(description="Unique id, e.g. relay_task_<uuid>")
    type: Literal["tool_execution", "workflow_run", "job"]
    status: TaskStatus = TaskStatus.QUEUED
    user_id: str = ""
    workspace_id: Optional[str] = None
    parent_id: Optional[str] = None
    priority: int = Field(default=0, ge=0, le=100)
    retry_count: int = 0
    max_retries: int = Field(default=0, ge=0, le=10)
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    created_at: str = ""
    updated_at: str = ""
    callback: dict[str, Any] = Field(
        default_factory=dict,
        description="Where to report completion (e.g. {type: 'webhook', url: ...}).",
    )


# ─────────────────────────────────────────────────────────────
#  Tool execution
# ─────────────────────────────────────────────────────────────


class ToolSpec(BaseModel):
    """Machine-readable description of one executable capability."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionRequest(BaseModel):
    task_id: str = ""
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="user_id / workspace_id / workspace snapshot the tool may read.",
    )
    timeout_ms: int = Field(default=20_000, ge=1_000, le=300_000)


class ToolExecutionResponse(BaseModel):
    task_id: str = ""
    tool: str
    ok: bool
    output: Any = None
    error: str = ""
    latency_ms: int = 0
    model: str = ""
    provider: str = ""


# ─────────────────────────────────────────────────────────────
#  Brain planning
# ─────────────────────────────────────────────────────────────


class PlanStep(BaseModel):
    """One step of a decomposition plan (spec §5)."""

    step_id: str = ""
    kind: Literal["tool", "subtask", "respond"] = "tool"
    tool: str = ""
    description: str = ""
    input: dict[str, Any] = Field(default_factory=dict)
    status: str = "planned"  # planned | running | ok | failed | skipped
    output: Any = None
    error: str = ""


class BrainPlanRequest(BaseModel):
    user_id: str = ""
    workspace_id: Optional[str] = None
    message: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    max_steps: int = Field(default=8, ge=1, le=16)


class BrainPlanResponse(BaseModel):
    plan_id: str
    message: str
    summary: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    model: str = ""
    provider: str = ""
    protocol: str = PROTOCOL_VERSION


class ChatPlanRequest(BaseModel):
    """One request for the deterministic planner (zero LLM cost)."""

    user_id: str = ""
    workspace_id: Optional[str] = None
    message: str = Field(min_length=1)
    history: list[dict[str, Any]] = Field(default_factory=list)
    force_grounded: Optional[bool] = None
    force_model: Optional[str] = None
    context: dict[str, Any] = Field(default_factory=dict)


class ChatPlanResponse(BaseModel):
    """The routing decision: intent, task, model, grounding and knobs.

    Port of the TS ``ChatPlan`` from ``src/RelAI/core/planner.server.ts``.
    """

    protocol: str = PROTOCOL_VERSION
    intent: str = ""
    task_type: str = ""
    priority: str = ""
    model: str = ""
    grounded: bool = False
    needs_memory: bool = False
    needs_knowledge: bool = False
    needs_overview: bool = False
    temperature: float = 0.0
    max_tokens: int = 0
    rationale: str = ""


# ─────────────────────────────────────────────────────────────
#  Agent turn (plan → execute → observe → evaluate → finish)
# ─────────────────────────────────────────────────────────────


class AgentTurnRequest(BaseModel):
    user_id: str = ""
    workspace_id: Optional[str] = None
    message: str = Field(min_length=1)
    history: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    tools: list[str] = Field(
        default_factory=list,
        description="Whitelist of tool names; empty = full registry.",
    )
    max_steps: int = Field(default=8, ge=1, le=16)


class AgentTurnResponse(BaseModel):
    turn_id: str
    text: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
    model: str = ""
    provider: str = ""
    latency_ms: int = 0
    protocol: str = PROTOCOL_VERSION


# ─────────────────────────────────────────────────────────────
#  Runtime events (the event bus abstraction, spec §11)
# ─────────────────────────────────────────────────────────────


class RuntimeEvent(BaseModel):
    """A structured event emitted by the runtime or the brain.

    Mirrors the spec's stream: USER_MESSAGE_RECEIVED → TASK_CREATED →
    PLAN_CREATED → TOOL_REQUESTED → TOOL_EXECUTING → TOOL_COMPLETED →
    RESULT_EVALUATED → TASK_COMPLETED → RESPONSE_STREAMED.
    """

    event_type: str = Field(description="e.g. task.queued, tool.completed")
    task_id: str = ""
    user_id: str = ""
    workspace_id: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = ""
    protocol: str = PROTOCOL_VERSION
