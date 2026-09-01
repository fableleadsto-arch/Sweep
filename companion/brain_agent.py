"""HTTP surface for the RelAI agent loop (Phase 2 step 3, Node delegation target).

Owns the ``/api/brain/agent/*`` routes:

* ``POST /api/brain/agent/turn``   - upgraded to the full ``relai_agent_run`` loop.
* ``POST /api/brain/agent/stream`` - SSE streaming of loop steps and the answer.

The ``/api/brain/agent/turn`` registered in ``companion.main`` was an earlier,
shallower orchestrator loop. Importing this module replaces that route in place
(no request/response contract change) and adds the stream endpoint, so the Node
side can delegate both agent and chat modes here.
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any, AsyncIterator, Literal, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent import AgentRuntime, Responder, relai_agent_run, relai_agent_stream
from .config import BrainSettings, get_settings
from .contracts import AgentTurnRequest, AgentTurnResponse, PlanStep
from .main import app, get_orchestrator, get_orchestrator_model, require_token
from .memory import MemoryService
from .orchestrator import (
    Orchestrator,
    OrchestratorModel,
    ProviderChainModel,
    _provider_meta,
    build_default_tools,
)
from .providers import ProviderChain
from .rag import RagService


class AgentStreamRequest(BaseModel):
    """CamelCase wire shape the Node delegate sends to the stream endpoint."""

    message: str = Field(min_length=1)
    history: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    userId: str = ""
    workspaceId: str = ""
    grounded: Optional[bool] = None
    mode: Literal["chat", "agent"] = "chat"
    maxSteps: Optional[int] = Field(default=None, ge=1, le=16)


class _ChainResponder:
    """Free-text responder backed by the provider chain (``json_mode=False``)."""

    def __init__(self, chain: ProviderChain) -> None:
        self._chain = chain

    async def __call__(self, system: str, messages: list[dict[str, Any]]) -> str:
        result = await self._chain.generate(
            system=system, messages=messages, json_mode=False
        )
        return (result.text or "").strip()


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _wire_step(step: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a step event with the camelCase key the Node client expects."""
    return {
        "tool": step.get("tool", ""),
        "argsPreview": step.get("args_preview", ""),
        "ok": step.get("ok", False),
        "summary": step.get("summary", ""),
        "ms": step.get("ms", 0),
    }


def _wire_event(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("type") == "step" and isinstance(event.get("step"), dict):
        return {"type": "step", "step": _wire_step(event["step"])}
    return event


def build_agent_runtime(
    settings: BrainSettings,
    model: Optional[OrchestratorModel],
    *,
    workspace_id: Optional[str] = None,
    responder: Optional[Responder] = None,
) -> AgentRuntime:
    """Assemble the agent runtime with memory + RAG wired to the Python services.

    ``model=None`` means chat-only (the model is only used for tool planning).
    Inject ``responder`` to replace the provider-chain text call (tests).
    """
    chain = ProviderChain(settings)
    memory_svc = MemoryService(settings)
    rag_svc = RagService(settings)

    async def memory_retriever(owner: str, query: str) -> str:
        return await memory_svc.build_memory_context(
            owner, query, workspace_id=workspace_id
        )

    async def rag_retriever(owner: str, query: str) -> str:
        try:
            result = await rag_svc.retrieve(workspace_id or "", query)
        except Exception:  # noqa: BLE001 - RAG is best-effort
            return ""
        return result.context or ""

    if responder is None:
        if isinstance(model, ProviderChainModel):
            responder = _ChainResponder(model.chain)
        elif model is None:
            responder = _ChainResponder(chain)

    return AgentRuntime(
        registry=build_default_tools(),
        model=model,
        responder=responder,
        memory_retriever=memory_retriever,
        rag_retriever=rag_retriever,
    )


async def run_turn_with_loop(
    request: AgentTurnRequest,
    model: OrchestratorModel,
    settings: BrainSettings,
) -> AgentTurnResponse:
    """Run the full agent loop and map the result onto the turn response."""
    started = time.monotonic()
    runtime = build_agent_runtime(settings, model, workspace_id=request.workspace_id)
    result = await relai_agent_run(
        message=request.message,
        runtime=runtime,
        user_id=request.user_id,
        workspace_id=request.workspace_id or "",
        history=request.history,
        context=request.context,
        max_steps=request.max_steps,
        tools_whitelist=request.tools or None,
    )
    steps = [
        PlanStep(
            step_id=f"step-{index + 1}",
            kind="tool",
            tool=step.tool,
            description=step.summary,
            input={},
            status="ok" if step.ok else "failed",
            error="" if step.ok else step.summary,
        )
        for index, step in enumerate(result.steps)
    ]
    provider, model_name = _provider_meta(model)
    return AgentTurnResponse(
        turn_id=f"turn-{secrets.token_hex(8)}",
        text=result.text,
        steps=steps,
        model=result.model or model_name,
        provider=provider,
        latency_ms=round((time.monotonic() - started) * 1000),
        protocol="1",
    )


async def stream_agent_events(
    request: AgentStreamRequest,
    model: OrchestratorModel,
    settings: BrainSettings,
    responder: Optional[Responder] = None,
) -> AsyncIterator[str]:
    """Yield SSE-framed events for the stream endpoint (chat or agent mode)."""
    chat_mode = request.mode == "chat"
    runtime = build_agent_runtime(
        settings,
        None if chat_mode else model,
        workspace_id=request.workspaceId or None,
        responder=responder,
    )
    try:
        if chat_mode:
            result = await relai_agent_run(
                message=request.message,
                runtime=runtime,
                user_id=request.userId,
                workspace_id=request.workspaceId or "",
                history=request.history,
                context=request.context,
                max_steps=request.maxSteps,
            )
            yield _sse({"type": "text", "text": result.text})
            if result.sources:
                yield _sse(
                    {
                        "type": "sources",
                        "sources": [
                            {"uri": s.uri, "title": s.title} for s in result.sources
                        ],
                    }
                )
            yield _sse({"type": "done"})
        else:
            async for event in relai_agent_stream(
                message=request.message,
                runtime=runtime,
                user_id=request.userId,
                workspace_id=request.workspaceId or "",
                history=request.history,
                context=request.context,
                max_steps=request.maxSteps,
            ):
                yield _sse(_wire_event(event))
    except Exception as exc:  # noqa: BLE001 - surfaced as an SSE error event
        yield _sse({"type": "error", "error": str(exc)})
        yield _sse({"type": "done"})


def get_agent_responder(
    settings: BrainSettings = Depends(get_settings),
) -> Optional[Responder]:
    """Override in tests to inject a fake free-text responder."""
    return None


router = APIRouter(prefix="/api/brain/agent", tags=["agent"])


@router.post(
    "/turn",
    response_model=AgentTurnResponse,
    dependencies=[Depends(require_token)],
)
async def agent_turn(
    request: AgentTurnRequest,
    settings: BrainSettings = Depends(get_settings),
    # Kept for contract compatibility with the previous shallow loop endpoint.
    orchestrator: Orchestrator = Depends(get_orchestrator),  # noqa: ARG002
    model: OrchestratorModel = Depends(get_orchestrator_model),
) -> AgentTurnResponse:
    return await run_turn_with_loop(request, model, settings)


@router.post("/stream", dependencies=[Depends(require_token)])
async def agent_stream(
    request: AgentStreamRequest,
    settings: BrainSettings = Depends(get_settings),
    model: OrchestratorModel = Depends(get_orchestrator_model),
    responder: Optional[Responder] = Depends(get_agent_responder),
) -> StreamingResponse:
    return StreamingResponse(
        stream_agent_events(request, model, settings, responder),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _install() -> None:
    """Replace the shallow main-app turn route with the full-loop implementation.

    Idempotent: if the original route is already gone this only adds the routes.
    """
    app.routes[:] = [
        route
        for route in app.routes
        if not (
            getattr(route, "path", None) == "/api/brain/agent/turn"
            and getattr(route, "methods", None)
            and "POST" in route.methods
        )
    ]
    app.include_router(router)


_install()
