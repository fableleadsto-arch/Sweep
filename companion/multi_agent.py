"""RelAI multi-agent orchestration engine — port of ``src/RelAI/core/multi-agent.server.ts``.

A LangGraph-inspired state machine for orchestrating agent workflows: a graph of
nodes with conditional edges, a shared ``AgentState`` that threads through every
node, human-in-the-loop interrupts, cycle detection and resumable checkpoints.

Everything is wired through injection seams (``AgentExecutionContext``) so the
engine runs with no API keys or external infra: research/search, approval,
tool execution and the final synthesis are all callables the caller provides.
This mirrors the TS module, which dynamically imports its capability modules.

The wire-facing names (``currentNode``, ``toolResults``, ...) intentionally keep
their TS spelling because ``finalState``/steps are surfaced to the Node gateway
and frontend unchanged.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Optional, Union

from .orchestrator import OrchestratorModel, ToolContext, ToolRegistry

# ─────────────────────────────────────────────────────────────
#  Types (TS-name parity on the wire-facing fields)
# ─────────────────────────────────────────────────────────────


class AgentNodeKind(str):
    """Node kinds the engine knows about (open enum — custom kinds allowed)."""

    START = "start"
    CLASSIFY = "classify"
    RESEARCH = "research"
    DRAFT = "draft"
    EXECUTE_TOOL = "execute_tool"
    SUB_AGENT = "sub_agent"
    APPROVAL = "approval"
    N8N_DISPATCH = "n8n_dispatch"
    RESPOND = "respond"
    END = "end"


@dataclass
class AgentState:
    """Shared context that threads through all nodes (port of ``AgentState``)."""

    currentNode: str = "start"
    messages: list[dict[str, str]] = field(default_factory=list)
    toolResults: list[dict[str, Any]] = field(default_factory=list)  # {tool, ok, data}
    subAgentResults: dict[str, Any] = field(default_factory=dict)
    pendingApprovalId: Optional[str] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


NodeExecute = Callable[[AgentState, "AgentExecutionContext"], Awaitable[AgentState]]


@dataclass
class AgentNode:
    """One step in the graph (port of ``AgentNode``)."""

    kind: str
    name: str
    execute: NodeExecute


@dataclass
class AgentEdge:
    """A routing rule between nodes; ``to`` may be state-conditional (port of ``AgentEdge``)."""

    from_node: str
    to: Union[str, Callable[[AgentState], str]]


@dataclass
class AgentGraph:
    """The overall workflow definition (port of ``AgentGraph``)."""

    name: str
    description: str
    nodes: dict[str, AgentNode]
    edges: list[AgentEdge]


@dataclass
class AgentExecutionContext:
    """Injected capabilities the nodes call. No external infra required."""

    user_id: str = ""
    workspace_id: str = ""
    snapshot: dict[str, Any] = field(default_factory=dict)
    registry: Optional[ToolRegistry] = None
    model: Optional[OrchestratorModel] = None
    responder: Optional[Callable[[str, list[dict[str, Any]]], Awaitable[str]]] = None
    search: Optional[Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]] = None
    create_approval: Optional[Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = None
    plan_automation: Optional[Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]] = None
    persist: Optional[Callable[["AgentRunResult", dict[str, Any]], Awaitable[Any]]] = None
    system_prompt: str = (
        "You are RelAI, the resident intelligence of Relay. Answer using only the "
        "evidence gathered by the workflow; cite sources and name any gap."
    )


@dataclass
class AgentRunResult:
    """Outcome of one graph execution (port of ``AgentRunResult``)."""

    success: bool
    text: str
    steps: list[dict[str, Any]]  # {node, kind, ms, ok}
    finalState: AgentState
    interrupted: bool = False
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────
#  State helpers
# ─────────────────────────────────────────────────────────────


def _next(
    state: AgentState,
    current_node: str,
    *,
    messages: Optional[list[dict[str, str]]] = None,
    tool_results: Optional[list[dict[str, Any]]] = None,
    sub_agent_results: Optional[dict[str, Any]] = None,
    pending_approval_id: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> AgentState:
    """Advance to a node, incrementing the step counter (spread-merge port)."""
    meta = dict(state.metadata)
    meta["steps"] = int(meta.get("steps", 0)) + 1
    if metadata:
        meta.update(metadata)
    return replace(
        state,
        currentNode=current_node,
        messages=messages if messages is not None else state.messages,
        toolResults=tool_results if tool_results is not None else state.toolResults,
        subAgentResults=sub_agent_results if sub_agent_results is not None else state.subAgentResults,
        pendingApprovalId=(
            pending_approval_id if pending_approval_id is not None else state.pendingApprovalId
        ),
        error=error if error is not None else state.error,
        metadata=meta,
    )


def _first_user_message(state: AgentState) -> str:
    """The operator's original request — never node bookkeeping markers."""
    for message in state.messages:
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


# ─────────────────────────────────────────────────────────────
#  Node implementations
# ─────────────────────────────────────────────────────────────


async def classify_node(state: AgentState, ctx: AgentExecutionContext) -> AgentState:
    """Label the request with a task type and record the marker."""
    from .routing import classify_task

    last = state.messages[-1].get("content", "") if state.messages else ""
    task_type = classify_task(last)
    return _next(
        state,
        "classify",
        messages=[*state.messages, {"role": "system", "content": f"[Classified as: {task_type}]"}],
    )


async def research_node(state: AgentState, ctx: AgentExecutionContext) -> AgentState:
    """Run a web research leg on the operator's original request."""
    query = _first_user_message(state)
    if ctx.search is None:
        tool_result = {
            "tool": "web_search",
            "ok": False,
            "data": {"error": "no search engine configured in this deployment"},
        }
        content = "Research results: [search unavailable — no search engine configured]"
    else:
        try:
            res = await ctx.search(query, {"limit": 5})
        except Exception as exc:  # noqa: BLE001 — research is best-effort
            res = {"error": str(exc)}
        hits = res.get("hits") if isinstance(res, dict) else None
        if hits is None and isinstance(res, dict):
            hits = res.get("results") or []
        hits = hits if isinstance(hits, list) else []
        tool_result = {"tool": "web_search", "ok": bool(hits), "data": hits[:3]}
        content = f"Research results: {json.dumps(hits[:3], default=str)}"
    return _next(
        state,
        "research",
        messages=[*state.messages, {"role": "tool", "content": content}],
        tool_results=[*state.toolResults, tool_result],
    )


def _pick_tool(user_msg: str) -> tuple[str, dict[str, Any]]:
    """Heuristic tool selection for the execute-tool node (port of TS logic).

    Full LLM-driven function dispatch lives in the agent loop; this is the
    graph's deterministic fallback for orchestration requests.
    """
    if user_msg.startswith(("http://", "https://")):
        return "read_url", {"url": user_msg}
    if re.search(r"\b(osint|sweep|company|domain|founders|funding|email)\b", user_msg, re.I):
        target = re.sub(
            r"\b(osint|sweep|research|analyze|the|company)\b", "", user_msg, flags=re.I
        ).strip()[:200]
        return "osint_sweep", {"target": target, "readPages": 2}
    return "web_search", {"query": user_msg[:200]}


async def execute_tool_node(state: AgentState, ctx: AgentExecutionContext) -> AgentState:
    """Run one tool via the registry using the deterministic heuristic."""
    user_msg = _first_user_message(state)
    tool_name, tool_args = _pick_tool(user_msg)
    if ctx.registry is None:
        ok, data = False, {"error": "no tool registry configured in this deployment"}
    else:
        res = await ctx.registry.execute(
            tool_name, tool_args, ToolContext(ctx.user_id, ctx.workspace_id, ctx.snapshot)
        )
        ok, data = res.ok, res.output if res.ok else {"error": res.error}
    return _next(
        state,
        "execute_tool",
        messages=[*state.messages, {"role": "tool", "content": json.dumps(data, default=str)[:1000]}],
        tool_results=[*state.toolResults, {"tool": tool_name, "ok": ok, "data": data}],
    )


async def approval_node(state: AgentState, ctx: AgentExecutionContext) -> AgentState:
    """Request human approval before a mutating action (human-in-the-loop)."""
    if ctx.create_approval is None:
        return _next(
            state,
            "approval",
            messages=[
                *state.messages,
                {"role": "system", "content": "[approval skipped: no approval infrastructure configured]"},
            ],
        )
    try:
        approval = await ctx.create_approval(
            {
                "created_by": ctx.user_id,
                "required_approver": ctx.user_id,
                "action_kind": "other",
                "title": "Multi-agent workflow approval",
                "description": (
                    f'The agent workflow "{state.currentNode}" requires your approval before proceeding.'
                ),
                "action_payload": {"state": json.dumps(state.messages[-3:], default=str)},
                "expires_in_ms": 600_000,  # 10 minutes
            }
        )
        pending_id = approval.get("id") if isinstance(approval, dict) else None
    except Exception as exc:  # noqa: BLE001 — approval failure is a state error
        return _next(
            state,
            "approval",
            messages=[*state.messages, {"role": "system", "content": f"[approval failed: {exc}]"}],
        )
    return _next(
        state,
        "approval",
        pending_approval_id=pending_id,
        metadata={"interrupted": True},
    )


async def n8n_dispatch_node(state: AgentState, ctx: AgentExecutionContext) -> AgentState:
    """Dispatch the workflow result to an n8n webhook (skips when unconfigured)."""
    import httpx

    webhook_base = os.environ.get("N8N_WEBHOOK_BASE") or os.environ.get("N8N_BASE_URL")
    if not webhook_base:
        return _next(
            state,
            "n8n_dispatch",
            messages=[
                *state.messages,
                {"role": "system", "content": "[n8n dispatch skipped: N8N_WEBHOOK_BASE not configured]"},
            ],
        )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                webhook_base,
                json={
                    "messages": state.messages[-5:],
                    "toolResults": state.toolResults[-3:],
                },
            )
            response.raise_for_status()
        data: dict[str, Any] = {"status": response.status_code}
        ok = True
    except Exception as exc:  # noqa: BLE001 — n8n failure is recorded, not fatal
        data, ok = {"error": str(exc)}, False
    return _next(
        state,
        "n8n_dispatch",
        messages=[*state.messages, {"role": "tool", "content": f"n8n dispatched: {data}"}],
        tool_results=[*state.toolResults, {"tool": "n8n_trigger", "ok": ok, "data": data}],
    )


async def respond_node(state: AgentState, ctx: AgentExecutionContext) -> AgentState:
    """Synthesize the final answer from the evidence the graph collected."""
    synthesis = "[Response generated — completing workflow.]"
    if state.toolResults:
        request = _first_user_message(state)
        evidence = "\n\n".join(
            f"[{t.get('tool')} {'ok' if t.get('ok') else 'failed'}] {json.dumps(t.get('data'), default=str)[:800]}"
            for t in state.toolResults[-6:]
        )
        prompt = (
            "Answer the request below using ONLY the evidence gathered. Be direct, cite "
            "sources, and name any gap.\n\n"
            f"REQUEST: {request}\n\nEVIDENCE:\n{evidence}"
        )
        if ctx.responder is not None:
            try:
                answer = (await ctx.responder(system=ctx.system_prompt, messages=[{"role": "user", "content": prompt}])).strip()
                if answer:
                    synthesis = answer
            except Exception:  # noqa: BLE001 — synthesis is best-effort
                pass
        elif ctx.model is not None:
            try:
                plan = await ctx.model.plan(
                    system=ctx.system_prompt, messages=[{"role": "user", "content": prompt}], tools=[], iteration=0
                )
                synthesis = str(plan.get("respond") or plan.get("text") or synthesis).strip()
            except Exception:  # noqa: BLE001 — synthesis is best-effort
                pass
    return _next(
        state,
        "respond",
        messages=[*state.messages, {"role": "assistant", "content": synthesis}],
    )


# ─────────────────────────────────────────────────────────────
#  Graph builders
# ─────────────────────────────────────────────────────────────


def build_default_graph() -> AgentGraph:
    """Default orchestration: classify → research → approve → execute → respond.

    Read-only helpers only; the approval node sets ``interrupted`` which makes
    the executor return before the approval edge is followed. ``resume_graph``
    clears the interrupt and re-enters at the current node.
    """
    nodes: dict[str, AgentNode] = {
        "start": AgentNode(
            kind=AgentNodeKind.START,
            name="Start",
            execute=lambda state, ctx: _async_start(state, AgentNodeKind.START),
        ),
        "classify": AgentNode(kind=AgentNodeKind.CLASSIFY, name="Classify Intent", execute=classify_node),
        "research": AgentNode(kind=AgentNodeKind.RESEARCH, name="Web Research", execute=research_node),
        "execute_tool": AgentNode(
            kind=AgentNodeKind.EXECUTE_TOOL, name="Execute Tool", execute=execute_tool_node
        ),
        "approval": AgentNode(kind=AgentNodeKind.APPROVAL, name="Human Approval", execute=approval_node),
        "n8n_dispatch": AgentNode(
            kind=AgentNodeKind.N8N_DISPATCH, name="n8n Dispatch", execute=n8n_dispatch_node
        ),
        "respond": AgentNode(kind=AgentNodeKind.RESPOND, name="Respond", execute=respond_node),
        "end": AgentNode(
            kind=AgentNodeKind.END,
            name="End",
            execute=lambda state, ctx: _async_end(state),
        ),
    }
    edges = [
        AgentEdge("start", "classify"),
        AgentEdge(
            "classify",
            lambda state: "approval" if state.toolResults else "research",
        ),
        AgentEdge("research", "approval"),
        AgentEdge(
            "approval",
            lambda state: "end" if state.pendingApprovalId else "execute_tool",
        ),
        AgentEdge(
            "execute_tool",
            lambda state: "respond" if state.metadata.get("steps", 0) >= state.metadata.get("maxSteps", 12) else "research",
        ),
        AgentEdge("respond", "n8n_dispatch"),
        AgentEdge("n8n_dispatch", "end"),
    ]
    return AgentGraph(
        name="default-multi-agent",
        description="Default multi-agent orchestration: classify → research → approve → execute → respond",
        nodes=nodes,
        edges=edges,
    )


def build_chat_graph() -> AgentGraph:
    """Chat-oriented graph for orchestration-style chat requests.

    Unlike the automation graph it has NO approval interrupt (read-only
    research/analysis needs no gate) and loops research → execute until the
    step budget is spent, then synthesizes via ``respond_node``.
    """
    nodes: dict[str, AgentNode] = {
        "start": AgentNode(
            kind=AgentNodeKind.START,
            name="Start",
            execute=lambda state, ctx: _async_start(state, AgentNodeKind.START),
        ),
        "classify": AgentNode(kind=AgentNodeKind.CLASSIFY, name="Classify Intent", execute=classify_node),
        "research": AgentNode(kind=AgentNodeKind.RESEARCH, name="Web Research", execute=research_node),
        "execute_tool": AgentNode(
            kind=AgentNodeKind.EXECUTE_TOOL, name="Execute Tool", execute=execute_tool_node
        ),
        "respond": AgentNode(kind=AgentNodeKind.RESPOND, name="Synthesize Answer", execute=respond_node),
        "end": AgentNode(
            kind=AgentNodeKind.END,
            name="End",
            execute=lambda state, ctx: _async_end(state),
        ),
    }
    edges = [
        AgentEdge("start", "classify"),
        AgentEdge("classify", lambda state: "execute_tool" if state.toolResults else "research"),
        AgentEdge("research", "execute_tool"),
        AgentEdge(
            "execute_tool",
            lambda state: "respond" if state.metadata.get("steps", 0) >= state.metadata.get("maxSteps", 12) else "research",
        ),
        AgentEdge("respond", "end"),
    ]
    return AgentGraph(
        name="chat-multi-agent",
        description="Chat orchestration: classify → research → execute → synthesize",
        nodes=nodes,
        edges=edges,
    )


async def _async_start(state: AgentState, kind: str) -> AgentState:
    return replace(state, currentNode=kind, metadata={**state.metadata, "steps": 0})


async def _async_end(state: AgentState) -> AgentState:
    return replace(state, currentNode=AgentNodeKind.END)


# ─────────────────────────────────────────────────────────────
#  Graph executor
# ─────────────────────────────────────────────────────────────


def _final_text(state: AgentState) -> str:
    """Prefer the last assistant synthesis over raw tool noise."""
    for message in reversed(state.messages):
        if message.get("role") == "assistant":
            return message.get("content", "")
    tail = [m.get("content", "") for m in state.messages if m.get("role") in ("assistant", "tool")]
    return "\n".join(tail[-3:]) if tail else ""


async def _persist_best_effort(
    ctx: AgentExecutionContext, result: AgentRunResult, message: str, graph_name: str
) -> None:
    if ctx.persist is None:
        return
    try:
        await ctx.persist(
            result,
            {
                "message": message,
                "user_id": ctx.user_id,
                "workspace_id": ctx.workspace_id,
                "graph_name": graph_name,
            },
        )
    except Exception:  # noqa: BLE001 — recording must never break the run
        pass


async def run_graph_loop(
    initial_state: AgentState,
    graph: AgentGraph,
    ctx: AgentExecutionContext,
) -> AgentRunResult:
    """Shared execution loop for graph workflows (port of ``runGraphLoop``).

    Runs nodes, checks interrupts, routes via edges and detects cycles. Both
    ``execute_graph`` and ``resume_graph`` delegate here.
    """
    step_log: list[dict[str, Any]] = []
    state = initial_state
    visited: set[str] = set()

    while state.currentNode != AgentNodeKind.END:
        # Budget guard. "respond"/"end" are exempt so an orchestrated run always
        # reaches its synthesis instead of stopping on raw tool output — the TS
        # loop breaks at the top of the iteration, which can strand the graph
        # before the respond node ever executes (deliberate port fix).
        if (
            int(state.metadata.get("steps", 0)) >= int(state.metadata.get("maxSteps", 12))
            and state.currentNode not in ("respond", AgentNodeKind.END)
        ):
            state = replace(
                state,
                messages=[*state.messages, {"role": "system", "content": "[Max steps reached — ending workflow.]"}],
            )
            break

        node = graph.nodes.get(state.currentNode)
        if node is None:
            state = replace(state, error=f"Unknown node: {state.currentNode}")
            break

        cycle_key = f"{state.currentNode}-{state.metadata.get('steps')}"
        if cycle_key in visited:
            state = replace(state, error=f"Cycle detected at node: {state.currentNode}")
            break
        visited.add(cycle_key)

        started = time.monotonic()
        try:
            state = await node.execute(state, ctx)
            step_log.append({"node": node.name, "kind": node.kind, "ms": _ms_since(started), "ok": True})
        except Exception as exc:  # noqa: BLE001 — node failures break the run
            step_log.append({"node": node.name, "kind": node.kind, "ms": _ms_since(started), "ok": False})
            state = replace(state, error=str(exc))
            break

        if state.metadata.get("interrupted"):
            result = AgentRunResult(
                success=True,
                text=f"Workflow paused for approval (ID: {state.pendingApprovalId}). Resume to continue.",
                steps=step_log,
                finalState=state,
                interrupted=True,
            )
            await _persist_best_effort(ctx, result, _first_user_message(state), graph.name)
            return result

        edge = next((e for e in graph.edges if e.from_node == state.currentNode), None)
        if edge is None:
            state = replace(state, currentNode=AgentNodeKind.END)
            break
        target = edge.to(state) if callable(edge.to) else edge.to
        state = replace(state, currentNode=target)

    text = _final_text(state)
    result = AgentRunResult(
        success=state.error is None,
        text=text or "Workflow completed.",
        steps=step_log,
        finalState=state,
        interrupted=False,
        error=state.error,
    )
    await _persist_best_effort(ctx, result, _first_user_message(state), graph.name)
    return result


def _ms_since(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


async def execute_graph(
    graph: AgentGraph,
    ctx: AgentExecutionContext,
    input_: dict[str, Any],
) -> AgentRunResult:
    """Execute a graph from start to end (port of ``executeGraph``)."""
    max_steps = int(input_.get("maxSteps", 12))
    initial_state = AgentState(
        currentNode=AgentNodeKind.START,
        messages=[{"role": "user", "content": input_.get("message", "")}],
        metadata={
            "startedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "userId": ctx.user_id,
            "workspaceId": ctx.workspace_id,
            "model": input_.get("model", "gemini-2.5-flash"),
            "steps": 0,
            "maxSteps": max_steps,
            "interrupted": False,
        },
    )
    return await run_graph_loop(initial_state, graph, ctx)


async def resume_graph(
    graph: AgentGraph,
    ctx: AgentExecutionContext,
    previous_state: AgentState,
) -> AgentRunResult:
    """Resume an interrupted run: clear the interrupt, re-enter at current node."""
    state = replace(
        previous_state,
        pendingApprovalId=None,
        metadata={**previous_state.metadata, "interrupted": False},
    )
    return await run_graph_loop(state, graph, ctx)


# ─────────────────────────────────────────────────────────────
#  Sub-agent delegation (CrewAI pattern)
# ─────────────────────────────────────────────────────────────


ROLE_PROMPTS: dict[str, str] = {
    "strategist": "You are a Strategist agent. Break down goals into actionable plans.",
    "researcher": "You are a Research agent. Gather intelligence and find relevant information.",
    "copywriter": "You are a Copywriter agent. Write compelling, human-first outreach copy.",
    "reviewer": "You are a Reviewer agent. Check for quality, safety, and compliance.",
    "planner": "You are an Automation Planner agent. Design n8n workflows from goals.",
}


async def run_sub_agent(
    role: str,
    goal: str,
    state: AgentState,
    ctx: AgentExecutionContext,
) -> dict[str, Any]:
    """Run a role-based sub-agent (port of ``runSubAgent``)."""
    system_prompt = ROLE_PROMPTS.get(role, f"You are a {role} agent. Complete your assigned task.")
    prompt = (
        f"Goal: {goal}\n\nContext: {json.dumps({'messages': state.messages[-5:], 'toolResults': state.toolResults[-3:]}, default=str)}"
    )
    text = ""
    if ctx.responder is not None:
        try:
            text = (await ctx.responder(system=system_prompt, messages=[{"role": "user", "content": prompt}])).strip()
        except Exception:  # noqa: BLE001 — sub-agents are best-effort
            text = ""
    elif ctx.model is not None:
        try:
            plan = await ctx.model.plan(system=system_prompt, messages=[{"role": "user", "content": prompt}], tools=[], iteration=0)
            text = str(plan.get("respond") or plan.get("text") or "").strip()
        except Exception:  # noqa: BLE001 — sub-agents are best-effort
            text = ""
    return {"role": role, "goal": goal, "result": text, "sources": [], "model": ""}


async def run_sub_agent_team(
    assignments: list[dict[str, str]],
    state: AgentState,
    ctx: AgentExecutionContext,
) -> dict[str, Any]:
    """Run several sub-agents in parallel (port of ``runSubAgentTeam``)."""
    results: dict[str, Any] = {}

    async def run_one(assignment: dict[str, str]) -> None:
        result = await run_sub_agent(assignment["role"], assignment["goal"], state, ctx)
        results[assignment["role"]] = result

    await asyncio.gather(*(run_one(a) for a in assignments))
    return results


# ─────────────────────────────────────────────────────────────
#  Preset workflows
# ─────────────────────────────────────────────────────────────


def _with_custom_node(
    graph: AgentGraph, node_id: str, node: AgentNode, edges: list[AgentEdge]
) -> AgentGraph:
    nodes = dict(graph.nodes)
    nodes[node_id] = node
    return AgentGraph(name=graph.name, description=graph.description, nodes=nodes, edges=edges)


async def lead_research_workflow(
    ctx: AgentExecutionContext,
    input_: dict[str, Any],
) -> AgentRunResult:
    """Lead research: classify → OSINT sweep → approve → notify. (port of TS preset)"""
    graph = build_default_graph()
    company = input_.get("company", "")

    async def osint_execute(state: AgentState, _ctx: AgentExecutionContext) -> AgentState:
        sweep: dict[str, Any] = {}
        if ctx.search is not None:
            try:
                sweep = await ctx.search(company, {"readPages": 3})
            except Exception:  # noqa: BLE001 — best-effort
                sweep = {"error": "sweep failed"}
        elif ctx.registry is not None:
            res = await ctx.registry.execute(
                "osint_sweep",
                {"target": company, "readPages": 3},
                ToolContext(ctx.user_id, ctx.workspace_id, ctx.snapshot),
            )
            sweep = res.output if res.ok else {"error": res.error}
        hits = sweep.get("hits") if isinstance(sweep, dict) else None
        hits = hits if isinstance(hits, list) else []
        by_source = sweep.get("bySource") if isinstance(sweep, dict) else {}
        emails = sweep.get("emails") if isinstance(sweep, dict) else []
        emails = emails if isinstance(emails, list) else []
        sources = len(by_source) if isinstance(by_source, dict) else len(hits)
        return _next(
            state,
            "osint",
            tool_results=[*state.toolResults, {"tool": "osint_sweep", "ok": bool(hits), "data": sweep}],
            messages=[
                *state.messages,
                {"role": "tool", "content": f"OSINT: {len(emails)} emails, {sources} sources"},
            ],
        )

    edges = [
        AgentEdge("start", "classify"),
        AgentEdge("classify", "osint"),
        AgentEdge("osint", "approval"),
        AgentEdge("approval", "n8n_dispatch"),
        AgentEdge("n8n_dispatch", "end"),
    ]
    graph = _with_custom_node(
        graph, "osint", AgentNode(kind=AgentNodeKind.RESEARCH, name="OSINT Sweep", execute=osint_execute), edges
    )
    return await execute_graph(graph, ctx, {"message": f"Research company: {company}", "maxSteps": 8})


async def outreach_workflow(
    ctx: AgentExecutionContext,
    input_: dict[str, Any],
) -> AgentRunResult:
    """Outreach drafting: classify → research → draft (copywriter) → approve. (port of TS preset)"""
    graph = build_default_graph()
    prospect = input_.get("prospectName", "")
    excerpt = input_.get("prospectExcerpt", "")
    offer = input_.get("offer", "")
    platform = input_.get("platform", "")

    async def draft_execute(state: AgentState, _ctx: AgentExecutionContext) -> AgentState:
        sub_result = await run_sub_agent(
            "copywriter",
            f'Write a {platform} message for {prospect}. They said: "{excerpt}". Offer: {offer}',
            state,
            ctx,
        )
        return _next(
            state,
            "draft",
            sub_agent_results={**state.subAgentResults, "copywriter": sub_result},
            messages=[*state.messages, {"role": "assistant", "content": f"[Draft ready for {prospect}]"}],
        )

    edges = [
        AgentEdge("start", "classify"),
        AgentEdge("classify", "research"),
        AgentEdge("research", "draft"),
        AgentEdge("draft", "approval"),
        AgentEdge("approval", "end"),
    ]
    graph = _with_custom_node(
        graph, "draft", AgentNode(kind=AgentNodeKind.DRAFT, name="Draft Outreach", execute=draft_execute), edges
    )
    return await execute_graph(graph, ctx, {"message": f"Draft {platform} outreach for {prospect}", "maxSteps": 8})


async def automation_design_workflow(
    ctx: AgentExecutionContext,
    input_: dict[str, Any],
) -> AgentRunResult:
    """Automation design: classify → design (planner) → approve → deploy. (port of TS preset)"""
    graph = build_default_graph()
    goal = input_.get("goal", "")

    async def design_execute(state: AgentState, _ctx: AgentExecutionContext) -> AgentState:
        plan: dict[str, Any] = {"name": "unplanned", "description": goal, "approvalGate": "unknown"}
        if ctx.plan_automation is not None:
            try:
                plan = await ctx.plan_automation(input_)
            except Exception:  # noqa: BLE001 — best-effort
                plan = {"name": "unplanned", "description": goal, "approvalGate": "unknown"}
        name = plan.get("name") if isinstance(plan, dict) else "unplanned"
        return _next(
            state,
            "design",
            sub_agent_results={**state.subAgentResults, "automation_plan": plan},
            messages=[*state.messages, {"role": "assistant", "content": f"[Automation plan: {name}]"}],
        )

    edges = [
        AgentEdge("start", "classify"),
        AgentEdge("classify", "design"),
        AgentEdge("design", "approval"),
        AgentEdge("approval", "n8n_dispatch"),
        AgentEdge("n8n_dispatch", "end"),
    ]
    graph = _with_custom_node(
        graph, "design", AgentNode(kind=AgentNodeKind.SUB_AGENT, name="Design Automation", execute=design_execute), edges
    )
    return await execute_graph(graph, ctx, {"message": f"Design automation: {goal}", "maxSteps": 8})


def default_run_id() -> str:
    """Stable id generator for run records."""
    return uuid.uuid4().hex
