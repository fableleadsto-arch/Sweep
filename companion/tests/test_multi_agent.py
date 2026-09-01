"""Tests for the multi-agent graph engine port (``companion/multi_agent.py``).

The engine is pure state machine — every capability (search, responder,
approval, persistence) is injected via ``AgentExecutionContext``, so tests run
with zero network/API keys. Graph runs must use odd ``maxSteps`` to reach the
``respond`` node (even budgets strand at ``execute_tool`` — TS parity).
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from companion.multi_agent import (
    AgentEdge,
    AgentExecutionContext,
    AgentGraph,
    AgentNode,
    AgentNodeKind,
    AgentState,
    _async_start,
    _next,
    approval_node,
    automation_design_workflow,
    build_chat_graph,
    build_default_graph,
    execute_graph,
    lead_research_workflow,
    outreach_workflow,
    resume_graph,
    run_sub_agent_team,
)
from companion.orchestrator import ToolRegistry


def _chat_ctx() -> AgentExecutionContext:
    async def search(query: str, opts: dict) -> dict:
        return {"hits": [{"url": "https://example.com/acme", "title": "Acme Corp"}]}

    async def responder(system: str, messages: list[dict]) -> str:
        return "Synthesis answer"

    registry = ToolRegistry()
    registry.register_fn(
        "web_search", "search", lambda i, ctx: {"results": [{"url": "https://example.com/acme", "title": "Acme Corp"}]}
    )
    registry.register_fn("read_url", "read", lambda i, ctx: {"url": i["url"], "title": "Acme Corp"})
    return AgentExecutionContext(
        user_id="u1", workspace_id="w1", search=search, responder=responder, registry=registry
    )


def _run(coro):
    return asyncio.run(coro)


# ── chat graph end-to-end ────────────────────────────────────


def test_chat_graph_end_to_end() -> None:
    async def run() -> None:
        ctx = _chat_ctx()
        result = await execute_graph(
            build_chat_graph(), ctx, {"message": "research competitors and then summarize", "maxSteps": 9}
        )
        assert result.success is True
        assert result.error is None
        assert result.text == "Synthesis answer"
        assert any(s["kind"] == "respond" for s in result.steps)
        assert len(result.finalState.toolResults) >= 4  # research + execute_tool loops

    _run(run())


def test_odd_budget_reaches_respond() -> None:
    async def run() -> None:
        ctx = _chat_ctx()
        result = await execute_graph(build_chat_graph(), ctx, {"message": "research competitors", "maxSteps": 9})
        assert result.success is True
        assert result.text == "Synthesis answer"
        # respond ran even though the step counter was already at/over budget.
        assert result.finalState.metadata["steps"] >= result.finalState.metadata["maxSteps"]

    _run(run())


def test_even_budget_strands_before_respond() -> None:
    async def run() -> None:
        ctx = _chat_ctx()
        result = await execute_graph(build_chat_graph(), ctx, {"message": "research competitors", "maxSteps": 8})
        assert result.success is True
        assert result.text != "Synthesis answer"
        assert "[Max steps reached" in result.finalState.messages[-1]["content"]

    _run(run())


# ── graph control-flow guards ────────────────────────────────


def test_cycle_detection() -> None:
    async def run() -> None:
        async def loop_back(s: AgentState, _c: AgentExecutionContext) -> AgentState:
            return replace(s, currentNode="loop")

        graph = AgentGraph(
            name="cycle-test",
            description="",
            nodes={
                "start": AgentNode(
                    AgentNodeKind.START, "Start", lambda s, c: _async_start(s, AgentNodeKind.START)
                ),
                "loop": AgentNode("loop", "Loop", loop_back),
            },
            edges=[AgentEdge("start", "loop"), AgentEdge("loop", "loop")],
        )
        result = await execute_graph(graph, AgentExecutionContext(), {"message": "x", "maxSteps": 12})
        assert result.success is False
        assert "Cycle detected" in (result.error or "")

    _run(run())


def test_max_steps_break_without_respond() -> None:
    async def run() -> None:
        async def spin(s: AgentState, _c: AgentExecutionContext) -> AgentState:
            return _next(s, "spin")

        graph = AgentGraph(
            name="spin",
            description="",
            nodes={
                "start": AgentNode(
                    AgentNodeKind.START, "Start", lambda s, c: _async_start(s, AgentNodeKind.START)
                ),
                "spin": AgentNode("spin", "Spin", spin),
            },
            edges=[AgentEdge("start", "spin"), AgentEdge("spin", "spin")],
        )
        result = await execute_graph(graph, AgentExecutionContext(), {"message": "x", "maxSteps": 4})
        assert result.success is True
        assert "[Max steps reached" in result.finalState.messages[-1]["content"]

    _run(run())


def test_unknown_node() -> None:
    async def run() -> None:
        graph = AgentGraph(
            name="u",
            description="",
            nodes={
                "start": AgentNode(
                    AgentNodeKind.START, "Start", lambda s, c: _async_start(s, AgentNodeKind.START)
                )
            },
            edges=[AgentEdge("start", "ghost")],
        )
        result = await execute_graph(graph, AgentExecutionContext(), {"message": "x", "maxSteps": 12})
        assert result.success is False
        assert "Unknown node: ghost" in (result.error or "")

    _run(run())


def test_node_failure_marks_step_failed() -> None:
    async def run() -> None:
        async def boom(_s: AgentState, _c: AgentExecutionContext) -> AgentState:
            raise RuntimeError("kaboom")

        graph = AgentGraph(
            name="f",
            description="",
            nodes={
                "start": AgentNode(
                    AgentNodeKind.START, "Start", lambda s, c: _async_start(s, AgentNodeKind.START)
                ),
                "boom": AgentNode("boom", "Boom", boom),
            },
            edges=[AgentEdge("start", "boom"), AgentEdge("boom", "end")],
        )
        result = await execute_graph(graph, AgentExecutionContext(), {"message": "x", "maxSteps": 12})
        assert result.success is False
        assert result.error == "kaboom"
        assert result.steps[-1]["ok"] is False

    _run(run())


# ── interrupt + resume (default graph) ───────────────────────


def test_interrupt_and_resume() -> None:
    async def run() -> None:
        calls = {"n": 0}

        async def create_approval(payload: dict) -> dict:
            calls["n"] += 1
            if calls["n"] == 1:
                return {"id": "appr_1"}
            raise RuntimeError("already decided")

        async def responder(system: str, messages: list[dict]) -> str:
            return "Approved and executed."

        ctx = AgentExecutionContext(
            user_id="u1", workspace_id="w1", responder=responder, create_approval=create_approval
        )

        first = await execute_graph(build_default_graph(), ctx, {"message": "research a competitor", "maxSteps": 8})
        assert first.interrupted is True
        assert first.finalState.pendingApprovalId == "appr_1"
        assert "appr_1" in first.text

        resumed = await resume_graph(build_default_graph(), ctx, first.finalState)
        assert resumed.success is True
        assert resumed.interrupted is False
        assert resumed.finalState.pendingApprovalId is None
        assert "Approved and executed." in resumed.text

    _run(run())


# ── preset workflows ─────────────────────────────────────────


def test_automation_design_workflow_preset() -> None:
    async def run() -> None:
        async def plan_automation(input_: dict) -> dict:
            return {"name": "Invoice reminder flow", "description": input_.get("goal"), "approvalGate": "manual"}

        ctx = AgentExecutionContext(user_id="u1", workspace_id="w1", plan_automation=plan_automation)
        result = await automation_design_workflow(ctx, {"goal": "chase unpaid invoices"})
        assert result.success is True
        assert any(s["node"] == "Design Automation" for s in result.steps)
        assert result.finalState.subAgentResults["automation_plan"]["name"] == "Invoice reminder flow"

    _run(run())


def test_lead_research_workflow_preset() -> None:
    async def run() -> None:
        async def search(query: str, opts: dict) -> dict:
            return {"hits": [{"url": "https://acme.com", "title": "Acme"}], "emails": ["ceo@acme.com"]}

        ctx = AgentExecutionContext(user_id="u1", workspace_id="w1", search=search)
        result = await lead_research_workflow(ctx, {"company": "Acme"})
        assert result.success is True
        assert any(s["node"] == "OSINT Sweep" for s in result.steps)
        assert result.finalState.toolResults[0]["tool"] == "osint_sweep"
        assert result.finalState.toolResults[0]["ok"] is True

    _run(run())


def test_outreach_workflow_preset() -> None:
    async def run() -> None:
        async def responder(system: str, messages: list[dict]) -> str:
            return "Hey, quick question about your stack."

        ctx = AgentExecutionContext(user_id="u1", workspace_id="w1", responder=responder)
        result = await outreach_workflow(
            ctx,
            {
                "prospectName": "Jane",
                "prospectExcerpt": "Looking for a booking tool",
                "offer": "free setup",
                "platform": "linkedin",
            },
        )
        assert result.success is True
        assert any(s["node"] == "Draft Outreach" for s in result.steps)
        assert result.finalState.subAgentResults["copywriter"]["result"] == "Hey, quick question about your stack."

    _run(run())


# ── sub-agents ───────────────────────────────────────────────


def test_sub_agent_team_runs_in_parallel() -> None:
    async def run() -> None:
        async def responder(system: str, messages: list[dict]) -> str:
            return system  # the ROLE_PROMPT identifies the role

        ctx = AgentExecutionContext(responder=responder)
        state = AgentState(messages=[{"role": "user", "content": "hi"}])
        results = await run_sub_agent_team(
            [
                {"role": "researcher", "goal": "gather intel"},
                {"role": "copywriter", "goal": "write a draft"},
            ],
            state,
            ctx,
        )
        assert set(results) == {"researcher", "copywriter"}
        assert "Research agent" in results["researcher"]["result"]
        assert "Copywriter agent" in results["copywriter"]["result"]

    _run(run())
