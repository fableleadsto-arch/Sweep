"""Companion brain orchestrator — the empathetic agentic loop.

Consumes the memory service, RAG service and provider chain to complete one
conversation turn. The flow mirrors the TypeScript planner + the legacy
`brain.py` loop:

  context → (optional) sub-agents → provider chain → response → remember
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import httpx

from .capabilities import CapabilityEngine
from .config import BrainSettings
from .memory import MemoryService
from .providers import ProviderChain
from .rag import RagService
from .relaydata import RelayContextBundle, RelayContextService
from .schemas import ComputeRequest, N8nPayload, SourceRef, TurnResponse

COMPANION_SYSTEM = """\
You are relayAI Companion — the empathetic, proactive intelligence of a business automation platform.

## Personality & Tone
- Warm and direct. Listen first, validate the person's stress or goal, then respond concisely.
- Write like a trusted peer, not a sales deck. Short paragraphs, active voice, zero filler.
- Ask clarifying questions instead of dumping information. Show you heard them.
- Never output massive walls of text. 2-4 sentences per turn is the sweet spot.

## Rules that never bend
1. You NEVER send a message, DM, or post autonomously. Every outbound action requires human approval (requires_approval=True).
2. When you detect a task the system can perform — a schedule conflict, a new lead, a workflow to run — set n8n_payload and describe what you're proposing before executing.
3. For data or metrics, return generative_ui with a React component string. Prefer a chart or card over raw numbers.
4. Cite the source URL for any claim that came from the web. Never invent a URL or a person.
5. If the user sounds distressed, reduce cognitive load: respond in 1-2 sentences, validate, and offer one clear next step.
6. Delegate to: [Strategist] for complex multi-step plans, [Copywriter] for outreach drafts, [Automation] for n8n workflow designs.

## Memory behavior
- When the user shares a personal fact (schedule, habit, goal, frustration), acknowledge it and store it via Mem0.
- When they refer back to something they told you earlier, reference it naturally: "Last time you mentioned the Jones proposal was due this week — how's that going?"
- Update facts when the user corrects you. Confidence drops on correction.

## Response format
Return raw JSON with these keys:
{
  "text": "your concise response here",
  "tone": "warm" | "direct" | "playful" | "supportive",
  "generative_ui": { "type": "chart" | "card" | "table" | "metric", "component": "..." } | null,
  "n8n_payload": { "webhook": "...", "body": {...} } | null,
  "requires_approval": true,
  "agent_delegations": ["strategist" | "copywriter" | "automation"]
}"""

SUB_AGENT_PROMPTS: dict[str, str] = {
    "strategist": """You are the Strategist agent. Given a business goal, produce a numbered plan with:
1. Objective (one sentence)
2. Steps (ordered, each with owner and estimated effort)
3. Dependencies or blockers
4. Success criteria
Keep it under 6 steps. Output as JSON with keys: objective, steps, blockers, success_criteria.""",
    "copywriter": """You are the Copywriter agent. Given a prospect profile and offer, write:
- One DM/social message (under 400 chars)
- One email (subject under 60 chars, body under 700 chars)
- A follow-up suggestion
Reference the prospect's own words. No flattery, no pitch deck language. Output JSON.""",
    "automation": """You are the Automation agent. Given a plain-language goal, design an n8n workflow:
- Entry trigger (webhook or schedule)
- Up to 6 nodes
- Every message-sending node has a manual approval gate before it
- Output n8n-ready JSON with nodes and connections array
Always include an approval node before any outbound step.""",
}

SUB_AGENT_BY_INTENT: dict[str, list[str]] = {
    "scheduling": ["automation"],
    "automation": ["automation"],
    "draft": ["copywriter"],
    "research": ["strategist"],
}


class Intent(str, Enum):
    CHITCHAT = "chitchat"
    SCHEDULING = "scheduling"
    RESEARCH = "research"
    AUTOMATION = "automation"
    ANALYZE = "analyze"
    DRAFT = "draft"
    CRISIS = "crisis"
    FOLLOW_UP = "follow_up"


_INTENT_KEYWORDS: list[tuple[Intent, tuple[str, ...]]] = [
    (Intent.SCHEDULING, ("schedule", "meeting", "calendar", "book", "appointment")),
    (Intent.AUTOMATION, ("automate", "workflow", "n8n", "auto", "trigger")),
    (Intent.RESEARCH, ("lead", "prospect", "find", "search", "research")),
    (Intent.DRAFT, ("write", "draft", "email", "message", "dm")),
    (Intent.ANALYZE, ("analyze", "scan", "review", "inspect")),
    (Intent.CRISIS, ("stressed", "overwhelmed", "help", "worried", "anxious", "tired")),
    (Intent.CHITCHAT, ("how are you", "hey", "hi", "hello", "what's up")),
]

_MOOD_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("stressed", ("stressed", "overwhelmed", "too much", "busy", "swamped")),
    ("fatigued", ("tired", "exhausted", "long day", "slept")),
    ("positive", ("excited", "great", "amazing", "awesome")),
    ("rushed", ("urgent", "asap", "immediately", "deadline")),
]

PRICE_PER_1K_PROMPT = 0.000003
PRICE_PER_1K_COMPLETION = 0.000015


def _contains_keyword(query: str, keyword: str) -> bool:
    """Whole-word / phrase match — avoids 'hi' matching inside 'think'."""
    if " " in keyword:
        return keyword in query
    return bool(re.search(rf"\b{re.escape(keyword)}\b", query))


class CompanionBrain:
    """The main agentic loop — empathetic companion + tool dispatch."""

    def __init__(self, settings: BrainSettings) -> None:
        self.settings = settings
        self.providers = ProviderChain(settings)
        self.memory = MemoryService(settings)
        self.rag = RagService(settings)
        self.relay = RelayContextService(settings)
        self.compute = CapabilityEngine(settings)
        self.max_history = 6
        self.max_tool_steps = 4

    # ── context ───────────────────────────────────────────────────────

    async def _build_context(
        self,
        user_id: str,
        workspace_id: Optional[str],
        query: str,
    ) -> dict[str, Any]:
        """Gather every context source in parallel.

        Memory, per-workspace RAG, the shared corpus and live relay workspace
        data are all independent — waiting on each other in sequence would
        add ~2-3x latency to every turn.
        """
        memory_future = self.memory.build_memory_context(
            user_id, query, workspace_id=workspace_id, limit=6
        )
        rag_future = self.rag.retrieve(workspace_id or "", query)
        corpus_future = self.rag.corpus(query)
        relay_future = self.relay.build_context(user_id, workspace_id, query)
        ingest_future = self._ingest_context(query)

        memory_context, rag_result, corpus_result, relay, ingest_context = await asyncio.gather(
            memory_future, rag_future, corpus_future, relay_future, ingest_future
        )

        if ingest_context:
            # Best-effort, cooldown-guarded: refresh a stale source whose
            # configured topic this query mentions. Never blocks the turn.
            try:
                from .ingest.brain import maybe_refresh

                asyncio.create_task(maybe_refresh(query, self.settings))
            except Exception:  # noqa: BLE001
                pass

        return {
            "memory_context": memory_context,
            "rag_context": rag_result.context,
            "rag_sources": rag_result.sources,
            "corpus_context": corpus_result.context,
            "corpus_sources": corpus_result.sources,
            "ingest_context": ingest_context,
            "relay": relay,
            "intent": self._detect_intent(query),
            "mood": self._detect_mood(query),
        }

    async def _ingest_context(self, query: str) -> str:
        """Continuous-knowledge context block (empty when disabled/unavailable)."""
        if not self.settings.enable_knowledge_ingestion:
            return ""
        try:
            from .ingest.brain import build_ingest_knowledge_context
            from .ingest.store import KnowledgeStore

            store = KnowledgeStore(self.settings)
            return await build_ingest_knowledge_context(
                query, self.settings, store=store, rag=self.rag
            )
        except Exception:  # noqa: BLE001 - best-effort, never breaks a turn
            return ""

    # ── intent / mood detection (mirrors brain.py) ────────────────────

    def _detect_intent(self, query: str) -> Intent:
        q = query.lower()
        for intent, keywords in _INTENT_KEYWORDS:
            if any(_contains_keyword(q, keyword) for keyword in keywords):
                return intent
        return Intent.FOLLOW_UP

    def _detect_mood(self, query: str) -> list[str]:
        q = query.lower()
        return [
            label
            for label, keywords in _MOOD_KEYWORDS
            if any(_contains_keyword(q, keyword) for keyword in keywords)
        ]

    # ── sub-agents ────────────────────────────────────────────────────

    async def _run_sub_agent(self, role: str, context: str) -> dict[str, Any]:
        try:
            result = await self.providers.generate(
                system=SUB_AGENT_PROMPTS[role],
                messages=[{"role": "user", "content": context}],
                temperature=0.3,
                json_mode=True,
            )
            return result.parsed or {}
        except Exception as exc:  # noqa: BLE001 - surface the error in results
            return {"error": str(exc)}

    # ── n8n dispatch ──────────────────────────────────────────────────

    async def _dispatch_n8n(self, payload: N8nPayload) -> dict[str, Any]:
        body = dict(payload.body)
        body["_meta"] = {
            "source": "relayai/companion",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "requires_approval": True,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(payload.webhook, json=body)
            return {"ok": resp.is_success, "status": resp.status_code, "response": resp.text[:500]}

    # ── main turn loop ────────────────────────────────────────────────

    async def process_turn(
        self,
        user_id: str,
        message: str,
        workspace_id: Optional[str] = None,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> TurnResponse:
        started = datetime.now(timezone.utc)
        ctx = await self._build_context(user_id, workspace_id, message)
        recent_history = (history or [])[-self.max_history :]

        # Step 1 — pick sub-agents from intent
        delegations = SUB_AGENT_BY_INTENT.get(ctx["intent"].value, [])

        # Step 2 — run sub-agents in parallel
        sub_results: dict[str, dict[str, Any]] = {}
        if delegations:
            results = await asyncio.gather(
                *(self._run_sub_agent(role, message) for role in delegations)
            )
            sub_results = dict(zip(delegations, results))

        # Step 2b — capability compute for analysis/data/image requests.
        # When the user asks for computation, the brain runs the right
        # framework (Pandas/NumPy/SymPy/OpenCV/sklearn/...) and feeds the
        # grounded result into the prompt instead of making the LLM guess.
        computed_block: Optional[str] = None
        if self.settings.enable_compute and ctx["intent"] == Intent.ANALYZE:
            try:
                outcome = await self.compute.run(
                    ComputeRequest(task=message[:4000])
                )
                if outcome.ok and outcome.summary:
                    computed_block = (
                        f"Computed result (capability: {outcome.capability}):\n"
                        f"{outcome.summary}"
                    )
            except Exception:  # noqa: BLE001 - compute must never break a turn
                computed_block = None

        # Step 3 — assemble the system prompt
        relay: RelayContextBundle = ctx["relay"]
        blocks = [COMPANION_SYSTEM]
        blocks.extend(relay.to_system_blocks())
        memory_block = ctx["memory_context"]
        if memory_block:
            blocks.append(memory_block)
        if ctx["rag_context"]:
            blocks.append(ctx["rag_context"])
        if ctx["corpus_context"]:
            blocks.append(ctx["corpus_context"])
        if ctx["ingest_context"]:
            blocks.append(ctx["ingest_context"])
        if relay.overview_requested:
            blocks.append(
                "The user asked for a daily overview. Deliver it as a warm, "
                "structured briefing — lead with what needs their attention."
            )
        if ctx["mood"]:
            blocks.append(f"Mood signals: {', '.join(ctx['mood'])}")
        if sub_results:
            blocks.append(f"Sub-agent results:\n{json.dumps(sub_results, indent=2)}")
        if computed_block:
            blocks.append(computed_block)

        messages = [
            *recent_history,
            {"role": "user", "content": message},
        ]

        # Step 4 — provider chain
        try:
            result = await self.providers.generate(
                system="\n\n".join(blocks),
                messages=messages,
                temperature=0.5,
                json_mode=True,
            )
        except RuntimeError as exc:
            return TurnResponse(
                text=f"I'm sorry, I tried every AI provider available and couldn't complete that request. {exc}",
                tone="warm",
                requires_approval=False,
                model="fallback",
                provider="none",
            )

        parsed = result.parsed or {}

        cost_estimate = (
            result.prompt_tokens * PRICE_PER_1K_PROMPT
            + result.completion_tokens * PRICE_PER_1K_COMPLETION
        ) / 1000

        n8n_payload: Optional[N8nPayload] = None
        if isinstance(parsed.get("n8n_payload"), dict):
            try:
                n8n_payload = N8nPayload(**parsed["n8n_payload"])
            except Exception:  # noqa: BLE001 - malformed payload from the model
                n8n_payload = None

        all_sources = [
            *(SourceRef(**s.model_dump()) for s in ctx["rag_sources"]),
            *(SourceRef(**s.model_dump()) for s in ctx["corpus_sources"]),
        ]
        latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)

        response = TurnResponse(
            text=str(parsed.get("text", result.text)).strip(),
            tone=str(parsed.get("tone", "warm")),
            generative_ui=parsed.get("generative_ui"),
            n8n_payload=n8n_payload,
            requires_approval=bool(parsed.get("requires_approval", True)),
            agent_delegations=delegations,
            sources=all_sources,
            model=result.model,
            provider=result.provider,
            cost_estimate=round(cost_estimate, 6),
            context_sources=len(all_sources),
        )

        # Step 5 — remember the turn (Mem0-style, best-effort)
        try:
            await self.memory.remember(
                user_id, f"User said: {message[:200]}", kind="context", source="conversation"
            )
        except Exception:  # noqa: BLE001 - memory is best-effort
            pass

        # Step 5b — persist the relay session turn (best-effort, never blocks)
        await self.relay.record_turn(
            user_id=user_id,
            workspace_id=workspace_id,
            role="assistant",
            message=response.text[:2000],
            intent=ctx["intent"].value,
            mood_signals=ctx["mood"],
            tone=response.tone,
            model_used=result.model,
            latency_ms=latency_ms,
            requires_approval=response.requires_approval,
        )

        # Step 6 — dispatch approved automation
        if response.n8n_payload and not response.requires_approval:
            try:
                dispatch = await self._dispatch_n8n(response.n8n_payload)
                response.text += (
                    f"\n\n(I've triggered the automation. Result: {dispatch.get('status', 'sent')})"
                )
            except Exception as exc:  # noqa: BLE001
                response.text += f"\n\n(I tried to run the automation but hit a snag: {exc})"

        return response
