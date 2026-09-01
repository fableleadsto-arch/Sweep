"""Relay AI request planning — port of ``src/RelAI/core/planner.server.ts``
and ``src/RelAI/core/intent.server.ts``.

The brain's decision layer. Runs before any LLM call or tool invocation and
decides, from the raw message alone: intent (heuristic, free), model routing,
web grounding, memory/knowledge retrieval and generation knobs. Pure logic —
no network, no model call — so every request takes the shortest, cheapest path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from .routing import (
    RELAY_DEFAULT_MODEL,
    ProviderType,
    RouterContext,
    RouterPriority,
    RouterTaskType,
    classify_task,
    select_model,
)

# ─────────────────────────────────────────────────────────────
#  Intent types
# ─────────────────────────────────────────────────────────────

PrimaryIntent = Literal[
    "ask_question",
    "request_research",
    "request_analysis",
    "request_creation",
    "request_modification",
    "request_automation",
    "request_coding",
    "request_writing",
    "request_translation",
    "request_summary",
    "request_recommendation",
    "request_comparison",
    "browse_web",
    "manage_data",
    "manage_workspace",
    "chat_converse",
    "system_command",
    "unknown",
]

ToolCategory = Literal[
    "search",
    "research",
    "browser",
    "coding",
    "document",
    "knowledge",
    "memory",
    "image",
    "audio",
    "video",
    "storage",
    "database",
    "auth",
    "automation",
    "maps",
    "communication",
    "calendar",
    "payments",
    "monitoring",
    "workspace",
    "reasoning",
    "utility",
]

ToolAction = str


@dataclass(frozen=True)
class IntentAnalysis:
    """What the user wants, whether tools/memory/workspace are needed."""

    primary_intent: PrimaryIntent
    task_type: RouterTaskType
    needs_tools: bool
    suggested_category: Optional[ToolCategory] = None
    suggested_actions: tuple[ToolAction, ...] = ()
    needs_memory: bool = True
    needs_workspace_context: bool = False
    internal_knowledge_sufficient: bool = True
    complexity: int = 3
    suggested_priority: RouterPriority = "balanced"
    is_follow_up: bool = False
    entities: tuple[str, ...] = ()
    confidence: float = 0.7


# ─────────────────────────────────────────────────────────────
#  Heuristic classifier
# ─────────────────────────────────────────────────────────────

_GREETING = re.compile(r"^(hi|hello|hey|thanks|thank you|ok|okay|bye|goodbye)$", re.IGNORECASE)
_REFERENCES = re.compile(r"\b(it|that|this|they|them|the|those|these|there|their)\b", re.IGNORECASE)
_NEEDS_TOOL = re.compile(
    r"(search|find|look\s+up|google|research|investigate|crawl|scan|fetch|read|check|get\s+the|what\s+is\s+the\s+(latest|current|recent))",
    re.IGNORECASE,
)
_NEEDS_CODE = re.compile(
    r"(code|function|bug|fix|refactor|debug|test|implement|build|create\s+(a|an)\s+(function|component|api|endpoint|route|app|service))",
    re.IGNORECASE,
)
_NEEDS_WRITE = re.compile(
    r"(draft|write|rewrite|compose|edit\s+(a|the|this)|create\s+(a|an)\s+(post|email|message|article|doc))",
    re.IGNORECASE,
)
_NEEDS_DATA = re.compile(
    r"(my\s+(contacts|leads|inbox|messages|workflows|projects|tasks|campaigns)|show\s+me|list\s+my|what\s+(do\s+i|have))",
    re.IGNORECASE,
)
_NEEDS_AUTOMATION = re.compile(r"(automate|workflow|schedule|trigger|when\s+.*\s+happen|if\s+.*\s+then)", re.IGNORECASE)
_NEEDS_RESEARCH = re.compile(
    r"(research|report|citation|source|academic|paper|study|compare\s+and\s+contrast|analysis\s+of)",
    re.IGNORECASE,
)
_NEEDS_SUMMARY = re.compile(r"(summarize|summarise|tl;dr|key\s+points|brief|overview|recap)", re.IGNORECASE)
_NEEDS_COMPARISON = re.compile(r"(compare|vs\.|versus|difference\s+between|which\s+is\s+better)", re.IGNORECASE)
_NEEDS_TRANSLATION = re.compile(
    r"(translate|translation|in\s+(french|spanish|german|chinese|japanese|korean|italian|portuguese|russian|arabic))",
    re.IGNORECASE,
)
_NEEDS_BROWSE = re.compile(
    r"(open\s+(site|page|url|website)|go\s+to|navigate|browse|click\s+(on|the)|fill\s+(in|out|the\s+form)|screenshot)",
    re.IGNORECASE,
)
_ANALYSIS_SIGNAL = re.compile(r"(analyze|analysis|insight|trend|pattern)", re.IGNORECASE)
_QUESTION_PREFIX = re.compile(r"^(what|how|why|when|where|who|can\s+you|could\s+you|would\s+you|do\s+you)", re.IGNORECASE)
_CREATION_SIGNAL = re.compile(r"(create|make|build|generate|new)", re.IGNORECASE)

_DOMAIN = re.compile(r"\b([a-z0-9]([a-z0-9-]*[a-z0-9])?\.[a-z]{2,})\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_NUMBER = re.compile(r"\b\d{4,}\b")


def heuristic_analysis(
    message: str,
    history: Optional[list[dict[str, str]]] = None,
) -> dict:
    """Fast, deterministic intent heuristics — no LLM call."""
    lower = message.lower().strip()

    if not lower or _GREETING.search(lower):
        return {
            "primary_intent": "chat_converse",
            "needs_tools": False,
            "internal_knowledge_sufficient": True,
            "complexity": 1,
            "suggested_priority": "latency",
            "is_follow_up": False,
        }

    is_short = len(message.split()) < 6
    has_references = bool(_REFERENCES.search(lower))
    has_history = (history is not None and len(history) > 1) if history else False
    is_follow_up = is_short and has_references and has_history

    needs_tool = bool(_NEEDS_TOOL.search(lower))
    needs_code = bool(_NEEDS_CODE.search(lower))
    needs_write = bool(_NEEDS_WRITE.search(lower))
    needs_data = bool(_NEEDS_DATA.search(lower))
    needs_automation = bool(_NEEDS_AUTOMATION.search(lower))
    needs_research = bool(_NEEDS_RESEARCH.search(lower))
    needs_summary = bool(_NEEDS_SUMMARY.search(lower))
    needs_comparison = bool(_NEEDS_COMPARISON.search(lower))
    needs_translation = bool(_NEEDS_TRANSLATION.search(lower))
    needs_browse = bool(_NEEDS_BROWSE.search(lower))

    primary_intent: PrimaryIntent = "chat_converse"
    if needs_browse:
        primary_intent = "browse_web"
    elif needs_research:
        primary_intent = "request_research"
    elif needs_code:
        primary_intent = "request_coding"
    elif needs_write:
        primary_intent = "request_writing"
    elif needs_automation:
        primary_intent = "request_automation"
    elif needs_summary:
        primary_intent = "request_summary"
    elif needs_comparison:
        primary_intent = "request_comparison"
    elif needs_translation:
        primary_intent = "request_translation"
    elif needs_data:
        primary_intent = "manage_data"
    elif needs_tool and _ANALYSIS_SIGNAL.search(lower):
        primary_intent = "request_analysis"
    elif needs_tool:
        primary_intent = "request_research"
    elif _QUESTION_PREFIX.search(lower):
        primary_intent = "ask_question"
    elif _CREATION_SIGNAL.search(lower):
        primary_intent = "request_creation"

    internal_knowledge_sufficient = (
        not needs_tool
        and not needs_code
        and not needs_research
        and not needs_data
        and not needs_browse
        and not needs_comparison
        and not needs_translation
    )

    entities: list[str] = []
    entities.extend(m[0] for m in _DOMAIN.findall(lower))
    entities.extend(_EMAIL.findall(lower))
    entities.extend(n for n in _NUMBER.findall(lower) if len(n) < 10)

    suggested_actions: list[ToolAction] = []
    if primary_intent in ("request_research", "ask_question"):
        suggested_actions.append("search:web")
        suggested_actions.append("read:url")
    if primary_intent == "request_analysis":
        suggested_actions.append("search:web")
        suggested_actions.append("analyze:intent")
    if primary_intent == "request_coding":
        suggested_actions.append("analyze:code")
        suggested_actions.append("search:codebase")
    if primary_intent == "request_automation":
        suggested_actions.append("plan:automation")
        suggested_actions.append("trigger:automation")
    if primary_intent == "manage_data":
        suggested_actions.append("query:workspace")
    if primary_intent == "request_summary":
        suggested_actions.append("summarize:text")
    if primary_intent == "browse_web":
        suggested_actions.append("browser:navigate")

    return {
        "primary_intent": primary_intent,
        "suggested_actions": suggested_actions,
        "needs_tools": needs_tool or needs_code or needs_research or needs_data or needs_browse,
        "needs_memory": True,
        "needs_workspace_context": needs_data,
        "internal_knowledge_sufficient": internal_knowledge_sufficient,
        "complexity": 7 if (needs_research or needs_code) else 5 if needs_tool else 4 if needs_write else 2,
        "suggested_priority": "quality" if (needs_code or needs_research) else "latency" if is_short else "balanced",
        "is_follow_up": is_follow_up,
        "entities": list(dict.fromkeys(entities)),
        "confidence": 0.6 if primary_intent == "chat_converse" else 0.85,
    }


_INTENT_TO_CATEGORY: dict[PrimaryIntent, Optional[ToolCategory]] = {
    "ask_question": "search",
    "request_research": "search",
    "request_analysis": "search",
    "request_creation": "utility",
    "request_modification": "utility",
    "request_automation": "automation",
    "request_coding": "coding",
    "request_writing": "utility",
    "request_translation": "utility",
    "request_summary": "utility",
    "request_recommendation": "search",
    "request_comparison": "search",
    "browse_web": "browser",
    "manage_data": "database",
    "manage_workspace": "workspace",
    "chat_converse": None,
    "system_command": "utility",
    "unknown": None,
}


def analyze_intent(
    message: str,
    history: Optional[list[dict[str, str]]] = None,
) -> IntentAnalysis:
    """Classify user intent with fast heuristics. Zero I/O, zero LLM."""
    heuristic = heuristic_analysis(message, history)

    task_type = classify_task(
        message,
        tools_requested=heuristic.get("suggested_actions") or [],
    )

    primary = heuristic.get("primary_intent", "unknown")
    return IntentAnalysis(
        primary_intent=primary if isinstance(primary, str) else "unknown",
        task_type=task_type,
        needs_tools=bool(heuristic.get("needs_tools", False)),
        suggested_category=_INTENT_TO_CATEGORY.get(primary),
        suggested_actions=tuple(heuristic.get("suggested_actions") or []),
        needs_memory=bool(heuristic.get("needs_memory", True)),
        needs_workspace_context=bool(heuristic.get("needs_workspace_context", False)),
        internal_knowledge_sufficient=bool(heuristic.get("internal_knowledge_sufficient", True)),
        complexity=int(heuristic.get("complexity", 3)),
        suggested_priority=heuristic.get("suggested_priority", "balanced"),
        is_follow_up=bool(heuristic.get("is_follow_up", False)),
        entities=tuple(heuristic.get("entities") or []),
        confidence=float(heuristic.get("confidence", 0.7)),
    )


# ─────────────────────────────────────────────────────────────
#  Chat plan
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChatPlan:
    """The full routing decision for one request (port of TS ``ChatPlan``)."""

    intent: IntentAnalysis
    task_type: RouterTaskType
    priority: RouterPriority
    model: str
    grounded: bool
    needs_memory: bool
    needs_knowledge: bool
    needs_overview: bool
    temperature: float
    max_tokens: int
    rationale: str


TASK_KNOBS: dict[RouterTaskType, tuple[float, int]] = {
    "coding": (0.2, 2400),
    "reasoning": (0.3, 1800),
    "writing": (0.7, 1200),
    "translation": (0.2, 1000),
    "ocr": (0.1, 800),
    "image_generation": (0.8, 600),
    "voice": (0.5, 800),
    "vision": (0.1, 1200),
    "research": (0.2, 2000),
    "analysis": (0.2, 1600),
    "chat": (0.5, 1200),
    "extraction": (0.1, 900),
    "classification": (0.1, 700),
    "summarization": (0.2, 1000),
    "planning": (0.3, 1600),
    "general": (0.5, 1200),
}

GROUNDED_INTENTS = frozenset({"request_research", "request_analysis", "request_comparison"})
MEMORY_INTENTS = frozenset(
    {
        "chat_converse",
        "ask_question",
        "request_research",
        "request_analysis",
        "request_comparison",
        "request_recommendation",
        "request_writing",
        "request_summary",
    }
)
KNOWLEDGE_INTENTS = frozenset(
    {"request_research", "request_analysis", "request_comparison", "request_recommendation"}
)

_SOCIAL = re.compile(
    r"^(hi|hello|hey|thanks|thank you|ok|okay|bye|goodbye|yo|sup|good\s*(morning|afternoon|evening))[.!]*$",
    re.IGNORECASE,
)

_OVERVIEW_DIRECT = re.compile(
    r"\bbrief me\b|\bwhat('s| is) on my plate\b|\bwhat('s| is) (going on|up|happening) today\b",
    re.IGNORECASE,
)
_OVERVIEW_TIME = re.compile(r"\b(today|todays|daily|morning|evening|tonight|the day)\b", re.IGNORECASE)
_OVERVIEW_WORD = re.compile(r"\b(overview|briefing)\b", re.IGNORECASE)
_READ_ME = re.compile(r"\bread me\b", re.IGNORECASE)


def is_social(message: str) -> bool:
    """True for pure greetings/thanks/goodbyes — no cost paid for those."""
    return bool(_SOCIAL.search(message.strip()))


def is_overview_request(message: str) -> bool:
    """True for a daily-briefing request; deliberately conservative.

    "overview" or "briefing" must co-occur with a time anchor (today/daily/
    morning/...) or an explicit "read me", so a generic "give me an overview
    of X" never triggers the workspace briefing.
    """
    m = message.strip().lower()
    if len(m) > 80:
        return False
    if _OVERVIEW_DIRECT.search(m):
        return True
    has_time = bool(_OVERVIEW_TIME.search(m))
    has_overview = bool(_OVERVIEW_WORD.search(m))
    if has_time and has_overview:
        return True
    return bool(_READ_ME.search(m)) and has_overview


def estimate_input_tokens(
    message: str,
    history: Optional[list[dict[str, str]]] = None,
) -> int:
    """Rough token estimate from message + history — cheap, no tokenizer."""
    chars = len(message)
    for turn in history or []:
        chars += len(turn.get("text", "")) + 8
    return min(int(chars // 4) + 1, 50_000)


def plan_chat_request(
    message: str,
    history: Optional[list[dict[str, str]]] = None,
    force_grounded: Optional[bool] = None,
    force_model: Optional[str] = None,
) -> ChatPlan:
    """Plan a request. Zero LLM cost, zero I/O."""
    message = message.strip()
    social = is_social(message)

    intent = analyze_intent(message, history)

    grounded: bool
    if force_grounded is not None:
        grounded = force_grounded
    else:
        grounded = False if social else intent.primary_intent in GROUNDED_INTENTS

    knob_task: RouterTaskType = (
        "general"
        if intent.primary_intent in ("ask_question", "chat_converse")
        else intent.task_type
    )
    knobs = TASK_KNOBS.get(knob_task, TASK_KNOBS["general"])

    routed = select_model(
        RouterContext(
            task_type=intent.task_type,
            priority=intent.suggested_priority,
            needs_tools=intent.needs_tools,
            needs_grounding=grounded,
            estimated_input_tokens=estimate_input_tokens(message, history),
        )
    )

    model = force_model or (routed.model if routed.provider == "gemini" else RELAY_DEFAULT_MODEL)

    needs_memory = (not social) and intent.primary_intent in MEMORY_INTENTS
    needs_knowledge = (not social) and intent.primary_intent in KNOWLEDGE_INTENTS
    needs_overview = (not social) and is_overview_request(message)

    rationale = " | ".join(
        [
            f"intent={intent.primary_intent}",
            f"task={intent.task_type}",
            f"model={model}",
            f"grounded={grounded}",
            f"memory={needs_memory}",
            f"overview={needs_overview}",
        ]
    )

    return ChatPlan(
        intent=intent,
        task_type=intent.task_type,
        priority=intent.suggested_priority,
        model=model,
        grounded=grounded,
        needs_memory=needs_memory,
        needs_knowledge=needs_knowledge,
        needs_overview=needs_overview,
        temperature=knobs[0],
        max_tokens=knobs[1],
        rationale=rationale,
    )
