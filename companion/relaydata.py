"""Relay AI workspace context — feeds the companion brain live relay data.

Mirrors the TypeScript stack exactly so the Python brain and the RelAI
engine reason from the same facts:

  * `src/lib/overview.server.ts`   → daily overview briefing assembly
  * `src/lib/daily-overview.ts`    → pure briefing builder (honest empty states)
  * `src/RelAI/core/planner.server.ts` → overview-request detection

What this feeds into a turn:
  * User profile (real name, timezone, communication style, channel)
  * Daily overview briefing (tasks, automations, approvals, inbox, AI activity)
  * Workspace snapshot (approvals waiting, unread inbox, live workflows, leads)
  * Relational memory-graph facts (high-importance nodes)
  * Companion pending tasks (approval queue from the companion schema)
  * Session + turn persistence (`companion_sessions` / `session_turns`)

Every leg is best-effort: a missing table, RLS change, or network blip must
never break a turn — it just yields an empty block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from .config import BrainSettings

# ─────────────────────────────────────────────────────────────────────────
#  Models
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class UserProfile:
    user_id: str
    display_name: str = ""
    first_name: str = ""
    timezone: str = "UTC"
    communication_style: str = "warm"  # warm | direct | playful | supportive
    preferred_channel: str = "text"    # text | voice | email
    response_length: str = "concise"   # concise | detailed
    mood_trend: list[str] = field(default_factory=list)
    source: str = "none"

    @property
    def greeting_name(self) -> Optional[str]:
        return self.first_name or (self.display_name or None)

    def to_block(self) -> str:
        """Prompt-ready profile block (never more than a few lines)."""
        lines = ["User profile:"]
        if self.greeting_name:
            lines.append(f"- Name: {self.greeting_name}")
        lines.append(f"- Communication style: {self.communication_style}")
        lines.append(f"- Preferred channel: {self.preferred_channel}")
        lines.append(f"- Preferred response length: {self.response_length}")
        if self.mood_trend:
            lines.append(f"- Recent mood trend: {', '.join(self.mood_trend[:5])}")
        return "\n".join(lines)


@dataclass
class OverviewTask:
    title: str
    project: Optional[str] = None
    priority: Optional[str] = None


@dataclass
class OverviewData:
    type: str  # "morning" | "evening"
    first_name: Optional[str] = None
    tasks_due_today: list[OverviewTask] = field(default_factory=list)
    tasks_overdue: list[OverviewTask] = field(default_factory=list)
    tasks_due_tomorrow: list[OverviewTask] = field(default_factory=list)
    open_tasks: list[OverviewTask] = field(default_factory=list)
    completed_yesterday: list[OverviewTask] = field(default_factory=list)
    completed_today: list[OverviewTask] = field(default_factory=list)
    workflow_runs_24h: list[dict[str, Any]] = field(default_factory=list)
    dispatches_24h: list[dict[str, Any]] = field(default_factory=list)
    agent_runs_24h: list[dict[str, Any]] = field(default_factory=list)
    pending_approvals: int = 0
    unread_conversations: int = 0
    new_memories_24h: int = 0
    ai_calls_today: int = 0


@dataclass
class WorkspaceSnapshot:
    workspace_id: Optional[str] = None
    workspace_name: str = ""
    pending_approvals: int = 0
    unread_conversations: int = 0
    open_conversations: int = 0
    active_workflows: int = 0
    workflow_runs_24h: list[dict[str, Any]] = field(default_factory=list)
    agent_runs_24h: list[dict[str, Any]] = field(default_factory=list)
    leads_count: int = 0
    contacts_count: int = 0

    def to_block(self) -> str:
        """Compact workspace snapshot for the system prompt."""
        if not self.workspace_id:
            return ""
        lines = ["Live workspace snapshot:"]
        if self.workspace_name:
            lines.append(f"- Workspace: {self.workspace_name}")
        if self.pending_approvals:
            lines.append(f"- {self.pending_approvals} action(s) awaiting approval")
        if self.unread_conversations:
            lines.append(f"- {self.unread_conversations} unread conversation(s)")
        if self.active_workflows:
            lines.append(f"- {self.active_workflows} active workflow(s)")
        failed_runs = [r for r in self.workflow_runs_24h if r.get("status") in ("failed", "error")]
        if failed_runs:
            lines.append(f"- {len(failed_runs)} failed automation run(s) in the last 24h")
        if self.agent_runs_24h:
            ok = [r for r in self.agent_runs_24h if r.get("status") in ("completed", "succeeded")]
            if ok:
                lines.append(f"- {len(ok)} AI task(s) completed in the last 24h")
        if self.leads_count:
            lines.append(f"- {self.leads_count} lead(s) on file")
        if self.contacts_count:
            lines.append(f"- {self.contacts_count} contact(s) on file")
        return "\n".join(lines)


@dataclass
class RelayContextBundle:
    profile: UserProfile = field(default_factory=UserProfile)
    overview: str = ""
    overview_requested: bool = False
    workspace: WorkspaceSnapshot = field(default_factory=WorkspaceSnapshot)
    graph_facts: list[str] = field(default_factory=list)
    companion_tasks: list[dict[str, Any]] = field(default_factory=list)

    def to_system_blocks(self) -> list[str]:
        blocks: list[str] = []
        profile_block = self.profile.to_block()
        if profile_block:
            blocks.append(profile_block)
        if self.workspace:
            ws_block = self.workspace.to_block()
            if ws_block:
                blocks.append(ws_block)
        if self.overview:
            blocks.append(f"Daily overview:\n{self.overview}")
        if self.graph_facts:
            blocks.append(
                "Relational memory facts:\n"
                + "\n".join(f"- {f}" for f in self.graph_facts[:6])
            )
        if self.companion_tasks:
            lines = ["Companion tasks awaiting attention:"]
            for t in self.companion_tasks[:5]:
                title = t.get("title", "Untitled")
                priority = t.get("priority") or "normal"
                lines.append(f"- [{priority}] {title}")
            blocks.append("\n".join(lines))
        return blocks


# ─────────────────────────────────────────────────────────────────────────
#  Pure helpers (unit-testable, no I/O)
# ─────────────────────────────────────────────────────────────────────────

_OVERVIEW_TIME_RE = re.compile(r"\b(today|todays|daily|morning|evening|tonight|the day)\b")
_OVERVIEW_WORD_RE = re.compile(r"\b(overview|briefing)\b")
_OVERVIEW_PHRASE_RE = re.compile(
    r"\bbrief me\b|\bwhat('s| is) on my plate\b|\bwhat('s| is) (going on|up|happening) today\b"
)
_OVERVIEW_READ_ME_RE = re.compile(r"\bread me\b")


def is_overview_request(message: str) -> bool:
    """Mirror `isOverviewRequest` in `src/RelAI/core/planner.server.ts`."""
    m = message.strip().lower()
    if not m or len(m) > 80:
        return False
    if _OVERVIEW_PHRASE_RE.search(m):
        return True
    has_time = bool(_OVERVIEW_TIME_RE.search(m))
    has_word = bool(_OVERVIEW_WORD_RE.search(m))
    if has_time and has_word:
        return True
    return bool(_OVERVIEW_READ_ME_RE.search(m)) and has_word


def local_day_bounds(
    time_zone: str,
    now: Optional[datetime] = None,
) -> tuple[datetime, datetime]:
    """Local-day boundaries (UTC instants) for an IANA timezone.

    Falls back to the UTC day when the zone can't be resolved — the same
    graceful behavior as `localDayBounds` in `daily-overview.ts`.
    """
    now = now or datetime.now(timezone.utc)
    try:
        tz = ZoneInfo(time_zone) if time_zone else timezone.utc
    except (ZoneInfoNotFoundError, ValueError):
        tz = timezone.utc
    local = now.astimezone(tz)
    start_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
    start = start_local.astimezone(timezone.utc)
    end = (start_local + timedelta(days=1)).astimezone(timezone.utc)
    return start, end


def _count_line(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def _count_of(rows: list[dict[str, Any]], statuses: set[str]) -> int:
    return sum(1 for r in rows if r.get("status") in statuses)


def _task_line(task: OverviewTask) -> str:
    if task.project:
        return f"{task.title} ({task.project})"
    return task.title


def build_overview_text(data: OverviewData, now: Optional[datetime] = None) -> str:
    """Pure briefing builder — mirrors `buildOverviewContent` + `overviewBriefingText`.

    Never fabricates facts: empty domains produce honest statements.
    """
    now = now or datetime.now(timezone.utc)
    name = data.first_name or "there"
    hour = now.astimezone(timezone.utc).hour
    if data.type == "evening":
        greeting = f"Good evening, {name}." if data.first_name else "Good evening."
        sub = f"Here's your {now.strftime('%A')} wrap-up."
    else:
        base = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")
        greeting = f"{base}, {name}." if data.first_name else f"{base}."
        sub = f"Here's your briefing for {now.strftime('%A, %B')} {now.day}, {now.year}."

    lines: list[str] = [f"{greeting} {sub}"]

    def section(title: str, items: list[str]) -> None:
        lines.append("")
        lines.append(title.upper())
        lines.extend(f"- {item}" for item in items)

    if data.type == "morning":
        section(
            "Today's schedule",
            [_task_line(t) for t in data.tasks_due_today[:6]]
            or ["No tasks due today."],
        )
        unfinished = (data.tasks_overdue + data.open_tasks)[:6]
        section(
            "Unfinished tasks",
            [_task_line(t) for t in unfinished]
            or ["Nothing open right now — all caught up."],
        )
        section(
            "Completed yesterday",
            [_task_line(t) for t in data.completed_yesterday[:5]]
            or ["No tasks completed yesterday."],
        )

        runs_ok = _count_of(data.workflow_runs_24h, {"completed", "success"})
        dispatch_ok = _count_of(data.dispatches_24h, {"completed", "dispatched"})
        runs_failed = _count_of(data.workflow_runs_24h, {"failed", "error"})
        auto_items: list[str] = []
        if runs_ok:
            auto_items.append(_count_line(runs_ok, "workflow run completed", "workflow runs completed") + " in the last 24 hours")
        if dispatch_ok:
            auto_items.append(_count_line(dispatch_ok, "automation dispatched", "automations dispatched") + " in the last 24 hours")
        if runs_failed:
            auto_items.append(_count_line(runs_failed, "run needs attention", "runs need attention"))
        if data.pending_approvals:
            auto_items.append(_count_line(data.pending_approvals, "approval waiting for you", "approvals waiting for you"))
        section(
            "Automation status",
            auto_items or ["No automation activity in the last 24 hours."],
        )

        agent_ok = _count_of(data.agent_runs_24h, {"completed", "succeeded"})
        ai_items: list[str] = []
        if agent_ok:
            ai_items.append(_count_line(agent_ok, "AI task completed", "AI tasks completed") + " overnight")
        if data.ai_calls_today:
            ai_items.append(_count_line(data.ai_calls_today, "AI call used today", "AI calls used today"))
        if data.new_memories_24h:
            ai_items.append(_count_line(data.new_memories_24h, "new memory captured", "new memories captured"))
        section("AI activity", ai_items or ["No AI activity recorded yet."])

        attention: list[str] = []
        if data.unread_conversations:
            attention.append(_count_line(data.unread_conversations, "unread conversation", "unread conversations"))
        if data.tasks_overdue:
            attention.append(_count_line(len(data.tasks_overdue), "task is overdue", "tasks are overdue"))
        if runs_failed:
            attention.append(_count_line(runs_failed, "failed automation run", "failed automation runs"))
        if data.pending_approvals:
            attention.append(_count_line(data.pending_approvals, "action awaiting approval", "actions awaiting approval"))
        section(
            "Needs attention",
            attention or ["Inbox clear — nothing needs attention right now."],
        )
    else:
        section(
            "Tasks completed today",
            [_task_line(t) for t in data.completed_today[:6]]
            or ["No tasks completed today yet."],
        )
        open_tasks = (data.tasks_overdue + data.open_tasks)[:6]
        section(
            "Still open",
            [_task_line(t) for t in open_tasks]
            or ["Nothing left open — great place to finish the day."],
        )

        runs_today = _count_of(data.workflow_runs_24h, {"completed", "success"})
        dispatch_today = _count_of(data.dispatches_24h, {"completed", "dispatched"})
        auto_items = []
        if runs_today:
            auto_items.append(_count_line(runs_today, "workflow run executed", "workflow runs executed"))
        if dispatch_today:
            auto_items.append(_count_line(dispatch_today, "automation dispatched", "automations dispatched"))
        if _count_of(data.workflow_runs_24h, {"failed", "error"}):
            auto_items.append("Some runs failed — review the Automation page")
        section("Automations executed", auto_items or ["No automations ran today."])

        section(
            "New memories captured",
            [
                _count_line(data.new_memories_24h, "new memory captured", "new memories captured")
                + " today — Relay is getting to know your work better."
            ]
            if data.new_memories_24h
            else ["No new memories captured today."],
        )

        agent_ok = _count_of(data.agent_runs_24h, {"completed", "succeeded"})
        hi_items: list[str] = []
        if agent_ok:
            hi_items.append(_count_line(agent_ok, "AI task completed", "AI tasks completed") + " today")
        if data.ai_calls_today:
            hi_items.append(_count_line(data.ai_calls_today, "AI call used today", "AI calls used today"))
        if data.completed_today:
            hi_items.append(f"{len(data.completed_today)} task{'s' if len(data.completed_today) != 1 else ''} checked off")
        section("Today at a glance", hi_items or ["Quiet day — no AI or task activity to report yet."])

        tomorrow = (data.tasks_due_tomorrow + data.tasks_overdue)[:6]
        section(
            "Up next",
            [_task_line(t) for t in tomorrow]
            or ["Nothing due tomorrow — a clean slate."],
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
#  Supabase REST helper
# ─────────────────────────────────────────────────────────────────────────


class _Rest:
    """Tiny Supabase REST wrapper shared by every relay context leg."""

    def __init__(self, settings: BrainSettings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return bool(self.settings.supabase_url and self.settings.supabase_key)

    @property
    def headers(self) -> dict[str, str]:
        return {
            "apikey": self.settings.supabase_key,
            "Authorization": f"Bearer {self.settings.supabase_key}",
            "Content-Type": "application/json",
        }

    async def select(
        self,
        table: str,
        *,
        columns: str = "*",
        filters: Optional[dict[str, Any] | list[tuple[str, str]]] = None,
        limit: int = 20,
        order: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> list[dict[str, Any]]:
        url = f"{self.settings.supabase_url}/rest/v1/{table}"
        params: list[tuple[str, str]] = [("select", columns), ("limit", str(limit))]
        if order:
            params.append(("order", order))
        if isinstance(filters, dict):
            params.extend((k, str(v)) for k, v in filters.items())
        elif filters:
            params.extend((k, str(v)) for k, v in filters)
        own = client is None
        client = client or httpx.AsyncClient(timeout=20.0)
        try:
            resp = await client.get(url, headers=self.headers, params=params)
            if not resp.is_success:
                return []
            data = resp.json()
            return data if isinstance(data, list) else []
        except httpx.HTTPError:
            return []
        finally:
            if own:
                await client.aclose()

    async def insert(
        self,
        table: str,
        rows: list[dict[str, Any]],
        *,
        representation: bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> list[dict[str, Any]]:
        url = f"{self.settings.supabase_url}/rest/v1/{table}"
        headers = dict(self.headers)
        if representation:
            headers["Prefer"] = "return=representation"
        own = client is None
        client = client or httpx.AsyncClient(timeout=20.0)
        try:
            resp = await client.post(url, headers=headers, json=rows)
            if not resp.is_success:
                return []
            if representation:
                data = resp.json()
                return data if isinstance(data, list) else []
            return [{}]
        except httpx.HTTPError:
            return []
        finally:
            if own:
                await client.aclose()


# ─────────────────────────────────────────────────────────────────────────
#  Service
# ─────────────────────────────────────────────────────────────────────────


class RelayContextService:
    """Feeds the companion brain with live relay AI workspace data."""

    def __init__(self, settings: BrainSettings) -> None:
        self.settings = settings
        self.rest = _Rest(settings)

    # ── profile ────────────────────────────────────────────────────────

    async def user_profile(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> UserProfile:
        """Real name + communication preferences, mirroring `user_profiles`
        (companion schema) with a `profiles`/`workspaces` fallback."""
        profile = UserProfile(user_id=user_id, source="none")
        if not self.rest.enabled:
            return profile

        # 1) Rich companion profile.
        rows = await self.rest.select(
            "user_profiles",
            columns="display_name,timezone,communication_style,response_length,preferred_channel,mood_trend",
            filters={"user_id": f"eq.{user_id}"},
            limit=1,
        )
        if rows:
            row = rows[0]
            profile.display_name = str(row.get("display_name") or "")
            profile.timezone = str(row.get("timezone") or "UTC")
            profile.communication_style = str(row.get("communication_style") or "warm")
            profile.preferred_channel = str(row.get("preferred_channel") or "text")
            profile.response_length = str(row.get("response_length") or "concise")
            trend = row.get("mood_trend")
            if isinstance(trend, list):
                profile.mood_trend = [str(t) for t in trend if t][:8]
            profile.source = "user_profiles"

        # 2) Display-name fallback from the relay `profiles` table.
        if not profile.display_name:
            rows = await self.rest.select(
                "profiles",
                columns="display_name,email",
                filters={"id": f"eq.{user_id}"},
                limit=1,
            )
            if rows:
                profile.display_name = str(rows[0].get("display_name") or "")

        # 3) Timezone from the workspace when the profile didn't set one.
        if workspace_id and (profile.timezone == "UTC" or not profile.timezone):
            rows = await self.rest.select(
                "workspaces",
                columns="timezone,name",
                filters={"id": f"eq.{workspace_id}"},
                limit=1,
            )
            if rows and rows[0].get("timezone"):
                profile.timezone = str(rows[0]["timezone"])
        profile.first_name = (profile.display_name or "").split()[0] if profile.display_name else ""
        return profile

    # ── overview ───────────────────────────────────────────────────────

    async def overview(
        self,
        user_id: str,
        workspace_id: Optional[str],
        briefing_type: str = "morning",
    ) -> str:
        """Assemble the daily overview briefing (mirrors `assembleOverviewContent`)."""
        if not self.rest.enabled or not workspace_id:
            return ""

        profile = await self.user_profile(user_id, workspace_id)
        now = datetime.now(timezone.utc)
        today_start, today_end = local_day_bounds(profile.timezone, now)
        yesterday_start = today_start - timedelta(days=1)
        tomorrow_start, tomorrow_end = today_end, today_end + timedelta(days=1)
        day_ago = now - timedelta(days=1)

        open_statuses = ("open", "scheduled", "in_progress", "overdue")

        def to_task(row: dict[str, Any]) -> OverviewTask:
            return OverviewTask(
                title=str(row.get("title") or "Task"),
                project=row.get("project", {}).get("name") if isinstance(row.get("project"), dict) else None,
                priority=row.get("priority"),
            )

        async def tasks_due(gte: datetime, lt: datetime) -> list[OverviewTask]:
            rows = await self.rest.select(
                "workflow_tasks",
                columns="title,priority,project:projects(name)",
                filters={
                    "workspace_id": f"eq.{workspace_id}",
                    "status": f"in.({','.join(open_statuses)})",
                    "due_date": f"gte.{gte.isoformat()}",
                },
                order="due_date.asc.nullslast",
                limit=8,
            )
            return [to_task(r) for r in rows]

        async def tasks_overdue() -> list[OverviewTask]:
            rows = await self.rest.select(
                "workflow_tasks",
                columns="title,priority,project:projects(name)",
                filters={
                    "workspace_id": f"eq.{workspace_id}",
                    "status": f"in.({','.join(open_statuses)})",
                    "due_date": f"lt.{today_start.isoformat()}",
                },
                order="due_date.asc.nullslast",
                limit=8,
            )
            return [to_task(r) for r in rows]

        async def tasks_done(start: datetime, end: datetime, limit: int) -> list[OverviewTask]:
            rows = await self.rest.select(
                "workflow_tasks",
                columns="title,priority,project:projects(name)",
                filters=[
                    ("workspace_id", f"eq.{workspace_id}"),
                    ("status", "eq.done"),
                    ("updated_at", f"gte.{start.isoformat()}"),
                    ("updated_at", f"lt.{end.isoformat()}"),
                ],
                order="updated_at.desc",
                limit=limit,
            )
            return [to_task(r) for r in rows]

        async def status_rows(table: str, ts_col: str) -> list[dict[str, Any]]:
            rows = await self.rest.select(
                table,
                columns="status",
                filters={
                    "workspace_id": f"eq.{workspace_id}",
                    ts_col: f"gte.{day_ago.isoformat()}",
                },
                limit=50,
            )
            return [{"status": str(r.get("status") or "")} for r in rows]

        async def count_query(table: str, ts_col: Optional[str] = None) -> int:
            filters: dict[str, Any] = {"workspace_id": f"eq.{workspace_id}"}
            if ts_col:
                filters[ts_col] = f"gte.{day_ago.isoformat()}"
            rows = await self.rest.select(table, columns="id", filters=filters, limit=1000)
            return len(rows)

        async def count_pending_approvals() -> int:
            rows = await self.rest.select(
                "pending_approvals",
                columns="id",
                filters={"workspace_id": f"eq.{workspace_id}", "status": "eq.pending"},
                limit=1000,
            )
            return len(rows)

        async def sum_unread() -> int:
            rows = await self.rest.select(
                "conversations",
                columns="unread_count",
                filters={"workspace_id": f"eq.{workspace_id}", "unread_count": "gt.0"},
                limit=200,
            )
            return sum(int(r.get("unread_count") or 0) for r in rows)

        async def agent_runs() -> list[dict[str, Any]]:
            rows = await self.rest.select(
                "agent_runs",
                columns="status,graph_name",
                filters={"workspace_id": f"eq.{workspace_id}", "created_at": f"gte.{day_ago.isoformat()}"},
                limit=50,
            )
            return [{"status": str(r.get("status") or ""), "graph": r.get("graph_name")} for r in rows]

        (
            due_today,
            overdue,
            due_tomorrow,
            completed_yesterday,
            completed_today,
            workflow_runs,
            dispatches,
            agent_runs_rows,
            approvals,
            unread,
            new_memories,
            ai_calls,
        ) = await _gather(
            tasks_due(today_start, today_end),
            tasks_overdue(),
            tasks_due(tomorrow_start, tomorrow_end),
            tasks_done(yesterday_start, today_start, 5),
            tasks_done(today_start, today_end, 6),
            status_rows("workflow_runs", "created_at"),
            status_rows("automation_dispatches", "dispatched_at"),
            agent_runs(),
            count_pending_approvals(),
            sum_unread(),
            count_query("mira_memory", "created_at"),
            count_query("ai_usage_logs", "created_at"),
        )

        open_rows = await self.rest.select(
            "workflow_tasks",
            columns="title,priority,project:projects(name)",
            filters={"workspace_id": f"eq.{workspace_id}", "status": f"in.({','.join(open_statuses)})"},
            limit=8,
        )
        open_tasks = [to_task(r) for r in open_rows]

        data = OverviewData(
            type=briefing_type,
            first_name=profile.greeting_name,
            tasks_due_today=due_today,
            tasks_overdue=overdue,
            tasks_due_tomorrow=due_tomorrow,
            open_tasks=open_tasks,
            completed_yesterday=completed_yesterday,
            completed_today=completed_today,
            workflow_runs_24h=workflow_runs,
            dispatches_24h=dispatches,
            agent_runs_24h=agent_runs_rows,
            pending_approvals=approvals,
            unread_conversations=unread,
            new_memories_24h=new_memories,
            ai_calls_today=ai_calls,
        )
        return build_overview_text(data, now)

    # ── workspace snapshot ─────────────────────────────────────────────

    async def workspace_snapshot(
        self,
        workspace_id: Optional[str],
    ) -> WorkspaceSnapshot:
        snapshot = WorkspaceSnapshot(workspace_id=workspace_id)
        if not self.rest.enabled or not workspace_id:
            return snapshot

        async def count(table: str, filters: Optional[dict[str, Any]] = None) -> int:
            base: dict[str, Any] = {"workspace_id": f"eq.{workspace_id}"}
            base.update(filters or {})
            rows = await self.rest.select(table, columns="id", filters=base, limit=1000)
            return len(rows)

        async def runs(table: str, ts_col: str) -> list[dict[str, Any]]:
            rows = await self.rest.select(
                table,
                columns="status",
                filters={"workspace_id": f"eq.{workspace_id}", ts_col: f"gte.{(datetime.now(timezone.utc) - timedelta(days=1)).isoformat()}"},
                limit=50,
            )
            return [{"status": str(r.get("status") or "")} for r in rows]

        unread_rows = await self.rest.select(
            "conversations",
            columns="unread_count",
            filters={"workspace_id": f"eq.{workspace_id}", "unread_count": "gt.0"},
            limit=200,
        )
        open_rows = await self.rest.select(
            "conversations",
            columns="id",
            filters={"workspace_id": f"eq.{workspace_id}", "status": "eq.open"},
            limit=1000,
        )

        (
            approvals,
            active_workflows,
            workflow_runs,
            agent_runs_rows,
            leads,
            contacts,
            ws_rows,
        ) = await _gather(
            count("pending_approvals", {"status": "eq.pending"}),
            count("workflows", {"enabled": "eq.true"}),
            runs("workflow_runs", "created_at"),
            runs("agent_runs", "created_at"),
            count("leads"),
            count("contacts"),
            self.rest.select("workspaces", columns="name", filters={"id": f"eq.{workspace_id}"}, limit=1),
        )

        snapshot.pending_approvals = approvals
        snapshot.active_workflows = active_workflows
        snapshot.workflow_runs_24h = workflow_runs
        snapshot.agent_runs_24h = agent_runs_rows
        snapshot.leads_count = leads
        snapshot.contacts_count = contacts
        snapshot.unread_conversations = sum(int(r.get("unread_count") or 0) for r in unread_rows)
        snapshot.open_conversations = len(open_rows)
        if ws_rows and ws_rows[0].get("name"):
            snapshot.workspace_name = str(ws_rows[0]["name"])
        return snapshot

    # ── relational memory graph + companion tasks ─────────────────────

    async def graph_facts(self, user_id: str, limit: int = 6) -> list[str]:
        if not self.rest.enabled:
            return []
        rows = await self.rest.select(
            "memory_graph_nodes",
            columns="content,importance",
            filters={"user_id": f"eq.{user_id}"},
            order="importance.desc",
            limit=limit,
        )
        return [str(r.get("content") or "") for r in rows if r.get("content")]

    async def pending_companion_tasks(self, user_id: str, limit: int = 5) -> list[dict[str, Any]]:
        if not self.rest.enabled:
            return []
        return await self.rest.select(
            "companion_tasks",
            columns="title,priority,task_type",
            filters={"user_id": f"eq.{user_id}", "status": "eq.pending"},
            order="priority.asc",
            limit=limit,
        )

    # ── full bundle ────────────────────────────────────────────────────

    async def build_context(
        self,
        user_id: str,
        workspace_id: Optional[str],
        query: str,
    ) -> RelayContextBundle:
        """Assemble every relay context source in parallel."""
        if not self.settings.enable_relay_context:
            return RelayContextBundle(
                profile=UserProfile(user_id=user_id, source="disabled"),
                overview_requested=is_overview_request(query),
            )
        want_overview = is_overview_request(query)
        profile_future = self.user_profile(user_id, workspace_id)
        workspace_future = self.workspace_snapshot(workspace_id)
        graph_future = self.graph_facts(user_id)
        tasks_future = self.pending_companion_tasks(user_id)

        if want_overview:
            overview_future = self.overview(user_id, workspace_id, "morning")
        else:
            overview_future = _empty_str()

        profile, workspace, graph_facts, tasks, overview = await _gather(
            profile_future,
            workspace_future,
            graph_future,
            tasks_future,
            overview_future,
        )
        return RelayContextBundle(
            profile=profile,
            overview=overview,
            overview_requested=want_overview,
            workspace=workspace,
            graph_facts=graph_facts,
            companion_tasks=tasks,
        )

    # ── session / turn persistence ─────────────────────────────────────

    async def record_turn(
        self,
        *,
        user_id: str,
        workspace_id: Optional[str],
        role: str,
        message: str,
        intent: Optional[str] = None,
        mood_signals: Optional[list[str]] = None,
        tone: Optional[str] = None,
        model_used: Optional[str] = None,
        latency_ms: Optional[int] = None,
        requires_approval: bool = True,
        device: str = "api",
    ) -> None:
        """Write a turn into `companion_sessions` + `session_turns`.

        Best-effort and never blocks the response loop. Reuses an active
        session for the user, otherwise opens one.
        """
        if not self.rest.enabled:
            return
        try:
            session_id = await self._active_session(user_id, workspace_id, device)
            if not session_id:
                return
            await self.rest.insert(
                "session_turns",
                [
                    {
                        "session_id": session_id,
                        "user_id": user_id,
                        "role": role,
                        "message": message[:2000],
                        "detected_intent": intent,
                        "mood_signals": mood_signals or [],
                        "tone_used": tone,
                        "model_used": model_used,
                        "latency_ms": latency_ms,
                        "requires_approval": requires_approval,
                    }
                ],
            )
        except Exception:  # noqa: BLE001 - persistence must never break a turn
            return

    async def _active_session(
        self,
        user_id: str,
        workspace_id: Optional[str],
        device: str,
    ) -> Optional[str]:
        rows = await self.rest.select(
            "companion_sessions",
            columns="id",
            filters={"user_id": f"eq.{user_id}", "status": "eq.active"},
            order="last_active_at.desc",
            limit=1,
        )
        if rows and rows[0].get("id"):
            return str(rows[0]["id"])
        created = await self.rest.insert(
            "companion_sessions",
            [
                {
                    "user_id": user_id,
                    "workspace_id": workspace_id,
                    "status": "active",
                    "device": device,
                    "context_snapshot": {},
                }
            ],
            representation=True,
        )
        if created and created[0].get("id"):
            return str(created[0]["id"])
        return None


# ─────────────────────────────────────────────────────────────────────────
#  async helpers
# ─────────────────────────────────────────────────────────────────────────

from collections.abc import Awaitable
from typing import TypeVar

_T = TypeVar("_T")


async def _gather(*awaitables: Awaitable[_T]) -> tuple[_T, ...]:
    """Await several coroutines together, tolerating per-leg failures.

    Each leg already catches its own errors and returns safe defaults, so a
    raise here is unexpected; keep the guard anyway for robustness.
    """
    import asyncio

    async def _safe(a: Awaitable[_T]) -> _T:
        try:
            return await a
        except Exception:  # noqa: BLE001
            return None  # type: ignore[return-value]

    return tuple(await asyncio.gather(*(_safe(a) for a in awaitables)))


async def _empty_str() -> str:
    return ""
