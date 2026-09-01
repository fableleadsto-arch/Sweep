"""Relay context service tests — pure helpers, filters, and degradation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

from companion.config import BrainSettings
from companion.relaydata import (
    RelayContextBundle,
    RelayContextService,
    UserProfile,
    _Rest,
    build_overview_text,
    is_overview_request,
    local_day_bounds,
)


@pytest.fixture()
def sb_settings(settings: BrainSettings) -> BrainSettings:
    """Settings with Supabase configured for REST-leg tests."""
    return settings.model_copy(
        update={
            "supabase_url": "https://example.supabase.co",
            "supabase_service_role_key": "test-key",
        }
    )


# ── overview-request detection (mirrors the TS planner) ──────────────────


@pytest.mark.parametrize(
    "message",
    [
        "what's on my plate today",
        "give me a daily briefing",
        "morning overview please",
        "read me the day's overview",
        "brief me for the day",
    ],
)
def test_is_overview_request_true(message: str) -> None:
    assert is_overview_request(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "",
        "hello",
        "can you schedule a meeting",
        "x" * 81,
    ],
)
def test_is_overview_request_false(message: str) -> None:
    assert is_overview_request(message) is False


# ── local day bounds ─────────────────────────────────────────────────────


def test_local_day_bounds_utc() -> None:
    now = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
    start, end = local_day_bounds("UTC", now)
    assert start == datetime(2026, 8, 4, 0, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 8, 5, 0, 0, tzinfo=timezone.utc)


def test_local_day_bounds_unknown_zone_falls_back_to_utc() -> None:
    now = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
    start, end = local_day_bounds("Not/AZone", now)
    assert start == datetime(2026, 8, 4, 0, 0, tzinfo=timezone.utc)


def test_local_day_bounds_shifted_zone() -> None:
    now = datetime(2026, 8, 4, 4, 0, tzinfo=timezone.utc)
    start, end = local_day_bounds("America/New_York", now)
    assert start == datetime(2026, 8, 4, 4, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 8, 5, 4, 0, tzinfo=timezone.utc)


# ── briefing text ────────────────────────────────────────────────────────


def test_build_overview_text_morning_greets_by_name() -> None:
    from companion.relaydata import OverviewData, OverviewTask

    data = OverviewData(
        type="morning",
        first_name="Sam",
        tasks_due_today=[OverviewTask(title="Ship launch email")],
        completed_yesterday=[OverviewTask(title="Fix login bug")],
    )
    text = build_overview_text(data, now=datetime(2026, 8, 4, 9, 0, tzinfo=timezone.utc))
    assert "Good morning, Sam." in text
    assert "TODAY'S SCHEDULE" in text
    assert "- Ship launch email" in text
    assert "No tasks overdue" in text or "Nothing open right now" in text


def test_build_overview_text_honest_empty_states() -> None:
    from companion.relaydata import OverviewData

    text = build_overview_text(OverviewData(type="morning"), now=datetime(2026, 8, 4, 9, 0, tzinfo=timezone.utc))
    assert "No tasks due today." in text
    assert "Inbox clear" in text


# ── system blocks ────────────────────────────────────────────────────────


def test_bundle_to_system_blocks_includes_profile() -> None:
    bundle = RelayContextBundle(
        profile=UserProfile(user_id="u1", first_name="Sam", communication_style="direct"),
        graph_facts=["Sam prefers async updates"],
        companion_tasks=[{"title": "Approve lead workflow", "priority": "high"}],
    )
    blocks = bundle.to_system_blocks()
    joined = "\n".join(blocks)
    assert "User profile:" in joined
    assert "Sam" in joined
    assert "direct" in joined
    assert "Approve lead workflow" in joined


# ── REST filters (duplicate-column params must survive) ─────────────────


def test_rest_select_keeps_duplicate_column_filters(sb_settings) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)

    async def scenario() -> None:
        rest = _Rest(sb_settings)
        async with httpx.AsyncClient(transport=transport) as client:
            await rest.select(
                "workflow_tasks",
                columns="title",
                filters=[
                    ("workspace_id", "eq.w1"),
                    ("status", "eq.done"),
                    ("updated_at", "gte.2026-08-03T00:00:00+00:00"),
                    ("updated_at", "lt.2026-08-04T00:00:00+00:00"),
                ],
                order="updated_at.desc",
                client=client,
            )

    asyncio.run(scenario())
    assert len(captured) == 1
    pairs = list(captured[0].url.params.multi_items())
    updated_at = [v for k, v in pairs if k == "updated_at"]
    assert "gte.2026-08-03T00:00:00+00:00" in updated_at
    assert "lt.2026-08-04T00:00:00+00:00" in updated_at


def test_relay_context_service_degrades_without_credentials(settings) -> None:
    """No Supabase configured → every leg returns safe defaults, never raises."""

    async def scenario() -> None:
        service = RelayContextService(settings)
        bundle = await service.build_context("u1", "w1", "what's on my plate today")
        assert bundle.profile.user_id == "u1"
        assert bundle.overview == ""
        assert bundle.workspace.workspace_id == "w1"
        assert bundle.graph_facts == []
        assert bundle.companion_tasks == []
        await service.record_turn(user_id="u1", workspace_id="w1", role="assistant", message="hi")

    asyncio.run(scenario())


def test_relay_context_respects_disable_knob(settings) -> None:
    settings.enable_relay_context = False

    async def scenario() -> None:
        service = RelayContextService(settings)
        bundle = await service.build_context("u1", "w1", "what's on my plate today")
        assert bundle.profile.source == "disabled"
        assert bundle.overview == ""
        assert bundle.workspace.to_block() == ""
        assert bundle.companion_tasks == []
        assert bundle.overview_requested is True

    asyncio.run(scenario())
