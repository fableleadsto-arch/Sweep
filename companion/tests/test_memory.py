from __future__ import annotations

from companion.memory import dedupe_by_content, score_entry
from companion.memory import MemoryEntry


def _entry(content: str, updated: str = "2026-08-01T00:00:00+00:00") -> MemoryEntry:
    return MemoryEntry(
        id="1",
        user_id="u",
        content=content,
        updated_at=updated,
    )


def test_score_prefers_overlap_and_phrase() -> None:
    close = _entry("the Jones proposal is due this week")
    unrelated = _entry("we shipped the mobile app release")
    assert score_entry("jones proposal deadline", close) > score_entry(
        "jones proposal deadline", unrelated
    )


def test_score_decays_with_age() -> None:
    fresh = _entry("client prefers morning calls", "2026-08-03T00:00:00+00:00")
    old = _entry("client prefers morning calls", "2026-01-01T00:00:00+00:00")
    assert score_entry("client morning calls", fresh) > score_entry("client morning calls", old)


def test_dedupe_drops_near_duplicates() -> None:
    a = _entry("prefers morning calls and short agendas")
    b = _entry("prefers morning calls and short agendas (noted)")
    result = dedupe_by_content([(a, 10.0), (b, 9.0)])
    assert len(result) == 1


def test_dedupe_keeps_distinct_facts() -> None:
    a = _entry("prefers morning calls")
    b = _entry("wants invoice templates in Notion")
    result = dedupe_by_content([(a, 10.0), (b, 9.0)])
    assert len(result) == 2


def test_remember_dedupes_exact_content(file_store) -> None:
    first = file_store.remember("u", "   user prefers  email   ")
    second = file_store.remember("u", "user prefers email")
    assert first.id == second.id
    assert len(file_store._load()) == 1


def test_search_ranks_relevant_first(file_store) -> None:
    file_store.remember("u", "the Jones proposal is due this week", source="conv")
    file_store.remember("u", "ordering new office furniture", source="conv")
    results = file_store.search("u", "jones proposal due")
    assert results[0].content.startswith("the Jones proposal")


def test_search_empty_query_returns_recent_first(file_store) -> None:
    file_store.remember("u", "old memory", kind="fact")
    file_store.remember("u", "newer memory", kind="fact")
    results = file_store.search("u", query="")
    assert results[0].content == "newer memory"


def test_profile_splits_facts_and_context(memory_service) -> None:
    memory_service.file_store.remember("u", "prefers concise replies", kind="preference")
    memory_service.file_store.remember("u", "mentioned vacation next month", kind="context")
    profile = memory_service.profile("u")
    assert "prefers concise replies" in profile["facts"]
    assert "mentioned vacation next month" in profile["recent_context"]
