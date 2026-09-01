"""Latch behaviour for the Qdrant credential-rejection path in companion/memory.py.

Mirrors the TS tests in src/RelAI/memory/qdrant.server.test.ts: a rejected
Qdrant key (``unauthorised: Invalid key``) must latch for a TTL, log exactly
one actionable warning, and make ``_get_client`` return None so every memory
operation falls back to the file store — then re-probe after the TTL.
"""

from __future__ import annotations

import time

from companion.memory import AUTH_REJECT_TTL_S, QdrantMemoryStore


class AuthError(Exception):
    """Qdrant's signature auth rejection — status 401 with the raw body."""

    def __init__(self) -> None:
        super().__init__('401 {"status":{"error":"unauthorised: Invalid key"}}')


def _store(settings, file_store) -> QdrantMemoryStore:
    """A store with a client already constructed (so the latch is the only gate)."""
    store = QdrantMemoryStore(settings, file_store)
    store._client = object()  # type: ignore[assignment]  # sentinel, never used
    return store


def test_is_auth_rejection_matches_qdrant_message(settings, file_store) -> None:
    store = _store(settings, file_store)
    assert store._is_auth_rejection(AuthError()) is True


def test_is_auth_rejection_matches_http_status(settings, file_store) -> None:
    store = _store(settings, file_store)

    class Forbidden(Exception):
        status_code = 403

    class Unauthorized(Exception):
        status_code = 401

    assert store._is_auth_rejection(Forbidden("nope")) is True
    assert store._is_auth_rejection(Unauthorized("nope")) is True


def test_non_auth_failure_does_not_latch(settings, file_store) -> None:
    store = _store(settings, file_store)
    assert store._note_auth_rejection(Exception("upstream 500")) is False
    assert store._auth_rejected_at is None


def test_latch_logs_once_and_blocks_client(settings, file_store, caplog) -> None:
    store = _store(settings, file_store)
    with caplog.at_level("WARNING", logger="companion.memory"):
        assert store._note_auth_rejection(AuthError()) is True
        assert store._note_auth_rejection(AuthError()) is True

    warnings = [r.message for r in caplog.records if "API key rejected" in r.message]
    assert len(warnings) == 1
    assert "unauthorised: Invalid key" in warnings[0]

    # While latched, _get_client() returns None even though a client exists.
    assert store._auth_rejected() is True
    assert store._get_client() is None


def test_recovery_after_ttl(settings, file_store) -> None:
    store = _store(settings, file_store)
    store._auth_rejected_at = time.time() - AUTH_REJECT_TTL_S - 1.0
    assert store._auth_rejected() is False
    assert store._get_client() is not None
