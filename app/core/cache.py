"""Request/result caching — shared across search and research modules."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Optional

_CACHE_TTL_MS = 300_000  # 5 minutes
_MAX_ENTRIES = 500

_store: dict[str, tuple[float, Any]] = {}


def cache_key(*parts: Any) -> str:
    """Build a deterministic cache key from arbitrary parts."""
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def cache_get(key: str) -> Optional[Any]:
    """Retrieve a cached value if still valid."""
    entry = _store.get(key)
    if entry is None:
        return None
    at, value = entry
    if (time.time() - at) * 1000 > _CACHE_TTL_MS:
        _store.pop(key, None)
        return None
    return value


def cache_set(key: str, value: Any) -> None:
    """Store a value in the cache."""
    if len(_store) >= _MAX_ENTRIES:
        oldest_key = min(_store, key=lambda k: _store[k][0])
        _store.pop(oldest_key, None)
    _store[key] = (time.time(), value)


def cache_clear() -> int:
    """Clear the cache. Returns number of entries removed."""
    count = len(_store)
    _store.clear()
    return count
