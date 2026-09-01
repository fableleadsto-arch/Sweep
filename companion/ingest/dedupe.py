"""Deduplication for ingested documents.

Two layers, both deterministic and testable:

* **Exact** — a SHA-256 of the normalized content plus the canonical URL. Two
  fetches of the same page / feed item collapse to one document.
* **Near-duplicate** — shingle-set Jaccard similarity over normalized text.
  Republished or lightly rewritten articles are caught even when the hash
  differs. An optional embedding-cosine check is applied by the pipeline when
  embeddings are available.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional

_WS = re.compile(r"\s+")
_ALNUM = re.compile(r"[^a-z0-9 ]")


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for hashing."""
    if not text:
        return ""
    lowered = _WS.sub(" ", text.lower())
    return _ALNUM.sub(" ", lowered).strip()


def content_hash(content: str) -> str:
    """SHA-256 hex of the normalized content — the exact-dup fingerprint."""
    return hashlib.sha256(normalize_text(content).encode("utf-8")).hexdigest()


def url_key(url: str) -> str:
    """Normalize a URL for identity checks (scheme/host case, trailing slash)."""
    return normalize_text(url).rstrip()


def shingles(text: str, n: int = 4) -> set[tuple[str, ...]]:
    """Word n-grams used for near-duplicate detection."""
    words = normalize_text(text).split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def shingle_jaccard(a: str, b: str, n: int = 4) -> float:
    """Jaccard similarity of two texts' shingle sets (0.0–1.0)."""
    set_a, set_b = shingles(a, n), shingles(b, n)
    if not set_a or not set_b:
        return 0.0 if (set_a or set_b) else 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


NEAR_DUPLICATE_THRESHOLD = 0.85


def is_near_duplicate(a: str, b: str, *, threshold: float = NEAR_DUPLICATE_THRESHOLD) -> bool:
    """True when two texts are likely the same underlying article."""
    return shingle_jaccard(a, b) >= threshold


def embedding_cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for the optional embedding-based dedup check."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
