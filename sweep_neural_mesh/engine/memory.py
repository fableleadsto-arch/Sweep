"""
Memory System — multi-layered memory for the Sweep Neural Engine.

Layers:
    Working Memory:   temporary information for the current task
    Evidence Memory:  sources, claims, entities, timestamps, relationships
    Semantic Memory:  embedding-based representations
    User Memory:      only information explicitly authorized for persistence

Every evidence object supports provenance tracking and status lifecycle:
    UNVERIFIED -> SUPPORTED / CONTRADICTED / DISPUTED -> VERIFIED

Sweep-original implementation. Does not silently convert inference into fact.
"""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("sweep.memory")


# ════════════════════════════════════════════════════════════════════
# EVIDENCE STATUS
# ════════════════════════════════════════════════════════════════════

class EvidenceStatus(Enum):
    UNVERIFIED = "UNVERIFIED"
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    DISPUTED = "DISPUTED"
    VERIFIED = "VERIFIED"


# ════════════════════════════════════════════════════════════════════
# EVIDENCE OBJECT
# ════════════════════════════════════════════════════════════════════

@dataclass
class EvidenceObject:
    """A piece of evidence with full provenance tracking."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source: str = "unknown"
    timestamp: str | None = None
    content: str = ""
    claim: str = ""
    confidence: float = 0.5
    provenance: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
    status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def content_hash(self) -> str:
        """Hash of content for deduplication."""
        return hashlib.md5(self.content.encode()).hexdigest()[:12]

    def update_status(self, new_status: EvidenceStatus, reason: str = "") -> None:
        """Update status with provenance."""
        old = self.status
        self.status = new_status
        if reason:
            self.provenance.append(f"{old.value} -> {new_status.value}: {reason}")
        logger.info(f"Evidence {self.id}: {old.value} -> {new_status.value}")


# ════════════════════════════════════════════════════════════════════
# WORKING MEMORY
# ════════════════════════════════════════════════════════════════════

class WorkingMemory:
    """Temporary information for the current task.

    Cleared when the task completes. Stores intermediate results,
    partial reasoning chains, and temporary data structures.
    """

    def __init__(self, max_items: int = 100) -> None:
        self._max_items = max_items
        self._items: dict[str, Any] = {}
        self._created = time.time()

    def put(self, key: str, value: Any) -> None:
        """Store a temporary item."""
        if len(self._items) >= self._max_items:
            # Evict oldest
            oldest = min(self._items, key=lambda k: self._items[k].get("_ts", 0) if isinstance(self._items[k], dict) else 0)
            del self._items[oldest]
        self._items[key] = {"value": value, "_ts": time.time()}

    def get(self, key: str) -> Any | None:
        """Retrieve a temporary item."""
        item = self._items.get(key)
        return item["value"] if item else None

    def has(self, key: str) -> bool:
        return key in self._items

    def clear(self) -> None:
        """Clear all working memory."""
        self._items.clear()

    def snapshot(self) -> dict[str, Any]:
        """Get a snapshot of current working memory."""
        return {k: v["value"] for k, v in self._items.items()}

    def __len__(self) -> int:
        return len(self._items)


# ════════════════════════════════════════════════════════════════════
# EVIDENCE MEMORY
# ════════════════════════════════════════════════════════════════════

class EvidenceMemory:
    """Stores sources, claims, entities, timestamps, and relationships.

    Evidence goes through a lifecycle:
        UNVERIFIED -> SUPPORTED / CONTRADICTED / DISPUTED -> VERIFIED

    Never silently converts inference into fact.
    """

    def __init__(self, max_items: int = 10000) -> None:
        self._max_items = max_items
        self._evidence: dict[str, EvidenceObject] = {}
        self._by_hash: dict[str, str] = {}  # content_hash -> id
        self._by_source: dict[str, list[str]] = {}  # source -> [ids]
        self._by_status: dict[str, list[str]] = {}  # status -> [ids]

    def add(self, evidence: EvidenceObject) -> EvidenceObject:
        """Add evidence, deduplicating by content hash."""
        h = evidence.content_hash
        if h in self._by_hash:
            existing_id = self._by_hash[h]
            existing = self._evidence.get(existing_id)
            if existing:
                # Update confidence if new evidence is more confident
                if evidence.confidence > existing.confidence:
                    existing.confidence = evidence.confidence
                    existing.provenance.append(
                        f"Updated confidence to {evidence.confidence} from {evidence.source}"
                    )
                return existing

        if len(self._evidence) >= self._max_items:
            self._evict_oldest()

        self._evidence[evidence.id] = evidence
        self._by_hash[h] = evidence.id
        self._by_source.setdefault(evidence.source, []).append(evidence.id)
        self._by_status.setdefault(evidence.status.value, []).append(evidence.id)
        return evidence

    def get(self, evidence_id: str) -> EvidenceObject | None:
        return self._evidence.get(evidence_id)

    def query_by_status(self, status: EvidenceStatus) -> list[EvidenceObject]:
        ids = self._by_status.get(status.value, [])
        return [self._evidence[i] for i in ids if i in self._evidence]

    def query_by_source(self, source: str) -> list[EvidenceObject]:
        ids = self._by_source.get(source, [])
        return [self._evidence[i] for i in ids if i in self._evidence]

    def get_supported(self) -> list[EvidenceObject]:
        return self.query_by_status(EvidenceStatus.SUPPORTED)

    def get_contradicted(self) -> list[EvidenceObject]:
        return self.query_by_status(EvidenceStatus.CONTRADICTED)

    def update_status(self, evidence_id: str, status: EvidenceStatus, reason: str = "") -> bool:
        ev = self._evidence.get(evidence_id)
        if not ev:
            return False
        old_status = ev.status.value
        ev.update_status(status, reason)
        # Update index
        self._by_status.setdefault(old_status, [])
        if evidence_id in self._by_status[old_status]:
            self._by_status[old_status].remove(evidence_id)
        self._by_status.setdefault(status.value, [])
        self._by_status[status.value].append(evidence_id)
        return True

    def deduplicate(self) -> int:
        """Remove duplicate evidence. Returns count removed."""
        seen_hashes: dict[str, str] = {}
        to_remove = []
        for eid, ev in self._evidence.items():
            h = ev.content_hash
            if h in seen_hashes:
                to_remove.append(eid)
            else:
                seen_hashes[h] = eid
        for eid in to_remove:
            del self._evidence[eid]
        return len(to_remove)

    def _evict_oldest(self) -> None:
        if self._evidence:
            oldest_id = min(self._evidence, key=lambda k: self._evidence[k].created_at)
            ev = self._evidence.pop(oldest_id)
            self._by_hash.pop(ev.content_hash, None)

    def __len__(self) -> int:
        return len(self._evidence)

    def stats(self) -> dict[str, Any]:
        return {
            "total": len(self._evidence),
            "by_status": {s: len(ids) for s, ids in self._by_status.items()},
            "by_source": {s: len(ids) for s, ids in self._by_source.items()},
        }


# ════════════════════════════════════════════════════════════════════
# SEMANTIC MEMORY
# ════════════════════════════════════════════════════════════════════

class SemanticMemory:
    """Embedding-based representations for similarity search.

    Stores text with embeddings for fast semantic retrieval.
    Uses a simple in-memory index (no external vector DB required).
    """

    def __init__(self, embedding_provider=None) -> None:
        self._provider = embedding_provider
        self._entries: dict[str, dict] = {}  # id -> {text, vector, metadata}
        self._id_counter = 0

    def store(self, text: str, metadata: dict | None = None) -> str:
        """Store text with embedding."""
        self._id_counter += 1
        entry_id = f"sem_{self._id_counter}"

        vector = None
        if self._provider:
            try:
                result = self._provider.embed(text)
                vector = result.vector
            except Exception:
                pass

        self._entries[entry_id] = {
            "text": text,
            "vector": vector,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        return entry_id

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for similar entries."""
        if not self._provider:
            return []

        try:
            q_result = self._provider.embed(query)
            q_vec = q_result.vector
        except Exception:
            return []

        scored = []
        for eid, entry in self._entries.items():
            if entry["vector"] is None:
                continue
            import numpy as np
            score = float(np.dot(q_vec, entry["vector"]) / (
                np.linalg.norm(q_vec) * np.linalg.norm(entry["vector"]) + 1e-8
            ))
            scored.append((eid, score, entry))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {"id": eid, "score": score, "text": entry["text"], "metadata": entry["metadata"]}
            for eid, score, entry in scored[:top_k]
        ]

    def __len__(self) -> int:
        return len(self._entries)


# ════════════════════════════════════════════════════════════════════
# USER MEMORY
# ════════════════════════════════════════════════════════════════════

class UserMemory:
    """Stores only information explicitly authorized for persistence.

    Respects deletion requests. Does not silently store user data.
    """

    def __init__(self) -> None:
        self._authorized: dict[str, Any] = {}
        self._consent_log: list[dict] = []

    def store(self, key: str, value: Any, consent: bool = False) -> bool:
        """Store user data only if consent is given."""
        if not consent:
            self._consent_log.append({
                "action": "denied", "key": key, "timestamp": time.time()
            })
            return False
        self._authorized[key] = value
        self._consent_log.append({
            "action": "stored", "key": key, "timestamp": time.time()
        })
        return True

    def retrieve(self, key: str) -> Any | None:
        return self._authorized.get(key)

    def delete(self, key: str) -> bool:
        if key in self._authorized:
            del self._authorized[key]
            self._consent_log.append({
                "action": "deleted", "key": key, "timestamp": time.time()
            })
            return True
        return False

    def delete_all(self) -> int:
        count = len(self._authorized)
        self._authorized.clear()
        self._consent_log.append({
            "action": "delete_all", "count": count, "timestamp": time.time()
        })
        return count


# ════════════════════════════════════════════════════════════════════
# MEMORY SYSTEM (orchestrator)
# ════════════════════════════════════════════════════════════════════

class MemorySystem:
    """Orchestrates all memory layers.

    Sweep-original implementation. Each layer is independent and replaceable.
    """

    def __init__(self, embedding_provider=None) -> None:
        self.working = WorkingMemory()
        self.evidence = EvidenceMemory()
        self.semantic = SemanticMemory(embedding_provider)
        self.user = UserMemory()

    def clear_working(self) -> None:
        """Clear working memory after task completion."""
        self.working.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "working_memory": len(self.working),
            "evidence_memory": self.evidence.stats(),
            "semantic_memory": len(self.semantic),
            "user_memory": len(self.user._authorized),
        }
