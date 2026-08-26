"""
Working Memory — the Forebrain's active context buffer.

Implements Baddeley's Working Memory Model adapted for reasoning:

    Working memory is NOT long-term storage — it's a limited-capacity
    "scratchpad" that holds the current reasoning context. Like the
    brain's prefrontal cortex maintaining active goals, recent findings,
    and pending actions.

Architecture (adapted from Baddeley):

    ┌─────────────────────────────────────────────────────┐
    │                WORKING MEMORY                        │
    │                                                     │
    │  ┌───────────────┐  ┌───────────────────────────┐  │
    │  │ Phonological   │  │ Visuospatial Sketchpad   │  │
    │  │ Loop           │  │ (Not applicable — we're   │  │
    │  │ (Query context)│  │  text-based)              │  │
    │  └───────┬───────┘  └───────────┬───────────────┘  │
    │          ↓                       ↓                  │
    │  ┌─────────────────────────────────────────────┐   │
    │  │         CENTRAL EXECUTIVE                    │   │
    │  │  Manages attention, coordinates sub-systems  │   │
    │  │  Decides what enters/leaves memory           │   │
    │  └─────────────────┬───────────────────────────┘   │
    │                    ↓                                │
    │  ┌─────────────────────────────────────────────┐   │
    │  │         EPISODIC BUFFER                      │   │
    │  │  Integrates info from all sub-systems        │   │
    │  │  Links working memory to long-term memory    │   │
    │  └─────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────┘

Key properties:
1. CAPACITY-LIMITED: holds 4-7 items (Miller's Law)
2. TEMPORAL: items decay over time unless rehearsed
3. INTERFERENCE: similar items interfere with each other
4. CENTRAL EXECUTIVE: controls what enters and what's discarded
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemorySlot(Enum):
    """Types of working memory slots."""
    QUERY = "query"                # the current query being processed
    GOAL = "goal"                  # current reasoning goal
    FINDING = "finding"            # a finding from a processing center
    HYPOTHESIS = "hypothesis"      # a working hypothesis
    ACTION = "action"              # a pending action
    CONTEXT = "context"            # contextual information


@dataclass
class WorkingMemoryItem:
    """A single item in working memory."""
    item_id: str
    slot_type: MemorySlot
    content: dict[str, Any]
    priority: float                # 0.0-1.0: how important to keep
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    rehearsal_count: int = 0       # how many times it's been refreshed

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def staleness(self) -> float:
        """How stale this item is (0.0 = fresh, 1.0 = very stale)."""
        time_since_access = time.time() - self.last_accessed
        return min(1.0, time_since_access / 300.0)  # decays over 5 minutes


class WorkingMemory:
    """
    Active context buffer for ongoing reasoning.

    Like the brain's working memory that maintains current goals,
    recent findings, and pending actions during reasoning, this module:

    1. Holds limited items (4-7, like Miller's Law)
    2. Applies temporal decay: items fade unless rehearsed
    3. Manages interference: similar items compete for slots
    4. Integrates with long-term memory (episodic/semantic)
    5. Provides context to all processing centers

    Working memory is the "gateway" between perception and action:
    it holds the current state of reasoning and makes it available
    to all processing centers simultaneously.

    Without working memory, each reasoning step would start from scratch.
    With it, we maintain context across steps: "I found X, which
    contradicts Y, so I should check Z next."
    """

    def __init__(self, capacity: int = 7) -> None:
        self._items: list[WorkingMemoryItem] = []
        self._capacity = capacity
        self._total_inserts = 0
        self._total_evictions = 0
        self._total_rehearsals = 0

    def insert(
        self,
        slot_type: MemorySlot,
        content: dict[str, Any],
        priority: float = 0.5,
    ) -> WorkingMemoryItem:
        """
        Insert a new item into working memory.

        If at capacity, the lowest-priority item is evicted.
        """
        self._total_inserts += 1

        item = WorkingMemoryItem(
            item_id=f"wm_{self._total_inserts}",
            slot_type=slot_type,
            content=content,
            priority=priority,
        )

        # If at capacity, evict lowest priority
        if len(self._items) >= self._capacity:
            self._items.sort(key=lambda i: i.priority)
            # Also factor in staleness
            evict_idx = 0
            for idx, existing in enumerate(self._items):
                # Evict the item with lowest (priority * (1 - staleness))
                existing_score = existing.priority * (1.0 - existing.staleness * 0.5)
                if idx == 0:
                    continue
                best_score = self._items[evict_idx].priority * (1.0 - self._items[evict_idx].staleness * 0.5)
                if existing_score < best_score:
                    evict_idx = idx

            evicted = self._items.pop(evict_idx)
            self._total_evictions += 1

            # Transfer evicted item to long-term memory hint
            self._on_eviction(evicted)

        self._items.append(item)
        return item

    def retrieve(
        self,
        slot_type: MemorySlot | None = None,
        max_items: int = 5,
    ) -> list[WorkingMemoryItem]:
        """
        Retrieve items from working memory.

        Returns items sorted by priority (most important first).
        Optionally filter by slot type.
        """
        items = self._items
        if slot_type:
            items = [i for i in items if i.slot_type == slot_type]

        # Sort by priority * freshness (recently accessed items rank higher)
        items.sort(
            key=lambda i: i.priority * (1.0 - i.staleness * 0.3),
            reverse=True,
        )

        # Record access
        for item in items[:max_items]:
            item.last_accessed = time.time()
            item.access_count += 1

        return items[:max_items]

    def rehearse(self, item_id: str) -> bool:
        """
        Rehearse an item to prevent decay.

        Like the phonological loop that refreshes items through
        subvocalization, this resets the decay timer and boosts priority.
        """
        for item in self._items:
            if item.item_id == item_id:
                item.last_accessed = time.time()
                item.rehearsal_count += 1
                # Rehearsal slightly boosts priority
                item.priority = min(1.0, item.priority + 0.05)
                self._total_rehearsals += 1
                return True
        return False

    def update_item(
        self,
        item_id: str,
        content: dict[str, Any] | None = None,
        priority: float | None = None,
    ) -> bool:
        """Update an existing item's content or priority."""
        for item in self._items:
            if item.item_id == item_id:
                if content is not None:
                    item.content = {**item.content, **content}
                if priority is not None:
                    item.priority = priority
                item.last_accessed = time.time()
                return True
        return False

    def get_context_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current working memory state.

        Used by processing centers to understand the current
        reasoning context without reading every item.
        """
        by_type: dict[str, list[WorkingMemoryItem]] = {}
        for item in self._items:
            slot_name = item.slot_type.value
            if slot_name not in by_type:
                by_type[slot_name] = []
            by_type[slot_name].append(item)

        summary = {
            "total_items": len(self._items),
            "capacity": self._capacity,
            "by_type": {
                slot: len(items) for slot, items in by_type.items()
            },
            "avg_priority": (
                sum(i.priority for i in self._items) / len(self._items)
                if self._items else 0.0
            ),
            "avg_staleness": (
                sum(i.staleness for i in self._items) / len(self._items)
                if self._items else 0.0
            ),
            "recent_findings": [
                {
                    "content_keys": list(i.content.keys()),
                    "priority": i.priority,
                }
                for i in sorted(
                    [i for i in self._items if i.slot_type == MemorySlot.FINDING],
                    key=lambda x: x.last_accessed,
                    reverse=True,
                )[:3]
            ],
        }
        return summary

    def decay_all(self) -> int:
        """
        Apply temporal decay to all items.

        Returns number of items that fell below minimum priority.
        """
        before = len(self._items)
        self._items = [
            i for i in self._items
            if i.priority * (1.0 - i.staleness * 0.5) > 0.05
        ]
        return before - len(self._items)

    def clear(self) -> None:
        """Clear all working memory (e.g., between reasoning episodes)."""
        self._items.clear()

    def _on_eviction(self, item: WorkingMemoryItem) -> None:
        """
        Called when an item is evicted from working memory.

        This is the bridge to long-term memory: evicted items
        that were high-priority get a hint stored for potential
        recall later.
        """
        # High-priority evicted items leave a "trace"
        if item.priority > 0.6:
            # In a full implementation, this would store to episodic memory
            # For now, we just track it
            pass

    @property
    def size(self) -> int:
        return len(self._items)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._items),
            "capacity": self._capacity,
            "total_inserts": self._total_inserts,
            "total_evictions": self._total_evictions,
            "total_rehearsals": self._total_rehearsals,
            "avg_priority": (
                sum(i.priority for i in self._items) / len(self._items)
                if self._items else 0.0
            ),
        }
