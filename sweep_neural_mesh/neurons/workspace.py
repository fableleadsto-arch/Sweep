"""
Global Workspace Broadcasting — the Forebrain's shared consciousness.

Implements Baars' Global Workspace Theory (GWT):

    The brain doesn't have a single "central processor" — instead,
    specialized processing centers compete for access to a shared
    workspace. When a center finds something important, it broadcasts
    its findings to ALL other centers via the global workspace.

    This is the mechanism behind consciousness: a "spotlight" of
    attention that makes information available to the entire brain.

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │                GLOBAL WORKSPACE                     │
    │  ┌─────────────────────────────────────────────┐   │
    │  │  Shared Blackboard: current findings,        │   │
    │  │  hypotheses, contradictions, key evidence     │   │
    │  └─────────────────────────────────────────────┘   │
    │         ↑↓        ↑↓        ↑↓        ↑↓           │
    │    Evidence  Credibility  Causal   Contradiction   │
    │    Center    Center       Center   Detector        │
    │         ↑↓        ↑↓        ↑↓        ↑↓           │
    │              Integration Hub + Consensus            │
    └─────────────────────────────────────────────────────┘

Key properties:
1. BROADCASTING: When a center publishes, ALL centers see it
2. COMPETITION: Centers compete for workspace access (saliency wins)
3. IGNITION: When enough evidence accumulates, workspace "ignites"
   and a global broadcast triggers decision-making
4. MODULATION: Workspace contents modulate all center processing
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkspaceEntry:
    """A single entry on the global workspace blackboard."""
    entry_id: str
    source_center: str              # which center published this
    content: dict[str, Any]         # the actual content
    salience: float                 # 0.0-1.0: how important is this
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0           # how many centers have read this
    decay_rate: float = 0.01        # how fast this entry fades

    @property
    def is_stale(self) -> bool:
        """Check if this entry has decayed below relevance."""
        age = time.time() - self.timestamp
        return self.salience * (1.0 - self.decay_rate * age) < 0.1


@dataclass
class BroadcastResult:
    """Result of a workspace broadcast."""
    broadcast_id: str
    entry: WorkspaceEntry
    reached_centers: list[str]      # which centers received the broadcast
    salience_rank: int              # rank among all workspace entries
    workspace_size: int
    triggered_ignition: bool        # did this broadcast trigger decision?


class GlobalWorkspace:
    """
    The shared workspace for inter-center communication.

    Like Baars' Global Workspace Theory, this module:

    1. Maintains a shared blackboard that all centers can read/write
    2. Handles broadcasting: when a center publishes, all others receive
    3. Manages competition: high-salience entries get more attention
    4. Detects ignition: when enough evidence accumulates, trigger consensus
    5. Applies decay: old entries fade unless refreshed

    The workspace is the mechanism that creates "awareness" — without it,
    each center works in isolation. With it, a contradiction detector's
    finding can immediately influence the credibility assessor's evaluation.

    Properties:
    - Capacity-limited: only N items active at once (like working memory)
    - Priority-scheduled: most salient items get broadcast first
    - Decaying: entries fade unless reinforced by new evidence
    - Competitive: new entries must compete with existing ones for space
    """

    def __init__(self, capacity: int = 12) -> None:
        # The shared blackboard
        self._entries: list[WorkspaceEntry] = []
        self._capacity = capacity

        # Broadcasting statistics
        self._broadcast_count = 0
        self._ignition_count = 0
        self._total_reads = 0

        # Ignition threshold: how many high-salience entries trigger decision
        self._ignition_threshold = 3
        self._ignition_salience_threshold = 0.6

    def publish(
        self,
        source_center: str,
        content: dict[str, Any],
        salience: float = 0.5,
    ) -> BroadcastResult:
        """
        Publish content from a processing center to the global workspace.

        This is the "writing" operation: a center puts its findings
        on the shared blackboard for all other centers to see.
        """
        self._broadcast_count += 1

        # Create entry
        entry = WorkspaceEntry(
            entry_id=f"ws_{self._broadcast_count}",
            source_center=source_center,
            content=content,
            salience=salience,
        )

        # If workspace is full, evict lowest-salience entry
        if len(self._entries) >= self._capacity:
            self._entries.sort(key=lambda e: e.salience)
            # Only evict if new entry is more salient than the lowest
            if self._entries[0].salience < salience:
                self._entries.pop(0)
            else:
                # New entry isn't important enough — still add but don't evict
                # (overcapacity is allowed temporarily)
                pass

        self._entries.append(entry)

        # Determine which centers would see this (all active centers)
        reached = self._get_active_centers()
        salience_rank = sum(1 for e in self._entries if e.salience > salience) + 1

        # Check for ignition
        high_salience_count = sum(
            1 for e in self._entries
            if e.salience >= self._ignition_salience_threshold
        )
        triggered_ignition = high_salience_count >= self._ignition_threshold
        if triggered_ignition:
            self._ignition_count += 1

        return BroadcastResult(
            broadcast_id=entry.entry_id,
            entry=entry,
            reached_centers=reached,
            salience_rank=salience_rank,
            workspace_size=len(self._entries),
            triggered_ignition=triggered_ignition,
        )

    def read(
        self,
        requesting_center: str,
        max_entries: int = 5,
    ) -> list[WorkspaceEntry]:
        """
        Read the most salient entries from the workspace.

        This is the "reading" operation: a center looks at the
        shared blackboard to see what other centers have found.
        """
        self._total_reads += 1

        # Sort by salience (most important first)
        available = [e for e in self._entries if not e.is_stale]
        available.sort(key=lambda e: e.salience, reverse=True)

        # Take top N
        selected = available[:max_entries]

        # Record access
        for entry in selected:
            entry.access_count += 1

        return selected

    def get_workspace_state(self) -> dict[str, Any]:
        """
        Get a summary of the current workspace state.

        Used by centers to understand what information is currently
        available in the shared workspace.
        """
        active = [e for e in self._entries if not e.is_stale]
        return {
            "active_entries": len(active),
            "total_entries": len(self._entries),
            "capacity": self._capacity,
            "top_salience": active[0].salience if active else 0.0,
            "sources": list(set(e.source_center for e in active)),
            "avg_age_seconds": (
                sum(time.time() - e.timestamp for e in active) / len(active)
                if active else 0.0
            ),
        }

    def apply_modulation(self, center_name: str) -> dict[str, Any]:
        """
        Generate modulation signals for a center based on workspace contents.

        Like how the global workspace modulates processing in all centers,
        this returns relevant information that should influence a center's
        processing.
        """
        active = [e for e in self._entries if not e.is_stale]

        # Find entries relevant to this center
        relevant = []
        for entry in active:
            # All entries are potentially relevant (global broadcast)
            relevant.append({
                "source": entry.source_center,
                "salience": entry.salience,
                "content_keys": list(entry.content.keys()),
                "age_seconds": time.time() - entry.timestamp,
            })

        # Summarize contradictions found
        contradictions = [
            e for e in active
            if "contradiction" in str(e.content).lower()
            or e.source_center == "contradiction_detector"
        ]

        # Summarize credibility concerns
        credibility = [
            e for e in active
            if "credibility" in str(e.content).lower()
            or e.source_center == "credibility_assessor"
        ]

        return {
            "relevant_entries": relevant[:10],
            "contradiction_count": len(contradictions),
            "credibility_concerns": len(credibility),
            "workspace_ignitions": self._ignition_count,
            "total_broadcasts": self._broadcast_count,
        }

    def decay_entries(self) -> int:
        """
        Decay all entries and remove stale ones.

        Returns the number of entries removed.
        """
        before = len(self._entries)
        self._entries = [e for e in self._entries if not e.is_stale]
        return before - len(self._entries)

    def clear(self) -> None:
        """Clear the workspace (e.g., between reasoning episodes)."""
        self._entries.clear()

    def _get_active_centers(self) -> list[str]:
        """Get list of active processing centers."""
        return [
            "evidence_gatherer",
            "credibility_assessor",
            "temporal_sequencer",
            "causal_linker",
            "contradiction_detector",
            "explanation_builder",
        ]

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def stats(self) -> dict[str, Any]:
        active = [e for e in self._entries if not e.is_stale]
        return {
            "active_entries": len(active),
            "total_entries": len(self._entries),
            "capacity": self._capacity,
            "broadcast_count": self._broadcast_count,
            "ignition_count": self._ignition_count,
            "total_reads": self._total_reads,
            "avg_salience": (
                sum(e.salience for e in active) / len(active)
                if active else 0.0
            ),
        }
