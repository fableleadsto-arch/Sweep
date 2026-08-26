"""
Telemetry — records Mesh execution metrics, profiling, and tracing.

Every node execution, routing decision, fusion operation, and
verification is recorded. This enables:
  - Performance analysis
  - Routing optimization
  - Failure diagnosis
  - Scientific comparison of pipelines
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    """A single trace event in the Mesh."""
    timestamp: float
    event_type: str
    node_id: str = ""
    node_name: str = ""
    graph_id: str = ""
    duration_ms: float = 0.0
    success: bool = True
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Telemetry:
    """
    Records and queries Mesh execution telemetry.

    Lightweight by default. Can be disabled entirely in production
    by setting enabled=False.
    """

    def __init__(self, enabled: bool = True, max_events: int = 10000) -> None:
        self.enabled = enabled
        self.max_events = max_events
        self._events: list[TraceEvent] = []
        self._counters: dict[str, int] = defaultdict(int)
        self._timers: dict[str, list[float]] = defaultdict(list)

    def record(
        self,
        event_type: str,
        node_id: str = "",
        node_name: str = "",
        graph_id: str = "",
        duration_ms: float = 0.0,
        success: bool = True,
        confidence: float = 0.0,
        **metadata: Any,
    ) -> None:
        if not self.enabled:
            return
        event = TraceEvent(
            timestamp=time.time(),
            event_type=event_type,
            node_id=node_id,
            node_name=node_name,
            graph_id=graph_id,
            duration_ms=duration_ms,
            success=success,
            confidence=confidence,
            metadata=metadata,
        )
        self._events.append(event)
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events:]
        self._counters[event_type] += 1
        if duration_ms > 0:
            self._timers[event_type].append(duration_ms)

    def inc_counter(self, name: str, amount: int = 1) -> None:
        self._counters[name] += amount

    def get_counter(self, name: str) -> int:
        return self._counters[name]

    def avg_latency(self, event_type: str) -> float:
        times = self._timers.get(event_type, [])
        if not times:
            return 0.0
        return sum(times) / len(times)

    def recent_events(self, n: int = 50) -> list[TraceEvent]:
        return self._events[-n:]

    def events_by_type(self, event_type: str) -> list[TraceEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def summary(self) -> dict[str, Any]:
        total = len(self._events)
        failures = sum(1 for e in self._events if not e.success)
        return {
            "total_events": total,
            "failures": failures,
            "success_rate": (total - failures) / total if total > 0 else 0,
            "counters": dict(self._counters),
            "avg_latencies": {
                k: self.avg_latency(k) for k in self._timers
            },
        }

    def clear(self) -> None:
        self._events.clear()
        self._counters.clear()
        self._timers.clear()

    def __repr__(self) -> str:
        return f"Telemetry(events={len(self._events)}, counters={len(self._counters)})"
