"""
Automatic Evidence Reports — structured investigation report generation.

The final output shouldn't just be a pile of search results.
It should produce something like:

    INVESTIGATION SUMMARY
    ─────────────────────
    Target: Person X
    Confidence: Moderate
    Evidence discovered: 23
    Independent sources: 11
    Conflicting sources: 2
    Timeline: 2019 → 2026

    Key findings: ...
    Supporting evidence: ...
    Contradictory evidence: ...
    Unresolved questions: ...
    Sources: ...

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │         EVIDENCE REPORT GENERATOR                    │
    │                                                     │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Summary Compiler                            │  │
    │  │  (aggregate evidence into overview)          │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Finding Extractor                           │  │
    │  │  (key insights from evidence)                │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Confidence Assessor                         │  │
    │  │  (overall confidence rating)                 │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Report Formatter                            │  │
    │  │  (structured output in multiple formats)     │  │
    │  └──────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Finding:
    """A key finding from the investigation."""
    description: str
    confidence: float
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    category: str = "general"  # identity, location, activity, etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "confidence": self.confidence,
            "supporting_count": len(self.supporting_evidence),
            "contradicting_count": len(self.contradicting_evidence),
            "sources": self.sources,
            "category": self.category,
        }


@dataclass
class InvestigationReport:
    """A structured investigation report."""
    target: str
    target_type: str

    # Summary
    confidence_level: str = "uncertain"  # confirmed, likely, possible, uncertain, contradicted
    confidence_score: float = 0.0

    # Evidence counts
    total_evidence: int = 0
    independent_sources: int = 0
    conflicting_sources: int = 0

    # Timeline
    timeline_start: str = ""
    timeline_end: str = ""

    # Content
    key_findings: list[Finding] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    contradictory_evidence: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    # Metadata
    generated_at: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "target_type": self.target_type,
            "confidence_level": self.confidence_level,
            "confidence_score": self.confidence_score,
            "total_evidence": self.total_evidence,
            "independent_sources": self.independent_sources,
            "conflicting_sources": self.conflicting_sources,
            "timeline": f"{self.timeline_start} → {self.timeline_end}" if self.timeline_start else "",
            "key_findings": [f.to_dict() for f in self.key_findings],
            "supporting_evidence_count": len(self.supporting_evidence),
            "contradictory_evidence_count": len(self.contradictory_evidence),
            "unresolved_questions": self.unresolved_questions,
            "sources": self.sources,
        }

    def to_text(self) -> str:
        """Generate a human-readable text report."""
        lines = [
            "═" * 60,
            "INVESTIGATION REPORT",
            "═" * 60,
            "",
            f"Target: {self.target}",
            f"Type: {self.target_type}",
            f"Confidence: {self.confidence_level.upper()} ({self.confidence_score:.0%})",
            "",
            "─" * 60,
            "SUMMARY",
            "─" * 60,
            f"Total evidence: {self.total_evidence}",
            f"Independent sources: {self.independent_sources}",
            f"Conflicting sources: {self.conflicting_sources}",
        ]

        if self.timeline_start:
            lines.append(f"Timeline: {self.timeline_start} → {self.timeline_end}")

        lines.extend(["", "─" * 60, "KEY FINDINGS", "─" * 60])

        for i, finding in enumerate(self.key_findings, 1):
            conf_tag = "●" if finding.confidence > 0.7 else "◐" if finding.confidence > 0.4 else "○"
            lines.append(f"  {i}. {conf_tag} {finding.description}")
            lines.append(f"     Confidence: {finding.confidence:.0%} | Sources: {len(finding.sources)}")
            if finding.contradicting_evidence:
                lines.append(f"     ⚠ {len(finding.contradicting_evidence)} contradicting evidence(s)")

        lines.extend(["", "─" * 60, "SUPPORTING EVIDENCE", "─" * 60])
        for i, ev in enumerate(self.supporting_evidence[:10], 1):
            lines.append(f"  {i}. {ev[:100]}")

        if self.contradictory_evidence:
            lines.extend(["", "─" * 60, "CONTRADICTORY EVIDENCE", "─" * 60])
            for i, ev in enumerate(self.contradictory_evidence[:5], 1):
                lines.append(f"  {i}. {ev[:100]}")

        if self.unresolved_questions:
            lines.extend(["", "─" * 60, "UNRESOLVED QUESTIONS", "─" * 60])
            for i, q in enumerate(self.unresolved_questions, 1):
                lines.append(f"  {i}. {q}")

        lines.extend(["", "─" * 60, "SOURCES", "─" * 60])
        for i, src in enumerate(self.sources, 1):
            lines.append(f"  {i}. {src}")

        lines.extend(["", "═" * 60])
        return "\n".join(lines)


class EvidenceReportGenerator:
    """
    Generates structured investigation reports from evidence.

    Takes evidence items, findings, and metadata, then compiles
    them into a structured InvestigationReport.
    """

    def __init__(self) -> None:
        self._evidence_items: list[dict[str, Any]] = []
        self._findings: list[Finding] = []

    def reset(self) -> None:
        self._evidence_items.clear()
        self._findings.clear()

    def add_evidence(
        self,
        text: str,
        source: str = "",
        confidence: float = 0.8,
        supports: bool | None = None,
        category: str = "general",
        timestamp: str = "",
    ) -> None:
        """Add an evidence item."""
        self._evidence_items.append({
            "text": text,
            "source": source,
            "confidence": confidence,
            "supports": supports,
            "category": category,
            "timestamp": timestamp,
        })

    def add_finding(
        self,
        description: str,
        confidence: float,
        supporting: list[str] | None = None,
        contradicting: list[str] | None = None,
        sources: list[str] | None = None,
        category: str = "general",
    ) -> None:
        """Add a key finding."""
        self._findings.append(Finding(
            description=description,
            confidence=confidence,
            supporting_evidence=supporting or [],
            contradicting_evidence=contradicting or [],
            sources=sources or [],
            category=category,
        ))

    def generate_report(
        self,
        target: str,
        target_type: str = "person",
    ) -> InvestigationReport:
        """Generate a complete investigation report."""
        t0 = time.perf_counter()

        # Compute statistics
        total = len(self._evidence_items)
        sources = list({e["source"] for e in self._evidence_items if e["source"]})
        independent = len(sources)

        supporting = [e["text"] for e in self._evidence_items if e.get("supports") is True]
        contradicting = [e["text"] for e in self._evidence_items if e.get("supports") is False]
        conflicting = len({e["source"] for e in self._evidence_items if e.get("supports") is False})

        # Timeline
        timestamps = [e["timestamp"] for e in self._evidence_items if e["timestamp"]]
        timeline_start = min(timestamps) if timestamps else ""
        timeline_end = max(timestamps) if timestamps else ""

        # Confidence assessment
        avg_confidence = (
            sum(e["confidence"] for e in self._evidence_items) / max(total, 1)
        )
        if contradicting:
            avg_confidence *= 0.8  # Penalize for contradictions

        confidence_level = self._assess_confidence_level(avg_confidence, len(contradicting))

        # Unresolved questions
        unresolved = []
        categories_seen = {e["category"] for e in self._evidence_items}
        for cat in ["identity", "location", "activities", "timeline"]:
            if cat not in categories_seen:
                unresolved.append(f"No evidence found for: {cat}")

        # Sort findings by confidence
        self._findings.sort(key=lambda f: f.confidence, reverse=True)

        report = InvestigationReport(
            target=target,
            target_type=target_type,
            confidence_level=confidence_level,
            confidence_score=avg_confidence,
            total_evidence=total,
            independent_sources=independent,
            conflicting_sources=conflicting,
            timeline_start=timeline_start,
            timeline_end=timeline_end,
            key_findings=self._findings[:10],
            supporting_evidence=supporting[:20],
            contradictory_evidence=contradicting[:10],
            unresolved_questions=unresolved,
            sources=sources[:20],
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

        return report

    @staticmethod
    def _assess_confidence_level(score: float, contradictions: int) -> str:
        """Assess confidence level from score and contradictions."""
        if contradictions > 3:
            return "contradicted"
        if score >= 0.9:
            return "confirmed"
        if score >= 0.7:
            return "likely"
        if score >= 0.5:
            return "possible"
        if score >= 0.3:
            return "uncertain"
        return "insufficient"
