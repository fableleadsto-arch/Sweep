"""
IntelligenceAnalyzer — reasons about organized intelligence and extracts insights.

Responsibilities:
  - Identify patterns and trends across intelligence clusters.
  - Generate insights and inferences.
  - Assess information completeness and gaps.
  - Produce actionable intelligence summaries.
  - Detect key themes and priorities.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any
from collections import Counter

from .organizer import OrganizedIntel, IntelCluster
from .gatherer import IntelSource


@dataclass
class Insight:
    """A single insight derived from intelligence analysis."""
    type: str          # "pattern", "gap", "contradiction", "trend", "inference", "priority"
    description: str
    confidence: float
    supporting_items: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalyzedIntel:
    """The output of intelligence analysis."""
    query: str
    insights: list[Insight]
    key_findings: list[str]
    information_gaps: list[str]
    completeness_score: float  # 0.0-1.0
    overall_confidence: float
    actionable_summary: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def insight_count(self) -> int:
        return len(self.insights)

    @property
    def high_priority_insights(self) -> list[Insight]:
        return [i for i in self.insights if i.confidence >= 0.8]


class IntelligenceAnalyzer:
    """Analyzes organized intelligence and extracts insights.

    Usage::

        analyzer = IntelligenceAnalyzer()
        analyzed = analyzer.analyze(query="quantum computing", organized=org)
        for insight in analyzed.insights:
            print(f"[{insight.type}] {insight.description}")
    """

    def analyze(
        self,
        query: str,
        organized: OrganizedIntel,
    ) -> AnalyzedIntel:
        """Analyze organized intelligence and extract insights.

        Steps:
          1. Identify patterns across clusters.
          2. Detect information gaps.
          3. Generate inferences.
          4. Assess completeness.
          5. Produce actionable summary.
        """
        t0 = time.perf_counter()

        insights: list[Insight] = []

        # 1. Pattern detection
        insights.extend(self._detect_patterns(organized))

        # 2. Gap detection
        gaps = self._detect_gaps(query, organized)
        insights.extend(gaps["insights"])

        # 3. Inference generation
        insights.extend(self._generate_inferences(organized))

        # 4. Contradiction analysis
        insights.extend(self._analyze_contradictions(organized))

        # 5. Priority assessment
        insights.extend(self._assess_priorities(organized))

        # 6. Key findings
        key_findings = self._extract_key_findings(organized)

        # 7. Completeness
        completeness = self._assess_completeness(query, organized)

        # 8. Overall confidence
        overall_conf = self._compute_overall_confidence(organized, insights)

        # 9. Actionable summary
        summary = self._generate_actionable_summary(
            query, organized, insights, key_findings,
        )

        lat = (time.perf_counter() - t0) * 1000

        return AnalyzedIntel(
            query=query,
            insights=insights,
            key_findings=key_findings,
            information_gaps=gaps["gaps"],
            completeness_score=completeness,
            overall_confidence=overall_conf,
            actionable_summary=summary,
            latency_ms=lat,
        )

    # ── Pattern Detection ───────────────────────────────────

    def _detect_patterns(self, organized: OrganizedIntel) -> list[Insight]:
        """Detect patterns across intelligence clusters."""
        insights = []

        # Source diversity pattern
        sources = set()
        for cluster in organized.clusters:
            for item in cluster.items:
                sources.add(item.source)
        if len(sources) >= 3:
            insights.append(Insight(
                type="pattern",
                description=f"Information gathered from {len(sources)} different sources: {', '.join(s.value for s in sources)}",
                confidence=0.85,
                supporting_items=organized.total_items,
            ))

        # Topic concentration
        topic_counts = Counter(c.topic for c in organized.clusters)
        if topic_counts:
            main_topic, count = topic_counts.most_common(1)[0]
            if count > 3:
                insights.append(Insight(
                    type="pattern",
                    description=f"Strong focus on '{main_topic}' with {count} related items",
                    confidence=0.80,
                    supporting_items=count,
                ))

        # Confidence gradient
        high_conf = sum(1 for c in organized.clusters for i in c.items if i.confidence >= 0.8)
        total = organized.total_items
        if total > 0 and high_conf / total > 0.7:
            insights.append(Insight(
                type="pattern",
                description=f"High confidence across {high_conf}/{total} items ({high_conf/total:.0%})",
                confidence=0.90,
                supporting_items=high_conf,
            ))

        return insights

    # ── Gap Detection ───────────────────────────────────────

    def _detect_gaps(
        self, query: str, organized: OrganizedIntel,
    ) -> dict[str, Any]:
        """Detect information gaps."""
        gaps = []
        insights = []

        # Check for missing topics
        expected_topics = self._get_expected_topics(query)
        covered_topics = set(organized.topics)
        missing = expected_topics - covered_topics

        for topic in missing:
            gaps.append(f"Missing information about: {topic}")
            insights.append(Insight(
                type="gap",
                description=f"No information found about '{topic}'",
                confidence=0.75,
                supporting_items=0,
                metadata={"topic": topic},
            ))

        # Check for low-coverage areas
        for cluster in organized.clusters:
            if cluster.size < 2:
                gaps.append(f"Limited information about: {cluster.topic}")
                insights.append(Insight(
                    type="gap",
                    description=f"Only {cluster.size} item(s) for '{cluster.topic}' — may be incomplete",
                    confidence=0.70,
                    supporting_items=cluster.size,
                    metadata={"topic": cluster.topic, "count": cluster.size},
                ))

        # Check for contradictions
        if organized.contradictions:
            gaps.append(f"{len(organized.contradictions)} contradiction(s) need resolution")

        return {"gaps": gaps, "insights": insights}

    # ── Inference Generation ────────────────────────────────

    def _generate_inferences(self, organized: OrganizedIntel) -> list[Insight]:
        """Generate inferences from the intelligence."""
        insights = []

        # Cross-cluster inferences
        if len(organized.clusters) >= 2:
            topics = [c.topic for c in organized.clusters]
            if len(set(topics)) >= 2:
                insights.append(Insight(
                    type="inference",
                    description=f"Multi-domain coverage ({', '.join(list(set(topics))[:3])}) suggests comprehensive understanding",
                    confidence=0.75,
                    supporting_items=organized.total_items,
                ))

        # Relation-based inferences
        total_relations = sum(len(c.relations) for c in organized.clusters)
        if total_relations >= 5:
            insights.append(Insight(
                type="inference",
                description=f"{total_relations} relationships identified — indicates structured knowledge",
                confidence=0.80,
                supporting_items=total_relations,
            ))

        # Entity diversity
        all_entities = set()
        for cluster in organized.clusters:
            for item in cluster.items:
                for e in item.entities:
                    all_entities.add(e["name"])
        if len(all_entities) >= 5:
            insights.append(Insight(
                type="inference",
                description=f"{len(all_entities)} distinct entities identified — rich entity landscape",
                confidence=0.80,
                supporting_items=len(all_entities),
            ))

        return insights

    # ── Contradiction Analysis ──────────────────────────────

    def _analyze_contradictions(self, organized: OrganizedIntel) -> list[Insight]:
        """Analyze contradictions."""
        insights = []

        if organized.contradictions:
            insights.append(Insight(
                type="contradiction",
                description=f"{len(organized.contradictions)} contradiction(s) detected — requires careful evaluation",
                confidence=0.90,
                supporting_items=len(organized.contradictions),
                metadata={"contradictions": organized.contradictions[:5]},
            ))

            # High-confidence contradictions are more concerning
            high_conf_contradictions = [
                c for c in organized.contradictions if c[2].startswith("Conflicting")
            ]
            if high_conf_contradictions:
                insights.append(Insight(
                    type="contradiction",
                    description=f"{len(high_conf_contradictions)} factual conflict(s) found in structured data",
                    confidence=0.85,
                    supporting_items=len(high_conf_contradictions),
                ))

        return insights

    # ── Priority Assessment ─────────────────────────────────

    def _assess_priorities(self, organized: OrganizedIntel) -> list[Insight]:
        """Assess priority areas."""
        insights = []

        # Most confident cluster
        if organized.clusters:
            best = max(organized.clusters, key=lambda c: c.avg_confidence)
            insights.append(Insight(
                type="priority",
                description=f"Highest confidence area: '{best.topic}' ({best.avg_confidence:.0%} avg confidence, {best.size} items)",
                confidence=best.avg_confidence,
                supporting_items=best.size,
            ))

        # Largest cluster
        if organized.clusters:
            largest = max(organized.clusters, key=lambda c: c.size)
            if largest.size > 3:
                insights.append(Insight(
                    type="priority",
                    description=f"Most documented area: '{largest.topic}' ({largest.size} items)",
                    confidence=0.80,
                    supporting_items=largest.size,
                ))

        return insights

    # ── Key Findings ────────────────────────────────────────

    def _extract_key_findings(self, organized: OrganizedIntel) -> list[str]:
        """Extract the most important findings."""
        findings = []

        # Top high-confidence items
        for cluster in organized.clusters:
            top_items = sorted(cluster.items, key=lambda x: -x.confidence)[:2]
            for item in top_items:
                if item.confidence >= 0.8:
                    findings.append(f"[{cluster.topic}] {item.content[:150]}")

        # Relations as findings
        for cluster in organized.clusters:
            for rel in cluster.relations[:2]:
                findings.append(f"{rel['subject']} {rel['predicate']} {rel['object']}")

        return findings[:10]

    # ── Completeness Assessment ─────────────────────────────

    def _assess_completeness(
        self, query: str, organized: OrganizedIntel,
    ) -> float:
        """Assess how complete the intelligence coverage is."""
        score = 0.0

        # Item count contribution
        if organized.total_items >= 10:
            score += 0.3
        elif organized.total_items >= 5:
            score += 0.2
        elif organized.total_items >= 2:
            score += 0.1

        # Topic coverage
        expected = self._get_expected_topics(query)
        covered = set(organized.topics)
        if expected:
            coverage = len(covered & expected) / len(expected)
            score += 0.3 * coverage

        # Source diversity
        sources = set()
        for c in organized.clusters:
            for i in c.items:
                sources.add(i.source)
        if len(sources) >= 3:
            score += 0.2
        elif len(sources) >= 2:
            score += 0.1

        # Confidence
        avg_conf = sum(
            i.confidence for c in organized.clusters for i in c.items
        ) / max(organized.total_items, 1)
        score += 0.2 * avg_conf

        # Penalty for contradictions
        if organized.contradictions:
            score -= 0.1 * min(len(organized.contradictions), 3)

        return max(0.0, min(1.0, score))

    # ── Overall Confidence ──────────────────────────────────

    def _compute_overall_confidence(
        self, organized: OrganizedIntel, insights: list[Insight],
    ) -> float:
        """Compute overall confidence in the analysis."""
        if not organized.clusters:
            return 0.0

        # Average item confidence
        item_confs = [
            i.confidence for c in organized.clusters for i in c.items
        ]
        avg_item = sum(item_confs) / len(item_confs) if item_confs else 0.0

        # Contradiction penalty
        contra_penalty = min(0.2, len(organized.contradictions) * 0.05)

        # Insight confidence boost
        insight_confs = [i.confidence for i in insights]
        avg_insight = sum(insight_confs) / len(insight_confs) if insight_confs else 0.5

        return max(0.0, min(1.0, 0.4 * avg_item + 0.3 * avg_insight + 0.3 * 0.8 - contra_penalty))

    # ── Actionable Summary ──────────────────────────────────

    def _generate_actionable_summary(
        self,
        query: str,
        organized: OrganizedIntel,
        insights: list[Insight],
        key_findings: list[str],
    ) -> str:
        """Generate an actionable summary."""
        parts = []

        parts.append(f"Intelligence analysis for: {query}")
        parts.append(f"Coverage: {organized.total_items} items across {len(organized.clusters)} topics")
        parts.append("")

        # Key findings
        if key_findings:
            parts.append("Key Findings:")
            for f in key_findings[:5]:
                parts.append(f"  • {f}")
            parts.append("")

        # Insights
        high_insights = [i for i in insights if i.confidence >= 0.7]
        if high_insights:
            parts.append("Insights:")
            for i in high_insights[:5]:
                parts.append(f"  [{i.type}] {i.description}")
            parts.append("")

        # Contradictions
        if organized.contradictions:
            parts.append(f"⚠ {len(organized.contradictions)} contradiction(s) detected")
            parts.append("")

        # Gaps
        gap_insights = [i for i in insights if i.type == "gap"]
        if gap_insights:
            parts.append("Information Gaps:")
            for g in gap_insights[:3]:
                parts.append(f"  • {g.description}")

        return "\n".join(parts)

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _get_expected_topics(query: str) -> set[str]:
        """Get expected topics for a query."""
        q = query.lower()
        topics = set()

        topic_map = {
            "physics": ["energy", "force", "particle", "quantum", "wave"],
            "biology": ["cell", "organism", "species", "dna", "evolution"],
            "chemistry": ["molecule", "element", "reaction", "compound"],
            "history": ["war", "revolution", "century", "ancient"],
            "geography": ["country", "city", "river", "mountain"],
            "technology": ["computer", "software", "data", "network"],
        }

        for topic, keywords in topic_map.items():
            if any(kw in q for kw in keywords):
                topics.add(topic)

        if not topics:
            topics.add("general")

        return topics
