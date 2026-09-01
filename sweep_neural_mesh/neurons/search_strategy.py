"""
Search Strategy Optimization — uncertainty-driven search planning.

Sweep shouldn't perform every search identically. It determines:
1. What do I know?
2. What don't I know?
3. What search would reduce that uncertainty?

This gives you an investigation loop:

    INPUT → UNDERSTAND → SEARCH → EXTRACT → COMPARE →
    IDENTIFY UNKNOWN → SEARCH AGAIN → CORRELATE →
    VALIDATE → REPORT

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │         SEARCH STRATEGY OPTIMIZER                    │
    │                                                     │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Knowledge State Tracker                      │  │
    │  │  (what is known vs unknown)                   │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Uncertainty Calculator                       │  │
    │  │  (which aspects need more evidence)           │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Search Plan Generator                        │  │
    │  │  (queries, sources, priorities)               │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Progress Evaluator                           │  │
    │  │  (did this search help?)                      │  │
    │  └──────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeAspect:
    """A specific aspect of knowledge about the investigation target."""
    aspect: str
    confidence: float = 0.0
    evidence_count: int = 0
    sources: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aspect": self.aspect,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "sources": self.sources,
        }


@dataclass
class SearchQuery:
    """A generated search query."""
    query: str
    target_aspect: str
    priority: float = 0.5
    source_hints: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "target_aspect": self.target_aspect,
            "priority": self.priority,
            "reasoning": self.reasoning,
        }


@dataclass
class SearchPlan:
    """A complete search plan."""
    queries: list[SearchQuery]
    total_priority: float = 0.0
    estimated_redundancy: float = 0.0
    coverage_gaps: list[str] = field(default_factory=list)
    round_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_count": len(self.queries),
            "total_priority": self.total_priority,
            "coverage_gaps": self.coverage_gaps,
            "round": self.round_number,
            "queries": [q.to_dict() for q in self.queries],
        }


@dataclass
class SearchResult:
    """Result from executing a search."""
    query: SearchQuery
    results: list[str] = field(default_factory=list)
    new_knowledge: dict[str, float] = field(default_factory=dict)
    confidence_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.query,
            "result_count": len(self.results),
            "new_aspects": len(self.new_knowledge),
            "confidence_delta": self.confidence_delta,
        }


@dataclass
class StrategyState:
    """Current state of the search strategy."""
    round_number: int = 0
    total_queries: int = 0
    total_results: int = 0
    knowledge_coverage: float = 0.0
    uncertainty_remaining: float = 1.0
    aspects_known: int = 0
    aspects_unknown: int = 0
    should_continue: bool = True
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_number,
            "total_queries": self.total_queries,
            "knowledge_coverage": self.knowledge_coverage,
            "uncertainty_remaining": self.uncertainty_remaining,
            "should_continue": self.should_continue,
            "reasoning": self.reasoning,
        }


class SearchStrategyOptimizer:
    """
    Uncertainty-driven search strategy optimizer.

    Manages the investigation loop by:
    1. Tracking what's known and unknown
    2. Computing uncertainty per aspect
    3. Generating targeted search queries
    4. Evaluating progress after each round
    5. Deciding when to stop investigating
    """

    # ── Default investigation aspects ────────────────────────
    _DEFAULT_ASPECTS = [
        "identity",
        "location",
        "affiliation",
        "timeline",
        "activities",
        "associates",
        "online_presence",
        "physical_description",
        "background",
        "claims",
    ]

    # ── Search source priorities ─────────────────────────────
    _SOURCE_PRIORITIES: dict[str, float] = {
        "official_documents": 0.95,
        "news_articles": 0.85,
        "academic_papers": 0.90,
        "government_records": 0.92,
        "social_media": 0.60,
        "forums": 0.40,
        "blogs": 0.50,
        "databases": 0.80,
        "web_search": 0.70,
    }

    def __init__(
        self,
        aspects: list[str] | None = None,
        confidence_threshold: float = 0.80,
        max_rounds: int = 10,
        min_improvement: float = 0.05,
    ) -> None:
        self._aspects = aspects or list(self._DEFAULT_ASPECTS)
        self._confidence_threshold = confidence_threshold
        self._max_rounds = max_rounds
        self._min_improvement = min_improvement

        self._knowledge: dict[str, KnowledgeAspect] = {}
        self._round_history: list[StrategyState] = []
        self._total_queries_ever = 0

        # Initialize all aspects
        for aspect in self._aspects:
            self._knowledge[aspect] = KnowledgeAspect(aspect=aspect)

    def update_knowledge(
        self,
        aspect: str,
        evidence: str,
        source: str = "",
        confidence: float = 0.7,
    ) -> None:
        """Update knowledge about a specific aspect."""
        if aspect not in self._knowledge:
            self._knowledge[aspect] = KnowledgeAspect(aspect=aspect)

        ka = self._knowledge[aspect]
        ka.evidence_count += 1
        ka.confidence = max(ka.confidence, confidence)
        if source and source not in ka.sources:
            ka.sources.append(source)
        ka.last_updated = time.time()

    def compute_uncertainty(self) -> dict[str, float]:
        """Compute uncertainty per aspect (1 - confidence)."""
        return {
            aspect: 1.0 - ka.confidence
            for aspect, ka in self._knowledge.items()
        }

    def get_coverage_gaps(self) -> list[str]:
        """Identify aspects with low confidence."""
        gaps = []
        for aspect, ka in self._knowledge.items():
            if ka.confidence < self._confidence_threshold:
                gaps.append(aspect)
        return gaps

    def generate_search_plan(self) -> SearchPlan:
        """Generate an optimized search plan based on current knowledge gaps."""
        gaps = self.get_coverage_gaps()
        uncertainty = self.compute_uncertainty()

        queries: list[SearchQuery] = []

        # Sort gaps by uncertainty (most uncertain first)
        sorted_gaps = sorted(gaps, key=lambda a: uncertainty.get(a, 1.0), reverse=True)

        for gap in sorted_gaps[:5]:  # Top 5 gaps
            ka = self._knowledge[gap]
            unc = uncertainty.get(gap, 1.0)

            # Generate query based on aspect type
            query_text = self._generate_query_for_aspect(gap, ka)
            if query_text:
                queries.append(SearchQuery(
                    query=query_text,
                    target_aspect=gap,
                    priority=unc,
                    reasoning=f"High uncertainty ({unc:.0%}) on '{gap}'",
                ))

        total_priority = sum(q.priority for q in queries)

        return SearchPlan(
            queries=queries,
            total_priority=total_priority,
            coverage_gaps=gaps,
            round_number=len(self._round_history),
        )

    def _generate_query_for_aspect(
        self, aspect: str, ka: KnowledgeAspect
    ) -> str:
        """Generate a search query for a specific aspect."""
        templates: dict[str, list[str]] = {
            "identity": ["{target} identity information", "who is {target}"],
            "location": ["{target} location", "where is {target} based"],
            "affiliation": ["{target} organization", "{target} company", "{target} employer"],
            "timeline": ["{target} history", "{target} timeline", "{target} career"],
            "activities": ["{target} activities", "{target} projects", "{target} work"],
            "associates": ["{target} associates", "{target} colleagues", "{target} contacts"],
            "online_presence": ["{target} social media", "{target} website", "{target} online"],
            "physical_description": ["{target} appearance", "{target} description"],
            "background": ["{target} background", "{target} education", "{target} experience"],
            "claims": ["{target} claims", "{target} statements", "{target} said"],
        }

        aspect_templates = templates.get(aspect, [f"{aspect} information about {{target}}"])
        # Use first template (can be improved with ML)
        return aspect_templates[0].replace("{target}", "target")

    def record_search_results(self, results: list[SearchResult]) -> StrategyState:
        """Record search results and update knowledge state."""
        for result in results:
            self._total_queries_ever += 1
            for aspect, conf in result.new_knowledge.items():
                self.update_knowledge(
                    aspect=aspect,
                    evidence=str(result.results[:1]) if result.results else "",
                    confidence=conf,
                )

        state = self._evaluate_state()
        self._round_history.append(state)
        return state

    def _evaluate_state(self) -> StrategyState:
        """Evaluate current knowledge state and decide whether to continue."""
        uncertainty = self.compute_uncertainty()
        gaps = self.get_coverage_gaps()

        avg_confidence = sum(
            ka.confidence for ka in self._knowledge.values()
        ) / max(len(self._knowledge), 1)
        coverage = 1.0 - (len(gaps) / max(len(self._knowledge), 1))
        avg_uncertainty = sum(uncertainty.values()) / max(len(uncertainty), 1)

        # Decision logic
        should_continue = True
        reasoning = ""

        if len(self._round_history) >= self._max_rounds:
            should_continue = False
            reasoning = f"Reached max rounds ({self._max_rounds})"
        elif avg_confidence >= self._confidence_threshold:
            should_continue = False
            reasoning = f"Average confidence ({avg_confidence:.0%}) exceeds threshold"
        elif len(gaps) == 0:
            should_continue = False
            reasoning = "All aspects covered"
        elif self._round_history:
            last = self._round_history[-1]
            improvement = coverage - last.knowledge_coverage
            if improvement < self._min_improvement and len(self._round_history) > 3:
                should_continue = False
                reasoning = f"Insufficient improvement ({improvement:.2%})"
        else:
            reasoning = f"{len(gaps)} aspects need more evidence"

        return StrategyState(
            round_number=len(self._round_history),
            total_queries=self._total_queries_ever,
            total_results=0,
            knowledge_coverage=coverage,
            uncertainty_remaining=avg_uncertainty,
            aspects_known=len(self._aspects) - len(gaps),
            aspects_unknown=len(gaps),
            should_continue=should_continue,
            reasoning=reasoning,
        )

    def get_state(self) -> StrategyState:
        """Get current strategy state."""
        return self._evaluate_state()

    def get_full_report(self) -> dict[str, Any]:
        """Get comprehensive strategy report."""
        state = self._evaluate_state()
        uncertainty = self.compute_uncertainty()

        return {
            "state": state.to_dict(),
            "knowledge": {a: ka.to_dict() for a, ka in self._knowledge.items()},
            "uncertainty": uncertainty,
            "coverage_gaps": self.get_coverage_gaps(),
            "rounds_completed": len(self._round_history),
            "round_history": [r.to_dict() for r in self._round_history[-5:]],
        }
