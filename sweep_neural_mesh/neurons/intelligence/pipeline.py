"""
IntelligencePipeline — orchestrates the full gather → organize → analyze flow.

This is the main entry point for Sweep's intelligence system.
It coordinates:
  1. Gathering intelligence from multiple sources.
  2. Organizing into structured knowledge.
  3. Analyzing for insights and patterns.
  4. Storing for future retrieval.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .gatherer import IntelligenceGatherer, GatheredIntel, IntelSource
from .organizer import IntelligenceOrganizer, OrganizedIntel
from .analyzer import IntelligenceAnalyzer, AnalyzedIntel
from .store import IntelligenceStore


@dataclass
class IntelligenceReport:
    """Complete intelligence report from a pipeline run."""
    query: str
    gathered: list[GatheredIntel]
    organized: OrganizedIntel
    analyzed: AnalyzedIntel
    stored_count: int
    total_latency_ms: float
    timestamp: float = field(default_factory=time.time)

    @property
    def summary(self) -> str:
        """One-line summary."""
        return (
            f"Intel: {len(self.gathered)} items gathered, "
            f"{len(self.organized.clusters)} topics, "
            f"{self.analyzed.insight_count} insights, "
            f"confidence={self.analyzed.overall_confidence:.0%}, "
            f"completeness={self.analyzed.completeness_score:.0%}"
        )


class IntelligencePipeline:
    """Orchestrates the full intelligence gathering and analysis pipeline.

    Usage::

        pipeline = IntelligencePipeline()
        report = pipeline.run(
            query="What is quantum computing?",
            documents=["Quantum computing uses qubits..."],
            evidence=["Quantum computers solve certain problems faster..."],
        )
        print(report.summary)
        print(report.analyzed.actionable_summary)
    """

    def __init__(self, store_path: str | None = None) -> None:
        self._gatherer = IntelligenceGatherer()
        self._organizer = IntelligenceOrganizer()
        self._analyzer = IntelligenceAnalyzer()
        self._store = IntelligenceStore()
        self._web_scraper = None  # Lazy init

        if store_path:
            try:
                self._store.load(store_path)
            except Exception:
                pass
        self._store_path = store_path

    def run(
        self,
        query: str,
        documents: list[str] | None = None,
        evidence: list[str] | None = None,
        world_knowledge: Any = None,
        live_retriever: Any = None,
        max_items: int = 20,
        store_results: bool = True,
        enable_web: bool = False,
    ) -> IntelligenceReport:
        """Run the full intelligence pipeline.

        Args:
            query:            What to gather intelligence about.
            documents:        User-provided documents to analyze.
            evidence:         Evidence from the reasoning pipeline.
            world_knowledge:  WorldKnowledge instance for fact-checking.
            live_retriever:   LiveKnowledgeRetriever for API queries.
            max_items:        Maximum items to gather.
            store_results:    Whether to store results for future retrieval.

        Returns:
            IntelligenceReport with gathered, organized, and analyzed intel.
        """
        t0 = time.perf_counter()

        # 1. Gather
        web_scraper = None
        if enable_web:
            web_scraper = self._ensure_web_scraper()

        gathered = self._gatherer.gather(
            query=query,
            documents=documents,
            evidence=evidence,
            world_knowledge=world_knowledge,
            live_retriever=live_retriever,
            max_results=max_items,
            web_scraper=web_scraper,
        )

        # 2. Organize
        organized = self._organizer.organize(query=query, intel=gathered)

        # 3. Analyze
        analyzed = self._analyzer.analyze(query=query, organized=organized)

        # 4. Store
        stored_count = 0
        if store_results:
            stored_count = self._store.store_intel(query, organized, analyzed)
            if self._store_path:
                try:
                    self._store.save(self._store_path)
                except Exception:
                    pass

        total_lat = (time.perf_counter() - t0) * 1000

        return IntelligenceReport(
            query=query,
            gathered=gathered,
            organized=organized,
            analyzed=analyzed,
            stored_count=stored_count,
            total_latency_ms=total_lat,
        )

    def gather_only(
        self,
        query: str,
        documents: list[str] | None = None,
        evidence: list[str] | None = None,
        world_knowledge: Any = None,
        live_retriever: Any = None,
    ) -> list[GatheredIntel]:
        """Gather intelligence without organizing/analyzing."""
        return self._gatherer.gather(
            query=query, documents=documents, evidence=evidence,
            world_knowledge=world_knowledge, live_retriever=live_retriever,
        )

    def _ensure_web_scraper(self):
        """Lazily initialize the web scraper."""
        if self._web_scraper is None:
            try:
                from ..web_scraper import WebScraper
                self._web_scraper = WebScraper()
            except Exception:
                return None
        return self._web_scraper

    def search_stored(self, query: str, max_results: int = 10):
        """Search previously stored intelligence."""
        return self._store.search(query, max_results)

    def get_store_stats(self) -> dict[str, Any]:
        """Get statistics about the intelligence store."""
        return self._store.stats()

    def get_knowledge_graph(self) -> dict[str, list[dict[str, str]]]:
        """Get the accumulated knowledge graph."""
        return self._organizer.get_knowledge_graph()
