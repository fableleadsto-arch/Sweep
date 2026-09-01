"""
WebResearcher — conducts multi-query, multi-source web research.

Unlike a simple scraper, the researcher:
  1. Generates multiple search queries from a topic
  2. Fetches from multiple sources (Wikipedia, arXiv, OpenAlex, web)
  3. Deduplicates and ranks findings
  4. Extracts key facts, entities, and relationships
  5. Produces a structured ResearchReport

Usage::

    researcher = WebResearcher()
    report = researcher.research("quantum computing applications")
    for finding in report.findings:
        print(f"[{finding.source}] {finding.title}")
        print(f"  {finding.text[:100]}")
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .scraper import WebScraper, ScrapedPage


@dataclass(frozen=True, slots=True)
class ResearchFinding:
    """A single finding from web research."""
    title: str
    text: str
    source: str
    url: str = ""
    confidence: float = 0.8
    entities: tuple[str, ...] = ()
    key_facts: tuple[str, ...] = ()
    relevance: float = 0.0
    latency_ms: float = 0.0


@dataclass
class ResearchReport:
    """Complete report from a web research session."""
    query: str
    findings: list[ResearchFinding]
    sources_queried: int = 0
    total_latency_ms: float = 0.0
    key_facts: list[str] = field(default_factory=list)
    entities_found: list[str] = field(default_factory=list)
    summary: str = ""

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    @property
    def source_breakdown(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.source] = counts.get(f.source, 0) + 1
        return counts

    @property
    def best_finding(self) -> ResearchFinding | None:
        if not self.findings:
            return None
        return max(self.findings, key=lambda f: f.confidence * f.relevance)


class WebResearcher:
    """Conducts multi-query, multi-source web research.

    Usage::

        researcher = WebResearcher()
        report = researcher.research("quantum computing applications")
        print(report.summary)
    """

    def __init__(self, scraper: WebScraper | None = None) -> None:
        self._scraper = scraper or WebScraper()
        self._all_findings: list[ResearchFinding] = []

    def research(
        self,
        query: str,
        max_results: int = 10,
        sources: list[str] | None = None,
        generate_queries: bool = True,
    ) -> ResearchReport:
        """Conduct web research on a topic.

        Args:
            query:            Research topic or question.
            max_results:      Maximum findings to return.
            sources:          Specific sources to use (None = all).
            generate_queries: Whether to generate multiple search queries.

        Returns:
            ResearchReport with findings, key facts, and summary.
        """
        t0 = time.perf_counter()
        sources = sources or ["wikipedia", "arxiv", "openalex"]

        # 1. Generate search queries
        queries = [query]
        if generate_queries:
            queries.extend(self._generate_queries(query))

        # 2. Fetch from each source for each query
        all_pages: list[ScrapedPage] = []
        for q in queries:
            for source in sources:
                pages = self._fetch_source(q, source, max_results=3)
                all_pages.extend(pages)

        # 3. Extract findings from pages
        findings: list[ResearchFinding] = []
        for page in all_pages:
            finding = self._page_to_finding(page, query)
            if finding:
                findings.append(finding)

        # 4. Deduplicate
        findings = self._deduplicate(findings)

        # 5. Rank by relevance
        findings = self._rank_by_relevance(findings, query)

        # 6. Extract key facts and entities
        all_text = " ".join(f.text for f in findings)
        key_facts = self._extract_key_facts(all_text)
        entities = self._extract_entities(all_text)

        # 7. Generate summary
        summary = self._generate_summary(query, findings, key_facts)

        # 8. Limit
        findings = findings[:max_results]

        # Track
        self._all_findings.extend(findings)

        total_lat = (time.perf_counter() - t0) * 1000

        return ResearchReport(
            query=query,
            findings=findings,
            sources_queried=len(sources) * len(queries),
            total_latency_ms=total_lat,
            key_facts=key_facts[:20],
            entities_found=entities[:30],
            summary=summary,
        )

    def research_question(self, question: str) -> ResearchReport:
        """Research a specific question and synthesize an answer.

        More opinionated than research(): generates more queries
        and produces a more focused report.
        """
        # Generate question-specific queries
        queries = self._generate_question_queries(question)

        # Fetch from all sources
        all_pages: list[ScrapedPage] = []
        for q in queries:
            all_pages.extend(self._scraper.search_and_fetch(q, max_results=3))

        # Extract findings
        findings = []
        for page in all_pages:
            finding = self._page_to_finding(page, question)
            if finding:
                findings.append(finding)

        findings = self._deduplicate(findings)
        findings = self._rank_by_relevance(findings, question)

        # Synthesize answer
        all_text = " ".join(f.text for f in findings[:5])
        answer_parts = self._synthesize_answer(question, findings, all_text)
        key_facts = self._extract_key_facts(all_text)
        entities = self._extract_entities(all_text)

        summary = f"Research on: {question}\n"
        summary += f"Sources consulted: {len(set(f.source for f in findings))}\n"
        summary += f"Key findings: {len(findings)}\n\n"
        summary += "\n".join(answer_parts[:5])

        return ResearchReport(
            query=question,
            findings=findings[:10],
            sources_queried=len(queries),
            total_latency_ms=0,  # Will be set by caller if needed
            key_facts=key_facts[:20],
            entities_found=entities[:30],
            summary=summary,
        )

    def get_all_findings(self) -> list[ResearchFinding]:
        """Return all findings across all research sessions."""
        return list(self._all_findings)

    def clear(self) -> None:
        """Clear all accumulated findings."""
        self._all_findings.clear()

    # ════════════════════════════════════════════════════════════
    # QUERY GENERATION
    # ════════════════════════════════════════════════════════════

    def _generate_queries(self, topic: str) -> list[str]:
        """Generate multiple search queries from a topic.

        Creates diverse queries to maximize coverage:
        - Overview/explanation queries
        - Definition queries
        - Recent developments
        - Academic/research queries
        - Specific aspect queries
        """
        queries = []
        topic_clean = topic.strip().rstrip("?.!")

        # Core variations
        queries.append(topic_clean)
        queries.append(f"{topic_clean} overview")
        queries.append(f"what is {topic_clean}")

        # Definition-focused
        queries.append(f"{topic_clean} definition explained")

        # Recent/updates
        queries.append(f"{topic_clean} latest developments")

        # Academic
        queries.append(f"{topic_clean} research")

        # How/why variants
        queries.append(f"how does {topic_clean} work")
        queries.append(f"why is {topic_clean} important")

        # Application/use cases
        queries.append(f"{topic_clean} applications use cases")

        return queries[:6]

    def _generate_question_queries(self, question: str) -> list[str]:
        """Generate queries specifically for answering a question.

        Parses the question structure to generate targeted queries.
        """
        q = question.strip().rstrip("?.!")
        queries = []

        # Direct topic extraction
        # "What is X?" -> "X"
        what_match = re.search(
            r"(?:what|who|where|when|why|how)\s+(?:is|are|was|were|does|do|did|has|have|had)\s+(.+)",
            q, re.IGNORECASE,
        )
        if what_match:
            topic = what_match.group(1).strip()
            queries.append(topic)
            queries.append(f"{topic} explained")
            queries.append(f"{topic} definition")
            queries.append(f"{topic} facts")
            queries.append(f"how does {topic} work")

        # "X vs Y" pattern
        vs_match = re.search(r"(.+?)\s+vs\.?\s+(.+)", q, re.IGNORECASE)
        if vs_match:
            a = vs_match.group(1).strip()
            b = vs_match.group(2).strip()
            queries.append(a)
            queries.append(b)
            queries.append(f"{a} vs {b} comparison")
            queries.append(f"difference between {a} and {b}")

        # "Why does X happen?" pattern
        why_match = re.search(r"why\s+(?:does|do|did|is|are|was|were)\s+(.+)", q, re.IGNORECASE)
        if why_match:
            topic = why_match.group(1).strip()
            queries.append(f"{topic} cause reason")
            queries.append(f"why {topic}")

        # "When was X?" pattern
        when_match = re.search(r"when\s+(?:was|were|did|is|are)\s+(.+)", q, re.IGNORECASE)
        if when_match:
            topic = when_match.group(1).strip()
            queries.append(f"{topic} date year history")

        # "Who invented/created X?" pattern
        who_match = re.search(r"who\s+(?:invented|created|discovered|founded|wrote|composed)\s+(.+)", q, re.IGNORECASE)
        if who_match:
            topic = who_match.group(1).strip()
            queries.append(f"{topic} inventor creator founder")
            queries.append(f"history of {topic}")

        # "How many/how much" pattern
        how_match = re.search(r"how\s+(?:many|much|far|long|old|fast)\s+(.+)", q, re.IGNORECASE)
        if how_match:
            topic = how_match.group(1).strip()
            queries.append(f"{topic} statistics numbers")

        # Always include the original
        queries.insert(0, q)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for qry in queries:
            key = qry.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(qry)

        return unique[:7]

    # ════════════════════════════════════════════════════════════
    # SOURCE FETCHING
    # ════════════════════════════════════════════════════════════

    def _fetch_source(self, query: str, source: str, max_results: int = 3) -> list[ScrapedPage]:
        """Fetch results from a specific source."""
        pages: list[ScrapedPage] = []

        if source == "wikipedia":
            titles = self._scraper._search_wikipedia(query, max_results)
            for title in titles:
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                page = self._scraper.fetch(url)
                if page.success:
                    pages.append(page)

        elif source == "arxiv":
            results = self._scraper._search_arxiv(query, max_results)
            for r in results:
                page = self._scraper.fetch(r["url"])
                if page.success:
                    pages.append(page)

        elif source == "openalex":
            pages = self._scraper._search_openalex(query, max_results)

        return pages

    # ════════════════════════════════════════════════════════════
    # FINDING PROCESSING
    # ════════════════════════════════════════════════════════════

    def _page_to_finding(self, page: ScrapedPage, query: str) -> ResearchFinding | None:
        """Convert a ScrapedPage to a ResearchFinding."""
        if not page.success or not page.text:
            return None

        # Extract entities from the page text
        entities = self._extract_entities(page.text[:2000])

        # Extract key facts
        key_facts = self._extract_key_facts(page.text[:2000])

        return ResearchFinding(
            title=page.title,
            text=page.text,
            source=page.source,
            url=page.url,
            confidence=page.confidence,
            entities=tuple(entities[:10]),
            key_facts=tuple(key_facts[:5]),
            latency_ms=page.latency_ms,
        )

    def _deduplicate(self, findings: list[ResearchFinding]) -> list[ResearchFinding]:
        """Remove duplicate and overlapping findings."""
        if not findings:
            return []

        unique: list[ResearchFinding] = []
        seen_titles: set[str] = set()
        seen_text_hashes: set[str] = set()

        for f in findings:
            # Skip duplicate titles
            title_key = f.title.lower().strip()
            if title_key in seen_titles:
                continue

            # Skip near-duplicate text (using first 100 chars as proxy)
            text_key = f.text[:100].lower().strip()
            text_hash = hash(text_key)
            if text_hash in seen_text_hashes:
                continue

            seen_titles.add(title_key)
            seen_text_hashes.add(text_hash)
            unique.append(f)

        return unique

    def _rank_by_relevance(
        self, findings: list[ResearchFinding], query: str,
    ) -> list[ResearchFinding]:
        """Rank findings by relevance to the query."""
        query_words = set(re.findall(r"\b[a-z]{3,}\b", query.lower()))

        def relevance(f: ResearchFinding) -> float:
            text_words = set(re.findall(r"\b[a-z]{3,}\b", f.text[:500].lower()))
            if not query_words or not text_words:
                return 0.0
            overlap = len(query_words & text_words)
            base = overlap / len(query_words)
            # Boost by confidence
            return base * 0.7 + f.confidence * 0.3

        for f in findings:
            object.__setattr__(f, "relevance", relevance(f))

        findings.sort(key=lambda x: x.relevance, reverse=True)
        return findings

    # ════════════════════════════════════════════════════════════
    # EXTRACTION
    # ════════════════════════════════════════════════════════════

    def _extract_key_facts(self, text: str) -> list[str]:
        """Extract key factual statements from text."""
        facts = []
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue

            # Prioritize sentences with numbers (likely factual)
            if re.search(r"\b\d{4}\b", sentence):  # Has a year
                facts.append(sentence)
            elif re.search(r"\b\d[\d,.]+\b", sentence):  # Has a number
                facts.append(sentence)
            elif re.search(r"\b\d+%\b", sentence):  # Has a percentage
                facts.append(sentence)
            elif any(w in sentence.lower() for w in [
                "according to", "research shows", "study found",
                "was discovered", "was invented", "is defined as",
                "first", "largest", "smallest", "most",
            ]):
                facts.append(sentence)

        # Deduplicate and limit
        seen = set()
        unique_facts = []
        for fact in facts:
            key = fact[:50].lower()
            if key not in seen:
                seen.add(key)
                unique_facts.append(fact)
        return unique_facts[:20]

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text."""
        entities = []

        # Proper nouns (capitalized words/phrases)
        proper_nouns = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
        for noun in proper_nouns:
            if len(noun) > 2 and noun.lower() not in (
                "the", "this", "that", "which", "where", "when",
                "what", "how", "who", "were", "was", "has", "had",
            ):
                entities.append(noun)

        # Organizations (inc, corp, ltd, university, etc.)
        orgs = re.findall(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Ltd|LLC|University|Institute|Laboratory|Foundation))\b",
            text,
        )
        entities.extend(orgs)

        # Deduplicate
        seen = set()
        unique = []
        for e in entities:
            key = e.lower()
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique[:30]

    # ════════════════════════════════════════════════════════════
    # SYNTHESIS
    # ════════════════════════════════════════════════════════════

    def _synthesize_answer(
        self, question: str, findings: list[ResearchFinding], all_text: str,
    ) -> list[str]:
        """Synthesize an answer from research findings."""
        answer_parts = []

        # Use top findings
        for finding in findings[:5]:
            # Extract the most relevant sentences
            sentences = re.split(r"(?<=[.!?])\s+", finding.text)
            relevant = []
            q_words = set(re.findall(r"\b[a-z]{3,}\b", question.lower()))

            for s in sentences:
                s_words = set(re.findall(r"\b[a-z]{3,}\b", s.lower()))
                if q_words & s_words:
                    relevant.append(s.strip())

            if relevant:
                # Take the most relevant sentence
                best = max(relevant, key=lambda s: len(set(re.findall(r"\b[a-z]{3,}\b", s.lower())) & q_words))
                answer_parts.append(f"[{finding.source}] {best}")

        return answer_parts

    def _generate_summary(
        self, query: str, findings: list[ResearchFinding], key_facts: list[str],
    ) -> str:
        """Generate a research summary."""
        sources = set(f.source for f in findings)

        summary = f"Research: {query}\n"
        summary += f"Sources: {', '.join(sources)}\n"
        summary += f"Findings: {len(findings)}\n"
        summary += f"Key facts: {len(key_facts)}\n\n"

        if key_facts:
            summary += "Key findings:\n"
            for fact in key_facts[:5]:
                summary += f"  • {fact}\n"

        return summary
