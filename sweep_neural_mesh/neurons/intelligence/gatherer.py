"""
IntelligenceGatherer — collects information from multiple sources.

Sources:
  - Wikipedia / Wikidata (encyclopedic knowledge)
  - User-provided documents and text
  - Evidence from reasoning pipeline
  - World knowledge base
  - Live API queries
  - Conversation history

Each source produces GatheredIntel objects with metadata
about origin, confidence, and timestamp.
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class IntelSource(Enum):
    """Where the intelligence came from."""
    WIKIPEDIA = "wikipedia"
    WIKIDATA = "wikidata"
    DOCUMENT = "document"
    EVIDENCE = "evidence"
    WORLD_KNOWLEDGE = "world_knowledge"
    LIVE_API = "live_api"
    CONVERSATION = "conversation"
    USER_INPUT = "user_input"
    REASONING = "reasoning"
    UNKNOWN = "unknown"


@dataclass
class GatheredIntel:
    """A single piece of gathered intelligence.

    Attributes:
        content:    The actual information text.
        source:     Where it came from.
        topic:      Extracted topic/category.
        confidence: How reliable this piece is (0.0-1.0).
        timestamp:  When it was gathered.
        entities:   Named entities found in the content.
        relations:  Extracted relationships (subject-predicate-object).
        metadata:   Arbitrary extra data.
        content_id: Unique hash for deduplication.
    """
    content: str
    source: IntelSource
    topic: str = ""
    confidence: float = 0.8
    timestamp: float = field(default_factory=time.time)
    entities: list[dict[str, str]] = field(default_factory=list)
    relations: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    content_id: str = ""

    def __post_init__(self):
        if not self.content_id:
            self.content_id = hashlib.md5(self.content.encode()).hexdigest()[:12]

    def overlaps_with(self, other: GatheredIntel, threshold: float = 0.3) -> bool:
        """Check if two intel items overlap significantly."""
        words_a = set(re.findall(r"\b[a-z]{4,}\b", self.content.lower()))
        words_b = set(re.findall(r"\b[a-z]{4,}\b", other.content.lower()))
        if not words_a or not words_b:
            return False
        overlap = len(words_a & words_b)
        union = len(words_a | words_b)
        return (overlap / union if union else 0.0) >= threshold


class IntelligenceGatherer:
    """Collects intelligence from multiple sources.

    Usage::

        gatherer = IntelligenceGatherer()
        intel = gatherer.gather("What is quantum computing?")
        for item in intel:
            print(f"[{item.source.value}] {item.content[:80]}")
    """

    def __init__(self) -> None:
        self._seen_ids: set[str] = set()
        self._gathered: list[GatheredIntel] = []
        self._web_scraper = None  # Lazy init

    def gather(
        self,
        query: str,
        documents: list[str] | None = None,
        evidence: list[str] | None = None,
        world_knowledge: Any = None,
        live_retriever: Any = None,
        max_results: int = 20,
        web_scraper: Any = None,
    ) -> list[GatheredIntel]:
        """Gather intelligence from all available sources.

        Args:
            query:            What to gather intelligence about.
            documents:        User-provided documents.
            evidence:         Evidence from the reasoning pipeline.
            world_knowledge:  WorldKnowledge instance for fact-checking.
            live_retriever:   LiveKnowledgeRetriever for API queries.
            max_results:      Maximum items to return.

        Returns:
            List of GatheredIntel, deduplicated and ranked by relevance.
        """
        results: list[GatheredIntel] = []

        # 1. Gather from documents
        if documents:
            for doc in documents:
                results.extend(self._gather_from_document(doc, query))

        # 2. Gather from evidence
        if evidence:
            for ev in evidence:
                results.extend(self._gather_from_evidence(ev, query))

        # 3. Gather from world knowledge
        if world_knowledge is not None:
            results.extend(self._gather_from_world_knowledge(query, world_knowledge))

        # 4. Gather from live APIs
        if live_retriever is not None:
            results.extend(self._gather_from_live(query, live_retriever))

        # 5. Gather from web scraping
        if web_scraper is not None:
            results.extend(self._gather_from_web(query, web_scraper))

        # 6. Deduplicate
        results = self._deduplicate(results)

        # 6. Rank by relevance to query
        results = self._rank_by_relevance(results, query)

        # 7. Limit
        results = results[:max_results]

        # Track
        self._gathered.extend(results)
        self._seen_ids.update(r.content_id for r in results)

        return results

    def gather_from_text(
        self, text: str, source: IntelSource = IntelSource.DOCUMENT,
    ) -> list[GatheredIntel]:
        """Gather intelligence from raw text."""
        items = self._extract_items(text, source)
        self._gathered.extend(items)
        return items

    def _gather_from_web(self, query: str, scraper: Any) -> list[GatheredIntel]:
        """Gather intelligence from web scraping."""
        results = []
        try:
            pages = scraper.search_and_fetch(query, max_results=3)
            for page in pages:
                if page.success and page.text:
                    # Split into meaningful chunks
                    chunks = self._chunk_text(page.text, max_chunk=500)
                    for chunk in chunks[:3]:
                        results.append(GatheredIntel(
                            content=chunk,
                            source=IntelSource.LIVE_API,
                            topic=self._extract_topic(chunk),
                            confidence=page.confidence * 0.9,
                            metadata={
                                "url": page.url,
                                "title": page.title,
                                "source_type": "web_scrape",
                            },
                        ))
        except Exception:
            pass
        return results

    @staticmethod
    def _chunk_text(text: str, max_chunk: int = 500) -> list[str]:
        """Split text into meaningful chunks."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        for s in sentences:
            if len(current) + len(s) > max_chunk and current:
                chunks.append(current.strip())
                current = s
            else:
                current = current + " " + s if current else s
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def get_all_gathered(self) -> list[GatheredIntel]:
        """Return all gathered intelligence."""
        return list(self._gathered)

    def clear(self) -> None:
        """Clear gathered intelligence."""
        self._gathered.clear()
        self._seen_ids.clear()

    # ── Internal ────────────────────────────────────────────

    def _gather_from_document(self, doc: str, query: str) -> list[GatheredIntel]:
        """Extract intelligence from a document."""
        return self._extract_items(doc, IntelSource.DOCUMENT, query=query)

    def _gather_from_evidence(self, ev: str, query: str) -> list[GatheredIntel]:
        """Extract intelligence from evidence."""
        items = self._extract_items(ev, IntelSource.EVIDENCE, query=query)
        # Boost confidence for evidence items
        for item in items:
            item.confidence = min(1.0, item.confidence + 0.1)
        return items

    def _gather_from_world_knowledge(self, query: str, wk: Any) -> list[GatheredIntel]:
        """Gather from the world knowledge base."""
        results = []
        try:
            check = wk.check_claim(query)
            if check.matching_entities:
                for entity_name in check.matching_entities[:3]:
                    entity = wk._entities.get(entity_name.lower())
                    if entity:
                        content = f"{entity.name} ({entity.category}): "
                        content += ", ".join(
                            f"{k}: {v}" for k, v in list(entity.properties.items())[:5]
                        )
                        results.append(GatheredIntel(
                            content=content,
                            source=IntelSource.WORLD_KNOWLEDGE,
                            topic=entity.category,
                            confidence=check.confidence,
                            entities=[{"name": entity.name, "type": entity.category}],
                            metadata={"entity_name": entity.name},
                        ))
        except Exception:
            pass
        return results

    def _gather_from_live(self, query: str, retriever: Any) -> list[GatheredIntel]:
        """Gather from live API retrieval."""
        results = []
        try:
            result = retriever.retrieve(query)
            if result and result.success and result.answer:
                results.append(GatheredIntel(
                    content=result.answer,
                    source=IntelSource.LIVE_API,
                    topic=self._extract_topic(query),
                    confidence=result.confidence,
                    metadata={"api_source": result.source, "raw_data": result.raw_data},
                ))
        except Exception:
            pass
        return results

    def _extract_items(
        self, text: str, source: IntelSource, query: str = "",
    ) -> list[GatheredIntel]:
        """Extract intelligence items from text."""
        items = []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # Extract entities
            entities = self._extract_entities(sentence)

            # Extract relations
            relations = self._extract_relations(sentence)

            # Determine topic
            topic = self._extract_topic(sentence)

            # Compute confidence based on content quality
            confidence = self._compute_confidence(sentence, source)

            items.append(GatheredIntel(
                content=sentence,
                source=source,
                topic=topic,
                confidence=confidence,
                entities=entities,
                relations=relations,
            ))

        return items

    def _extract_entities(self, text: str) -> list[dict[str, str]]:
        """Simple entity extraction using patterns."""
        entities = []

        # Proper nouns (capitalized words)
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        for noun in proper_nouns:
            if len(noun) > 2:
                entities.append({"name": noun, "type": "entity"})

        # Numbers with units
        numbers = re.findall(r'\b(\d[\d,\.]*\s*(?:meters|km|kg|°C|°F|mph|%|years|billion|million))\b', text)
        for num in numbers:
            entities.append({"name": num, "type": "measurement"})

        return entities[:10]

    def _extract_relations(self, text: str) -> list[dict[str, str]]:
        """Extract subject-predicate-object relations."""
        relations = []

        # "X is Y" pattern
        is_patterns = re.findall(
            r'(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are|was|were)\s+(.+?)(?:\.|$)',
            text,
        )
        for subj, obj in is_patterns:
            relations.append({"subject": subj.strip(), "predicate": "is", "object": obj.strip()})

        # "X causes Y" pattern
        cause_patterns = re.findall(
            r'(\b.+?\b)\s+(?:causes?|leads to|results in)\s+(.+?)(?:\.|$)',
            text, re.IGNORECASE,
        )
        for subj, obj in cause_patterns:
            relations.append({"subject": subj.strip(), "predicate": "causes", "object": obj.strip()})

        return relations[:5]

    def _extract_topic(self, text: str) -> str:
        """Extract the main topic from text."""
        text_lower = text.lower()

        topic_keywords = {
            "physics": ["energy", "force", "gravity", "quantum", "particle", "atom", "electron"],
            "biology": ["cell", "dna", "organism", "species", "evolution", "protein", "gene"],
            "chemistry": ["molecule", "reaction", "element", "compound", "acid", "bond"],
            "geography": ["country", "city", "river", "mountain", "ocean", "continent"],
            "history": ["war", "revolution", "century", "ancient", "empire", "dynasty"],
            "technology": ["computer", "software", "internet", "algorithm", "data", "network"],
            "medicine": ["disease", "treatment", "symptom", "diagnosis", "patient", "drug"],
            "mathematics": ["equation", "theorem", "number", "function", "proof", "calculate"],
            "astronomy": ["star", "planet", "galaxy", "universe", "telescope", "orbit"],
            "ecology": ["climate", "ecosystem", "species", "habitat", "conservation", "biodiversity"],
        }

        scores: dict[str, int] = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[topic] = score

        if scores:
            return max(scores, key=scores.get)
        return "general"

    def _compute_confidence(self, text: str, source: IntelSource) -> float:
        """Compute confidence based on content quality signals."""
        base = 0.7

        # Source adjustment
        source_boost = {
            IntelSource.WIKIPEDIA: 0.15,
            IntelSource.WIKIDATA: 0.12,
            IntelSource.WORLD_KNOWLEDGE: 0.10,
            IntelSource.DOCUMENT: 0.05,
            IntelSource.EVIDENCE: 0.10,
            IntelSource.LIVE_API: 0.08,
        }
        base += source_boost.get(source, 0.0)

        # Content quality signals
        if re.search(r'\b\d{4}\b', text):  # Has dates
            base += 0.03
        if re.search(r'\b\d[\d,\.]+\b', text):  # Has numbers
            base += 0.02
        if len(text.split()) > 10:  # Substantial content
            base += 0.03
        if any(w in text.lower() for w in ["according to", "research shows", "study found"]):
            base += 0.05

        return min(0.99, base)

    def _deduplicate(self, items: list[GatheredIntel]) -> list[GatheredIntel]:
        """Remove duplicate and overlapping items."""
        if not items:
            return []

        unique: list[GatheredIntel] = []
        seen_content: set[str] = set()

        for item in items:
            # Skip exact duplicates
            if item.content_id in self._seen_ids:
                continue
            if item.content in seen_content:
                continue

            # Skip near-duplicates
            is_dup = False
            for existing in unique:
                if item.overlaps_with(existing, threshold=0.7):
                    # Keep the higher-confidence one
                    if item.confidence > existing.confidence:
                        unique.remove(existing)
                    else:
                        is_dup = True
                    break

            if not is_dup:
                unique.append(item)
                seen_content.add(item.content)

        return unique

    def _rank_by_relevance(
        self, items: list[GatheredIntel], query: str,
    ) -> list[GatheredIntel]:
        """Rank items by relevance to the query."""
        query_words = set(re.findall(r"\b[a-z]{3,}\b", query.lower()))

        def relevance(item: GatheredIntel) -> float:
            item_words = set(re.findall(r"\b[a-z]{3,}\b", item.content.lower()))
            if not query_words or not item_words:
                return 0.0
            overlap = len(query_words & item_words)
            return overlap / len(query_words)

        items.sort(key=lambda x: (relevance(x), x.confidence), reverse=True)
        return items
