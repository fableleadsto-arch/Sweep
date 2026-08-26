"""
Processing Centers — the specialized regions of Sweep's reasoning brain.

Each center does ONE thing and does it well, just like biological
brain regions. They never call each other directly — they communicate
exclusively through Signals and Synapses.

    ┌──────────────────────────────────────────────┐
    │           REASONING CORTEX                   │
    │                                              │
    │  ┌─────────┐  ┌──────────┐  ┌──────────┐   │
    │  │Evidence  │  │Credibility│  │Temporal  │   │
    │  │Gatherer  │  │Assessor   │  │Sequencer │   │
    │  └────┬────┘  └─────┬────┘  └────┬─────┘   │
    │       │              │            │          │
    │       └──────┬───────┘────────────┘          │
    │              ↓                               │
    │  ┌───────────────────────────────────┐       │
    │  │       INTEGRATION HUB             │       │
    │  └───────────────┬───────────────────┘       │
    │                  ↓                           │
    │  ┌─────────┐  ┌──────────┐  ┌──────────┐   │
    │  │Causal   │  │Contradic-│  │Consensus │   │
    │  │Linker   │  │tion Det. │  │ Engine   │   │
    │  └────┬────┘  └─────┬────┘  └────┬─────┘   │
    │       │              │            │          │
    │       └──────┬───────┘────────────┘          │
    │              ↓                               │
    │  ┌───────────────────────────────────┐       │
    │  │    EXPLANATION NARRATOR            │       │
    │  └───────────────────────────────────┘       │
    └──────────────────────────────────────────────┘
"""
from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Any

from .signal import Signal, SignalType
from .world_knowledge import WorldKnowledge
from .logical_inference import LogicalInferenceEngine

logger = logging.getLogger("sweep.neurons.centers")

# Singleton world knowledge instance (initialized once)
_world_knowledge: WorldKnowledge | None = None
_logical_engine: LogicalInferenceEngine | None = None


def get_world_knowledge() -> WorldKnowledge:
    global _world_knowledge
    if _world_knowledge is None:
        _world_knowledge = WorldKnowledge()
    return _world_knowledge


def get_logical_engine() -> LogicalInferenceEngine:
    global _logical_engine
    if _logical_engine is None:
        _logical_engine = LogicalInferenceEngine()
    return _logical_engine


class ProcessingCenter(ABC):
    """
    Base class for all processing centers.

    Each center receives signals, applies its specialized computation,
    and produces output signals. Centers never share state — they
    communicate only through signals.
    """

    name: str = "base_center"

    @abstractmethod
    def process(self, signals: list[Signal]) -> list[Signal]:
        """Process incoming signals and produce output signals."""
        ...

    def _filter_signals(self, signals: list[Signal], signal_type: SignalType) -> list[Signal]:
        """Helper: extract signals of a specific type."""
        return [s for s in signals if s.signal_type == signal_type]

    def _stamp(self, signals: list[Signal]) -> list[Signal]:
        """Helper: stamp all signals with this center's name."""
        return [s.stamp(self.name) for s in signals]


# ──────────────────────────────────────────────────────────────────
# CENTER 1: Evidence Gatherer
# Receives raw input, identifies and scores individual evidence items.
# This is the "sensory receptor" — it converts raw data into signals.
# ──────────────────────────────────────────────────────────────────

class EvidenceGatherer(ProcessingCenter):
    """
    Extracts and scores evidence items from raw input.

    Like sensory receptors converting light/sound/pressure into
    neural signals, this center converts raw text/data into
    scored evidence signals.
    """
    name = "evidence_gatherer"

    def process(self, signals: list[Signal]) -> list[Signal]:
        raw_signals = self._filter_signals(signals, SignalType.RAW)
        if not raw_signals:
            return []

        evidence_items: list[Signal] = []
        for raw in raw_signals:
            items = raw.data.get("evidence", [])
            sources_list = raw.data.get("sources", [])
            query = str(raw.data.get("query", ""))
            if isinstance(items, str):
                items = [items]

            for idx, item in enumerate(items):
                if isinstance(item, str):
                    item = {"text": item}

                text = item.get("text", "")
                score = self._score_evidence(text, item, query)

                # Determine if evidence supports or refutes the query
                direction = self._detect_support_direction(text, query)

                # ── WORLD KNOWLEDGE GATE ──
                # If evidence contradicts known facts, flag it as suspicious
                wk = get_world_knowledge()
                wk_check = wk.check_claim(text)
                if not wk_check.plausible and wk_check.confidence > 0.7:
                    # Evidence contradicts world knowledge -- flip direction and reduce score
                    if direction == "supports":
                        direction = "refutes"
                    score *= 0.3  # heavy penalty for factually wrong evidence
                elif wk_check.plausible and wk_check.confidence > 0.7 and direction == "supports":
                    score = min(1.0, score * 1.2)  # boost for factually correct supporting evidence

                # Wire source metadata: associate source with evidence item
                source = item.get("source", "")
                if not source and idx < len(sources_list):
                    source = sources_list[idx]
                if not source:
                    url_match = re.search(r'https?://([^/\s]+)', text)
                    if url_match:
                        source = url_match.group(1)

                evidence_signals = Signal(
                    data={
                        **item,
                        "evidence_text": text,
                        "original_query": raw.data.get("query", ""),
                        "source": source,
                        "support_direction": direction,
                    },
                    signal_type=SignalType.EVIDENCE,
                    confidence=score,
                    source_center=self.name,
                    metadata={
                        "word_count": len(text.split()),
                        "has_url": bool(re.search(r'https?://', text)),
                        "has_date": bool(re.search(r'\d{4}[-/]\d{1,2}', text)),
                        "source_domain": source,
                        "support_direction": direction,
                    },
                )
                evidence_items.append(evidence_signals)

        logger.debug(f"EvidenceGatherer: produced {len(evidence_items)} evidence signals from {len(raw_signals)} raw inputs")
        return self._stamp(evidence_items)

    @staticmethod
    def _stem(word: str) -> str:
        """Simple suffix-stripping stemmer for word overlap detection."""
        w = word.lower()
        # Order matters: longest suffixes first
        for suffix in ("ation", "ment", "ness", "ible", "able", "tion", "sion", "ence", "ance", "ities", "ous", "ive", "ful", "less", "ally", "ment", "ness", "ing", "ies", "ied", "ily", "ers", "est", "ity", "ant", "ent", "ism", "ist", "ize", "ise", "ing", "ful", "ous", "ive", "ial", "ual", "ble", "cle", "ity", "ure", "ence", "ance"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                return w[:-len(suffix)]
        # Simple plural/verb endings
        if w.endswith("es") and len(w) > 4:
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss") and len(w) > 4:
            return w[:-1]
        if w.endswith("ed") and len(w) > 5:
            return w[:-2]
        if w.endswith("ing") and len(w) > 6:
            return w[:-3]
        return w

    def _stemmed_overlap(self, words_a: set[str], words_b: set[str]) -> int:
        """Count stemmed word overlaps between two sets."""
        stems_a = {self._stem(w) for w in words_a}
        stems_b = {self._stem(w) for w in words_b}
        return len(stems_a & stems_b)

    def _detect_support_direction(self, evidence_text: str, query: str) -> str:
        """
        Detect whether evidence supports or refutes the query.

        Returns "supports", "refutes", "mixed", or "neutral".
        """
        if not query or not evidence_text:
            return "neutral"

        ev_lower = evidence_text.lower()
        q_lower = query.lower()

        # ── Extract query content words ──
        q_words = set(re.findall(r'\b[a-z]{3,}\b', q_lower))
        stop = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "was", "this", "that", "with", "does", "did", "is", "do", "have", "has", "was", "were", "will", "would", "could", "should"}
        q_content = q_words - stop

        if not q_content:
            return "neutral"

        # ── Stemmed word overlap for relevance detection ──
        ev_words_raw = set(re.findall(r'\b[a-z]{3,}\b', ev_lower))
        has_relevant_overlap = bool(q_content & ev_words_raw) or self._stemmed_overlap(q_content, ev_words_raw) > 0

        # ── Negation detection ──
        negation_patterns = [
            r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bneither\b",
            r"\bdoesn.t\b", r"\bdidn.t\b", r"\bwon.t\b", r"\bisn.t\b",
            r"\bwasn.t\b", r"\bcan.t\b", r"\bcouldn.t\b", r"\bshouldn.t\b",
            r"\bunable\b", r"\bfail", r"\blacks?\b", r"\babsent\b",
            r"\bwithout\b", r"\bimpossible\b", r"\bcannot\b",
            r"\bpoor\b", r"\bincapable\b", r"\bnon\b",
        ]
        has_negation = any(re.search(p, ev_lower) for p in negation_patterns)

        # ── Semantic opposition detection ──
        # Direct "refutes" when evidence opposes query concepts
        opposition_map = {
            "flat": ["spheroid", "round", "sphere", "globe", "curved", "circular", "oblate"],
            "cold": ["hot", "heat", "thermal", "burns", "flame", "temperature"],
            "fly": ["swim", "flightless", "walk on land"],
            "liquid": ["solid", "gas", "vapor", "frozen"],
            "alive": ["dead", "deceased", "extinct", "fossil", "inanimate"],
            "visible": ["invisible", "undetectable"],
            "teeth": ["beak", "toothless"],
            "tall": ["shorter", "smaller"],
            "tallest": ["shorter", "smaller", "taller"],
            "fastest": ["slower"],
            "fast": ["slower", "slow", "stationary", "immobile"],
            "faster": ["slower", "slow", "stationary", "immobile"],
            "highest": ["lower", "shorter"],
            "biggest": ["smaller"],
            "oldest": ["newer", "younger"],
            "see": ["invisible", "undetectable"],
            "fruit": ["vegetable"],
            "fish": ["mammal", "reptile", "amphibian"],
            "sandwich": ["taco", "wrap"],
            "real": ["abstract", "conceptual"],
            "breathe": ["drown", "suffocate"],
            "drown": ["buoyant", "float easily"],
            "vegetable": ["fruit"],
            "bird": ["mammal", "reptile"],
            "mammal": ["fish", "reptile", "insect"],
            "round": ["square", "rectangular", "flat"],
            "planet": ["star"],
            "mineral": ["vitamin"],
            "at night": ["daytime", "during the day"],
            "talk": ["silent", "mute"],
            "heavy": ["lightweight", "featherweight"],
        }

        ev_words = ev_words_raw
        found_opposition = False
        for q_word in q_content:
            if q_word in opposition_map:
                opposing_words = opposition_map[q_word]
                for ow in opposing_words:
                    ow_words = set(ow.split())
                    if ow_words.issubset(ev_words) or ow in ev_lower:
                        support_context = ["describes", "means", "defined as"]
                        is_supporting = any(sc in ev_lower for sc in support_context)
                        if not is_supporting:
                            found_opposition = True
                            break
                if found_opposition:
                    break

        # ── Explicit "mixed" detection: single evidence with BOTH yes/true AND refutation ──
        # e.g., "Culinarily yes, botanically it is a fruit"
        has_yes_or_true = bool(re.search(r'\b(yes|true|correct|affirmative)\b', ev_lower))
        has_explicit_support = bool(re.search(
            r'\b(classified|categorized|considered|treated as|known as|type of|form of)\b', ev_lower))
        has_explicit_refutation = bool(re.search(
            r'\b(not a|not the|myth|false|incorrect|debunked|actually)\b', ev_lower))
        has_negation_near_query = False
        if has_negation:
            for qw in q_content:
                qw_stem = self._stem(qw)
                for ev_word in ev_words:
                    if self._stem(ev_word) == qw_stem:
                        neg_positions = [m.start() for p in negation_patterns for m in re.finditer(p, ev_lower)]
                        ev_pos = ev_lower.find(ev_word)
                        if any(abs(np - ev_pos) < 60 for np in neg_positions):
                            has_negation_near_query = True
                            break
                if has_negation_near_query:
                    break

        # ── "but/however/although" clause splitting → mixed detection ──
        # Evidence like "Most mammals can swim, but sloths and gorillas are poor swimmers"
        # contains both supporting and refuting elements
        conjunctions = re.split(r'\b(?:but|however|although|though|whereas|yet|on the other hand)\b', ev_lower)
        if len(conjunctions) >= 2:
            has_support_in_any = False
            has_refute_in_any = False
            for clause in conjunctions:
                clause_has_neg = any(re.search(p, clause) for p in negation_patterns)
                clause_has_support_words = bool(re.search(
                    r'\b(yes|true|correct|can|do|does|is|are|will|have|has|make|need|produce|use|contains?|include)\b', clause))
                if clause_has_neg:
                    has_refute_in_any = True
                elif clause_has_support_words:
                    has_support_in_any = True
            if has_support_in_any and has_refute_in_any:
                return "mixed"

        # ── Numerical comparison detection ──
        # If evidence contains numbers being compared (e.g., "300000 km/s" vs "0.34 km/s")
        # and the comparison implies the opposite of the query
        numbers_in_ev = re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:km|mph|m/s|km/s|%|kg|lb|miles?|feet|meters?)?', ev_lower)
        if len(numbers_in_ev) >= 2 and q_content:
            faster_words = {"faster", "quicker", "speedier", "more rapid"}
            slower_words = {"slower", "sluggish", "less rapid"}
            bigger_words = {"bigger", "larger", "greater", "heavier", "taller", "higher"}
            smaller_words = {"smaller", "less", "lighter", "shorter", "lower"}
            query_implies_faster = bool(q_content & faster_words)
            query_implies_slower = bool(q_content & slower_words)
            query_implies_bigger = bool(q_content & bigger_words)
            query_implies_smaller = bool(q_content & smaller_words)
            if query_implies_faster or query_implies_slower or query_implies_bigger or query_implies_smaller:
                nums = [float(n) for n in numbers_in_ev]
                has_comparison = bool(re.search(r'\b(while|vs\.?|compared|versus|whereas)\b', ev_lower))
                # Approach: find which query entity is mentioned near which number
                # by checking proximity of entity words to number positions
                q_entity_words = [w for w in q_content if len(w) > 3]
                entity_nums: dict[str, float] = {}
                for qw in q_entity_words:
                    pos = ev_lower.find(qw)
                    if pos >= 0:
                        best_dist = 999
                        best_num = None
                        for m in re.finditer(r'\b(\d+(?:\.\d+)?)\s*(?:km|mph|m/s|km/s|%|kg|lb|miles?|feet|meters?)?', ev_lower):
                            n = float(m.group(1))
                            dist = abs(m.start() - pos)
                            if dist < best_dist:
                                best_dist = dist
                                best_num = n
                        if best_num is not None:
                            entity_nums[qw] = best_num
                if len(entity_nums) >= 2:
                    entity_list = list(entity_nums.values())
                    smaller_is = entity_list[0] < entity_list[1]
                    if query_implies_faster and smaller_is:
                        return "refutes"
                    if query_implies_slower and not smaller_is:
                        return "refutes"
                    if query_implies_bigger and smaller_is:
                        return "refutes"
                    if query_implies_smaller and not smaller_is:
                        return "refutes"
                elif has_comparison and len(nums) == 2:
                    if query_implies_faster and nums[0] < nums[1]:
                        return "refutes"
                    if query_implies_slower and nums[0] > nums[1]:
                        return "refutes"
                    if query_implies_bigger and nums[0] < nums[1]:
                        return "refutes"
                    if query_implies_smaller and nums[0] > nums[1]:
                        return "refutes"

        if (has_yes_or_true or has_explicit_support) and (has_negation_near_query or found_opposition or has_explicit_refutation):
            return "mixed"

        # If opposition found and NOT mixed → direct refutes
        if found_opposition:
            return "refutes"

        # ── Explicit refutation patterns ──
        q_alternation = "|".join(sorted(q_content, key=len, reverse=True))
        refutation_patterns = [
            r"myth\b", r"false\b", r"incorrect\b", r"not\s+true\b",
            r"no\s+evidence\b", r"debunked\b", r"contrary\s+to\b",
            r"actual\w*\b.*(?:" + q_alternation + r")",
        ]
        for pattern in refutation_patterns:
            if re.search(pattern, ev_lower):
                if has_relevant_overlap:
                    return "refutes"

        # ── Negation near query keywords → refutes ──
        if has_negation_near_query:
            return "refutes"

        # ── Explicit support patterns ──
        support_patterns = [
            r"\bsupports?\b", r"\bconfirms?\b", r"\bdemonstrates?\b",
            r"\bshows?\s+that\b", r"\bindicates?\b", r"\bproves?\b",
            r"\byes\b", r"\btrue\b", r"\bcorrect\b",
            r"\bclassified\b.*\bas\b", r"\btype\s+of\b", r"\bform\s+of\b",
        ]
        if any(re.search(p, ev_lower) for p in support_patterns):
            return "supports"

        # ── Direct factual answer detection ──
        # Short evidence that is a number/name answering a "what/how many/who" question
        ev_stripped = evidence_text.strip()
        ev_word_count = len(ev_stripped.split())
        # Number answering "how many" or "what is the X of Y"
        if ev_word_count <= 5 and re.search(r'\d+', ev_stripped):
            if re.search(r'\bhow\s+many\b', q_lower) or re.search(r'\bwhat\s+(?:is|are|was)\b', q_lower):
                return "supports"
        # Name answering "who" or "what is the capital/who discovered" etc.
        if ev_word_count <= 6 and re.search(r'^[A-Z]', ev_stripped):
            if re.search(r'\bwho\b', q_lower) or re.search(r'\bwhat\s+(?:is|are|was)\s+the\b', q_lower):
                return "supports"
        # Factual answer patterns: "fact: X", "answer: X", "result: X"
        if re.search(r'(?:relevant fact|fact|answer|result)\s*[:;]', ev_lower):
            if has_relevant_overlap or re.search(r'\b\d+\b', ev_lower):
                return "supports"
        # If evidence is just a number and query asks "how many"
        if re.search(r'^\d+\b', ev_stripped) and re.search(r'how many', q_lower):
            return "supports"
        # If evidence is a name (capitalized words) and query asks "who"
        if re.search(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', ev_stripped) and re.search(r'\bwho\b', q_lower):
            return "supports"

        # ── Default: if evidence is relevant → supports ──
        if has_relevant_overlap:
            return "supports"
        return "neutral"

    def _score_evidence(self, text: str, item: dict, query: str = "") -> float:
        """Score evidence based on query relevance, specificity, and structure."""
        if not text:
            return 0.0

        score = 0.40  # lower base — rewards strong evidence, penalizes weak

        # ── Query relevance (most important — up to +0.25) ──
        if query:
            stop = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "was", "this", "that", "with"}
            ev_words = set(re.findall(r'\b[a-z]{3,}\b', text.lower())) - stop
            q_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower())) - stop
            if q_words:
                exact_overlap = len(ev_words & q_words)
                stemmed_overlap = self._stemmed_overlap(ev_words, q_words)
                relevance = max(exact_overlap, stemmed_overlap) / max(1, len(q_words))
                score += min(0.25, relevance * 0.25)
                # Heavy penalty for zero relevance (irrelevant evidence)
                if relevance == 0.0:
                    # Check if evidence contains a direct factual answer (number, name after colon)
                    has_answer_hint = bool(re.search(
                        r'(?:fact|answer|result|confirmed|equals?|is)\s*[:;]?\s*\d+'
                        r'|(?:one relevant fact|relevant fact)\s*[:;]',
                        text.lower()))
                    # Check if evidence contains a capitalized name (potential person/place)
                    has_named_entity = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
                    # Check if evidence contains a number that could answer a "how many" query
                    has_quantity = bool(re.search(r'\b\d+\b', text)) and bool(re.search(r'how many', query.lower()))
                    if has_answer_hint or has_named_entity or has_quantity:
                        score += 0.10  # partial relevance for answer-like evidence
                    else:
                        score -= 0.15

        # ── Information density: word count ──
        word_count = len(text.split())
        if word_count > 50:
            score += 0.20
        elif word_count > 20:
            score += 0.12
        elif word_count < 5:
            # Short evidence penalty — but reduce if it contains a named entity or number
            has_named = bool(re.search(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
            has_number = bool(re.search(r'\b\d+\b', text))
            if has_named or has_number:
                score -= 0.05  # lighter penalty for short factual evidence
            else:
                score -= 0.20

        # ── Numerical precision (+0.10) ──
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        if numbers:
            score += min(0.10, len(numbers) * 0.04)

        # ── Attribution (+0.12) ──
        attributions = [
            "according to", "reported by", "study", "research", "survey",
            "found", "shows", "data", "evidence", "confirmed", "documented",
            "published", "observed", "measured", "recorded",
        ]
        if any(a in text.lower() for a in attributions):
            score += 0.12

        # ── Named entity density (+0.08) ──
        named_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if named_entities:
            score += min(0.08, len(named_entities) * 0.03)

        # ── URL presence (+0.05) ──
        if re.search(r'https?://', text):
            score += 0.05

        # ── Hedging language penalty (-0.08) ──
        hedges = ["might", "could", "possibly", "perhaps", "maybe", "seems", "appears", "allegedly"]
        if any(h in text.lower() for h in hedges):
            score -= 0.08

        # ── Vague/generic penalty (-0.15) ──
        generic = ["it is known", "generally", "people say", "obviously", "clearly", "everyone knows"]
        if any(g in text.lower() for g in generic):
            score -= 0.15

        # ── Very generic one-word evidence ──
        if text.lower().strip() in ("yes", "no", "true", "false", "maybe"):
            score -= 0.25

        return max(0.0, min(1.0, score))


# ──────────────────────────────────────────────────────────────────
# CENTER 2: Credibility Assessor
# Evaluates how trustworthy each piece of evidence is.
# Like the amygdala assessing threat — this center assesses trust.
# ──────────────────────────────────────────────────────────────────

class CredibilityAssessor(ProcessingCenter):
    """
    Scores the credibility of each evidence signal.

    Uses source reputation, internal consistency, and
    corroboration with other evidence to estimate trust.
    """
    name = "credibility_assessor"

    # Known high-credibility domains/sources
    TRUSTED_SOURCES: set[str] = {
        # Academic & research
        "wikipedia", "github", "arxiv", "pubmed", "nature", "science",
        "ieee", "acm", "doi", "springer", "elsevier", "wiley", "jstor",
        "scholar", "researchgate", "plos", "cell.com", "lancet",
        # News & media
        "reuters", "apnews", "bbc", "nytimes", "washingtonpost",
        "theguardian", "economist", "ft.com", "bloomberg", "wsj",
        "ap", "upi", "afp", "nhk", "dw.com", "aljazeera",
        # Government & institutional
        "gov", "edu", "mil", "nist", "who", "cdc", "nih", "fda",
        "un.org", "worldbank", "imf", "oecd", "nasa", "esa",
        # Tech & industry
        "stackoverflow", "docs.python.org", "developer.mozilla",
        "ietf.org", "w3.org", "linux.org", "kaggle",
    }

    # Known low-credibility patterns
    UNTRUSTED_PATTERNS: list[str] = [
        r"click\s*here", r"you\s*won.t\s*believe",
        r"secret\s*trick", r"miracle", r"100%\s*guaranteed",
        r"buy\s*now", r"limited\s*time", r"act\s*now",
        r"this\s*one\s*weird", r"doctors?\s*hate",
        r"exposed!", r"shocking\s*truth", r"cover[\s-]?up",
        r"mainstream\s*media", r"wake\s*up\s*sheeple",
        r"free\s*money", r"get\s*rich\s*quick",
        r"conspiracy", r"they\s*don.t\s*want\s*you\s*to\s*know",
    ]

    # Hedging language (reduces credibility)
    HEDGING_WORDS: list[str] = [
        "allegedly", "reportedly", "supposedly", "claimed",
        "some say", "many believe", "it is said", "rumor",
        "unconfirmed", "anonymous source", "people are saying",
    ]

    # Expertise indicators (boosts credibility)
    EXPERTISE_INDICATORS: list[str] = [
        "peer.reviewed", "methodology", "sample size", "statistical",
        "randomized", "controlled trial", "meta.analysis", "systematic review",
        "longitudinal", "cross.sectional", "double.blind", "placebo",
        "confidence interval", "p.value", "standard deviation",
    ]

    def process(self, signals: list[Signal]) -> list[Signal]:
        evidence_signals = self._filter_signals(signals, SignalType.EVIDENCE)
        if not evidence_signals:
            return []

        assessed: list[Signal] = []
        for sig in evidence_signals:
            credibility = self._assess(sig)
            assessed.append(Signal(
                data={
                    "evidence_text": sig.data.get("evidence_text", ""),
                    "credibility_score": credibility,
                    "source": sig.data.get("source", "unknown"),
                    "original_query": sig.data.get("original_query", ""),
                },
                signal_type=SignalType.CREDIBILITY,
                confidence=credibility,
                source_center=self.name,
                metadata={
                    "original_confidence": sig.confidence,
                    "assessment_factors": self._get_factors(sig),
                },
                history=list(sig.history),
            ))

        logger.debug(f"CredibilityAssessor: assessed {len(assessed)} evidence items")
        return self._stamp(assessed)

    def _assess(self, signal: Signal) -> float:
        """Compute credibility score 0.0–1.0 using source metadata from pipeline."""
        text = signal.data.get("evidence_text", "")
        text_lower = text.lower()
        source = str(signal.data.get("source", "")).lower()
        score = 0.5

        # Source credibility — use actual source if wired through pipeline
        if any(trusted in source for trusted in self.TRUSTED_SOURCES):
            score += 0.25
        elif source and source not in ("unknown", ""):
            score += 0.08  # has a source, small boost

        # Check for untrusted patterns
        for pattern in self.UNTRUSTED_PATTERNS:
            if re.search(pattern, text_lower):
                score -= 0.30
                break

        # Hedging language penalty
        for hedge in self.HEDGING_WORDS:
            if hedge in text_lower:
                score -= 0.10
                break

        # Expertise indicators boost
        expertise_count = sum(1 for ind in self.EXPERTISE_INDICATORS if ind in text_lower)
        if expertise_count > 0:
            score += min(0.15, expertise_count * 0.05)

        # Citation/reference boost
        if re.search(r'\[\d+\]|\(20\d{2}\)|doi:|arxiv:|pmid:', text_lower):
            score += 0.10

        # Text quality signals
        if len(text) > 100 and "." in text:
            score += 0.10
        if len(text) > 300:
            score += 0.05  # substantial text

        # Penalize very short evidence (likely not substantive)
        if len(text.split()) < 5:
            score -= 0.10

        # Signal itself carried some confidence from evidence gatherer
        score = score * 0.7 + signal.confidence * 0.3

        return max(0.0, min(1.0, score))

    def _get_factors(self, signal: Signal) -> list[str]:
        factors: list[str] = []
        source = str(signal.data.get("source", "")).lower()
        if any(t in source for t in self.TRUSTED_SOURCES):
            factors.append("trusted_source")
        elif source and source not in ("unknown", ""):
            factors.append("identified_source")
        text = signal.data.get("evidence_text", "")
        if len(text) > 100:
            factors.append("substantial_text")
        if re.search(r'\[\d+\]|\(20\d{2}\)', text):
            factors.append("has_citations")
        return factors


# ──────────────────────────────────────────────────────────────────
# CENTER 3: Temporal Sequencer
# Orders evidence chronologically and applies time-decay.
# Like the hippocampus encoding time — recency matters.
# ──────────────────────────────────────────────────────────────────

class TemporalSequencer(ProcessingCenter):
    """
    Orders evidence in time and applies recency weighting.

    Recent evidence is generally more relevant, but well-established
    historical facts retain their weight. The sequencer tracks
    temporal relationships between evidence items.
    """
    name = "temporal_sequencer"

    def process(self, signals: list[Signal]) -> list[Signal]:
        evidence_signals = self._filter_signals(signals, SignalType.EVIDENCE)
        if not evidence_signals:
            return []

        # Extract dates and sort
        dated_items: list[tuple[float, Signal]] = []
        undated: list[Signal] = []

        for sig in evidence_signals:
            text = sig.data.get("evidence_text", "")
            date_score = self._extract_date_relevance(text)
            if date_score > 0:
                dated_items.append((date_score, sig))
            else:
                undated.append(sig)

        # Sort dated items by relevance (most relevant first)
        dated_items.sort(key=lambda x: x[0], reverse=True)

        output: list[Signal] = []
        for rank, (date_score, sig) in enumerate(dated_items):
            temporal_confidence = sig.confidence * (0.5 + 0.5 * date_score)
            output.append(Signal(
                data={
                    "evidence_text": sig.data.get("evidence_text", ""),
                    "temporal_rank": rank,
                    "date_relevance": date_score,
                    "original_query": sig.data.get("original_query", ""),
                },
                signal_type=SignalType.TEMPORAL,
                confidence=temporal_confidence,
                source_center=self.name,
                metadata={
                    "original_confidence": sig.confidence,
                    "ranked_among": len(dated_items),
                },
                history=list(sig.history),
            ))

        # Undated evidence gets neutral temporal score
        for sig in undated:
            output.append(Signal(
                data={
                    "evidence_text": sig.data.get("evidence_text", ""),
                    "temporal_rank": -1,
                    "date_relevance": 0.5,
                    "original_query": sig.data.get("original_query", ""),
                },
                signal_type=SignalType.TEMPORAL,
                confidence=sig.confidence * 0.75,
                source_center=self.name,
                metadata={"undated": True},
                history=list(sig.history),
            ))

        logger.debug(f"TemporalSequencer: ranked {len(dated_items)} dated + {len(undated)} undated → {len(output)} outputs")
        return self._stamp(output)

    def _extract_date_relevance(self, text: str) -> float:
        """Extract date from text and return recency score 0.0–1.0."""
        # Look for year patterns
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            year = int(year_match.group())
            # 2026 is "now" — score decays as you go further back
            current_year = 2026
            age = current_year - year
            if age < 0:
                return 0.5  # future date, suspicious
            if age < 2:
                return 1.0  # very recent
            if age < 5:
                return 0.85
            if age < 10:
                return 0.70
            if age < 30:
                return 0.55
            return 0.40  # older

        # Look for relative time
        if re.search(r'(today|yesterday|this\s+(week|month|year))', text, re.IGNORECASE):
            return 0.95
        if re.search(r'(last\s+(week|month|year)|recently)', text, re.IGNORECASE):
            return 0.80

        return 0.0  # no date found


# ──────────────────────────────────────────────────────────────────
# CENTER 4: Causal Linker
# Finds causal and semantic relationships between evidence items.
# Like association cortex connecting related concepts.
# ──────────────────────────────────────────────────────────────────

class CausalLinker(ProcessingCenter):
    """
    Links evidence items by causal and semantic relationships.

    Finds which evidence items support, explain, or cause each other.
    Uses keyword overlap, entity co-occurrence, and causal language
    as signals of connection.
    """
    name = "causal_linker"

    CAUSAL_MARKERS: list[str] = [
        r"because", r"therefore", r"caused?\s+by", r"leads?\s+to",
        r"result", r"consequence", r"due\s+to", r"as\s+a\s+result",
        r"since", r"thus", r"hence", r"accordingly",
    ]

    SUPPORT_MARKERS: list[str] = [
        r"confirms?", r"supports?", r"agrees?\s+with",
        r"corroborates?", r"consistent\s+with", r"evidence\s+(for|of)",
        r"demonstrates?", r"shows?\s+that", r"indicates?",
    ]

    def process(self, signals: list[Signal]) -> list[Signal]:
        evidence_signals = self._filter_signals(signals, SignalType.EVIDENCE)
        if len(evidence_signals) < 2:
            return []

        links: list[Signal] = []
        for i, sig_a in enumerate(evidence_signals):
            for sig_b in evidence_signals[i + 1:]:
                link = self._find_link(sig_a, sig_b)
                if link:
                    links.append(link)

        logger.debug(f"CausalLinker: found {len(links)} causal/supportive links from {len(evidence_signals)} evidence pairs")
        return self._stamp(links)

    def _find_link(self, sig_a: Signal, sig_b: Signal) -> Signal | None:
        """Try to find a causal or supportive link between two evidence items."""
        text_a = sig_a.data.get("evidence_text", "").lower()
        text_b = sig_b.data.get("evidence_text", "").lower()

        if not text_a or not text_b:
            return None

        # ── Keyword overlap (Jaccard) ──
        stop = {"that", "this", "with", "from", "have", "been", "were", "they", "their", "than"}
        words_a = set(re.findall(r'\b[a-z]{4,}\b', text_a)) - stop
        words_b = set(re.findall(r'\b[a-z]{4,}\b', text_b)) - stop
        if not words_a or not words_b:
            return None

        overlap = len(words_a & words_b)
        union = len(words_a | words_b)
        similarity = overlap / union if union > 0 else 0.0

        # ── Named entity co-occurrence (+0.15 per shared entity, max +0.30) ──
        # Extract capitalized words (potential named entities)
        entities_a = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                                     sig_a.data.get("evidence_text", "")))
        entities_b = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                                     sig_b.data.get("evidence_text", "")))
        shared_entities = entities_a & entities_b
        entity_boost = min(0.30, len(shared_entities) * 0.15)

        # ── Rare word overlap bonus ──
        # Words appearing in only one text are more informative
        all_words = words_a | words_b
        rare_bonus = 0.0
        if len(words_a & words_b) >= 2:
            # If 2+ shared words are relatively long (4+ chars), they're likely content words
            content_shared = {w for w in words_a & words_b if len(w) >= 5}
            rare_bonus = min(0.10, len(content_shared) * 0.03)

        # ── Causal language detection ──
        has_causal_a = any(re.search(m, text_a) for m in self.CAUSAL_MARKERS)
        has_causal_b = any(re.search(m, text_b) for m in self.CAUSAL_MARKERS)
        has_support_a = any(re.search(m, text_a) for m in self.SUPPORT_MARKERS)
        has_support_b = any(re.search(m, text_b) for m in self.SUPPORT_MARKERS)

        # ── Contradictory link detection ──
        contradictory_pairs = [
            ("increase", "decrease"), ("rise", "fall"), ("grow", "shrink"),
            ("positive", "negative"), ("support", "oppose"), ("confirm", "deny"),
            ("safe", "dangerous"), ("effective", "ineffective"), ("true", "false"),
        ]
        is_contradictory = False
        for a, b in contradictory_pairs:
            if (a in text_a and b in text_b) or (b in text_a and a in text_b):
                is_contradictory = True
                break

        # ── Determine link type and strength ──
        if is_contradictory:
            # Contradictory evidence → weak link (they're about the same topic but opposing)
            link_type = "contradictory"
            strength = similarity * 0.5 + entity_boost * 0.3
        elif has_causal_a or has_causal_b:
            link_type = "causal"
            strength = similarity * 1.3 + entity_boost * 0.5 + rare_bonus
        elif has_support_a or has_support_b:
            link_type = "supportive"
            strength = similarity * 1.1 + entity_boost * 0.4 + rare_bonus
        elif similarity > 0.12 or entity_boost > 0.10:
            link_type = "semantic"
            strength = similarity + entity_boost + rare_bonus
        else:
            return None

        strength = min(1.0, strength)
        if strength < 0.05:
            return None

        return Signal(
            data={
                "link_type": link_type,
                "evidence_a": text_a[:200],
                "evidence_b": text_b[:200],
                "similarity": round(similarity, 4),
                "shared_entities": sorted(w for w in shared_entities)[:10],
                "shared_keywords": sorted(words_a & words_b)[:10],
                "original_query": sig_a.data.get("original_query", ""),
            },
            signal_type=SignalType.CAUSAL,
            confidence=strength,
            source_center=self.name,
            metadata={
                "source_a_id": sig_a.signal_id,
                "source_b_id": sig_b.signal_id,
                "link_type": link_type,
                "entity_count": len(shared_entities),
            },
        )


# ──────────────────────────────────────────────────────────────────
# CENTER 5: Contradiction Detector
# Finds evidence items that conflict with each other.
# Like the anterior cingulate cortex detecting conflict.
# ──────────────────────────────────────────────────────────────────

class ContradictionDetector(ProcessingCenter):
    """
    Detects contradictory evidence items.

    Uses negation patterns, opposing statements, numerical contradictions,
    scope contradictions, and factual inconsistency detection to flag conflicts.
    """
    name = "contradiction_detector"

    NEGATION_PATTERNS: list[str] = [
        r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bneither\b",
        r"\bdoesn.t\b", r"\bdidn.t\b", r"\bwon.t\b", r"\bisn.t\b",
        r"\bwasn.t\b", r"\bcan.t\b", r"\bcouldn.t\b", r"\bshouldn.t\b",
        r"\bfail", r"\bdenied?\b", r"\brefuted?\b", r"\bdebunk",
        r"\bunable\b", r"\blacks?\b", r"\babsent\b", r"\bwithout\b",
        r"\bfails?\s+to\b", r"\black\s+of\b", r"\bno\s+evidence\b",
    ]

    OPPOSING_PAIRS: list[tuple[str, str]] = [
        ("increase", "decrease"), ("positive", "negative"),
        ("true", "false"), ("proven", "disproven"),
        ("confirm", "deny"), ("support", "oppose"),
        ("beneficial", "harmful"), ("safe", "dangerous"),
        ("effective", "ineffective"), ("rise", "fall"),
        ("grow", "shrink"), ("improve", "worsen"),
        ("allow", "prohibit"), ("enable", "prevent"),
        ("higher", "lower"), ("better", "worse"),
        ("more", "less"), ("majority", "minority"),
        ("all", "none"), ("always", "never"),
        ("every", "no"), ("complete", "absent"),
        ("correct", "incorrect"), ("accurate", "inaccurate"),
        ("legal", "illegal"), ("moral", "immoral"),
        ("healthy", "unhealthy"), ("progress", "regress"),
        ("success", "failure"), ("causes", "prevents"),
        ("significant", "insignificant"),
    ]

    # Scope qualifiers that indicate universal vs specific claims
    UNIVERSAL_QUALIFIERS: list[str] = [
        "all", "every", "always", "never", "none", "no", "universal",
        "entire", "whole", "completely", "totally", "absolutely",
    ]

    SPECIFIC_QUALIFIERS: list[str] = [
        "some", "most", "often", "sometimes", "usually", "typically",
        "generally", "many", "several", "a few", "rarely",
    ]

    def process(self, signals: list[Signal]) -> list[Signal]:
        evidence_signals = self._filter_signals(signals, SignalType.EVIDENCE)
        if len(evidence_signals) < 2:
            return []

        contradictions: list[Signal] = []
        for i, sig_a in enumerate(evidence_signals):
            for sig_b in evidence_signals[i + 1:]:
                conflict = self._detect_conflict(sig_a, sig_b)
                if conflict:
                    contradictions.append(conflict)

        logger.debug(f"ContradictionDetector: found {len(contradictions)} contradictions from {len(evidence_signals)} evidence pairs")
        return self._stamp(contradictions)

    def _detect_conflict(self, sig_a: Signal, sig_b: Signal) -> Signal | None:
        """Detect if two evidence items contradict each other using multiple methods."""
        text_a = sig_a.data.get("evidence_text", "").lower()
        text_b = sig_b.data.get("evidence_text", "").lower()

        if not text_a or not text_b:
            return None

        stop = {"that", "this", "with", "from", "have", "been", "they", "their", "than", "also", "just"}
        words_a = set(re.findall(r'\b[a-z]{4,}\b', text_a)) - stop
        words_b = set(re.findall(r'\b[a-z]{4,}\b', text_b)) - stop
        overlap = words_a & words_b

        # Must share some context to contradict
        if len(overlap) < 2:
            return None

        conflict_type = None
        strength = 0.0

        # ── Method 1: Negation asymmetry ──
        neg_a = sum(1 for m in self.NEGATION_PATTERNS if re.search(m, text_a))
        neg_b = sum(1 for m in self.NEGATION_PATTERNS if re.search(m, text_b))
        negation_asymmetric = (neg_a > 0 and neg_b == 0) or (neg_b > 0 and neg_a == 0)

        # ── Method 2: Opposing word pairs ──
        opposing = False
        for pair_a, pair_b in self.OPPOSING_PAIRS:
            if (pair_a in text_a and pair_b in text_b) or (pair_b in text_a and pair_a in text_b):
                opposing = True
                break

        # ── Method 3: Numerical contradictions ──
        # "X increased by 50%" vs "X decreased by 20%"
        numbers_a = re.findall(r'\b(\d+(?:\.\d+)?)\s*%?\b', text_a)
        numbers_b = re.findall(r'\b(\d+(?:\.\d+)?)\s*%?\b', text_b)
        numerical_conflict = False
        if numbers_a and numbers_b and overlap:
            # Same topic (overlap) but different numbers + opposing direction words
            direction_words_a = set(re.findall(r'\b(increase|rise|grow|up|gain|more|higher|larger)\b', text_a))
            direction_words_b = set(re.findall(r'\b(decrease|fall|shrink|down|loss|less|lower|smaller)\b', text_b))
            if direction_words_a and direction_words_b:
                numerical_conflict = True

        # ── Method 4: Scope contradictions ──
        has_universal_a = any(q in text_a for q in self.UNIVERSAL_QUALIFIERS)
        has_universal_b = any(q in text_b for q in self.UNIVERSAL_QUALIFIERS)
        has_specific_a = any(q in text_a for q in self.SPECIFIC_QUALIFIERS)
        has_specific_b = any(q in text_b for q in self.SPECIFIC_QUALIFIERS)
        scope_conflict = (has_universal_a and has_specific_b) or (has_universal_b and has_specific_a)

        # Determine the conflict type and compute strength
        if negation_asymmetric and opposing:
            conflict_type = "direct_opposition"
            strength = 0.8
        elif numerical_conflict:
            conflict_type = "numerical"
            strength = 0.7
        elif negation_asymmetric:
            conflict_type = "negation"
            strength = 0.6
        elif opposing:
            conflict_type = "opposing"
            strength = 0.5
        elif scope_conflict:
            conflict_type = "scope"
            strength = 0.35
        else:
            return None

        # Boost strength based on overlap (more shared context = stronger contradiction)
        overlap_ratio = len(overlap) / min(len(words_a), len(words_b)) if min(len(words_a), len(words_b)) > 0 else 0
        strength = min(1.0, strength + overlap_ratio * 0.2)

        if strength < 0.2:
            return None

        return Signal(
            data={
                "evidence_a": text_a[:200],
                "evidence_b": text_b[:200],
                "conflict_type": conflict_type,
                "shared_context": sorted(overlap)[:10],
                "original_query": sig_a.data.get("original_query", ""),
            },
            signal_type=SignalType.CONTRADICTION,
            confidence=strength,
            source_center=self.name,
            metadata={
                "source_a_id": sig_a.signal_id,
                "source_b_id": sig_b.signal_id,
                "conflict_type": conflict_type,
            },
        )


# ──────────────────────────────────────────────────────────────────
# CENTER 6: Explanation Narrator (stub — implemented in narrator.py)
# Translates the reasoning trace into natural language.
# Like Broca's area converting thoughts into speech.
# ──────────────────────────────────────────────────────────────────

class ExplanationBuilder(ProcessingCenter):
    """
    Collects all processed signals and builds a structured
    explanation payload for the Narrator.

    This is the bridge between computation and language.
    """
    name = "explanation_builder"

    def process(self, signals: list[Signal]) -> list[Signal]:
        # Gather all signal types into a structured explanation payload
        evidence = [s for s in signals if s.signal_type == SignalType.EVIDENCE]
        credibility = [s for s in signals if s.signal_type == SignalType.CREDIBILITY]
        temporal = [s for s in signals if s.signal_type == SignalType.TEMPORAL]
        causal = [s for s in signals if s.signal_type == SignalType.CAUSAL]
        contradictions = [s for s in signals if s.signal_type == SignalType.CONTRADICTION]
        consensus = [s for s in signals if s.signal_type == SignalType.CONSENSUS]

        explanation_data = {
            "evidence_count": len(evidence),
            "evidence_items": [
                {
                    "text": s.data.get("evidence_text", "")[:300],
                    "confidence": round(s.confidence, 3),
                    "source": s.data.get("source", "unknown"),
                }
                for s in sorted(evidence, key=lambda x: x.confidence, reverse=True)[:10]
            ],
            "credibility_summary": {
                "avg_score": round(
                    sum(s.confidence for s in credibility) / len(credibility), 3
                ) if credibility else 0.0,
                "high_trust_count": sum(1 for s in credibility if s.confidence > 0.7),
            },
            "temporal_summary": {
                "has_recent": any(
                    s.data.get("date_relevance", 0) > 0.8 for s in temporal
                ),
                "date_range": self._get_date_range(temporal),
            },
            "causal_links": [
                {
                    "type": s.data.get("link_type", "unknown"),
                    "strength": round(s.confidence, 3),
                    "shared": s.data.get("shared_entities", [])[:5],
                }
                for s in sorted(causal, key=lambda x: x.confidence, reverse=True)[:5]
            ],
            "contradictions_found": len(contradictions),
            "contradiction_details": [
                {
                    "type": s.data.get("conflict_type", "unknown"),
                    "strength": round(s.confidence, 3),
                }
                for s in contradictions[:5]
            ],
            "consensus_decision": consensus[0].data if consensus else None,
        }

        return [Signal(
            data=explanation_data,
            signal_type=SignalType.EXPLANATION,
            confidence=consensus[0].confidence if consensus else 0.5,
            source_center=self.name,
        )]

    def _get_date_range(self, signals: list) -> str:
        """Extract a human-readable date range from temporal signals."""
        if not signals:
            return "no dates found"
        dates = []
        for s in signals:
            dr = s.data.get("date_relevance", 0)
            if dr > 0.8:
                dates.append("recent")
            elif dr > 0.5:
                dates.append("moderate")
            elif dr > 0:
                dates.append("older")
        if not dates:
            return "undated"
        return ", ".join(sorted(set(dates)))
