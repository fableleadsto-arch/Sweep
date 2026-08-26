"""
Named Entity Recognition — spaCy NER with structured output.

Architecture:
    Text Input
        ↓
    [spaCy en_core_web_sm — pre-trained NER]
        ↓
    Entity (text, label, confidence, start, end)
        ↓
    Aggregated result by label

spaCy model lazy-loaded on first use.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            try:
                logger.warning("en_core_web_sm not found, downloading...")
                from spacy.cli import download
                download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.error(f"Failed to load spaCy NER: {e}")
                _nlp = None
    return _nlp


@dataclass
class Entity:
    text: str
    label: str
    confidence: float
    start: int
    end: int
    kb_id: str = ""


@dataclass
class NERResult:
    text: str
    entities: list[Entity]
    by_label: dict[str, list[Entity]] = field(default_factory=dict)
    latency_ms: float = 0.0
    backend: str = "spacy"


class NEREngine:
    def __init__(self):
        self._backend = "spacy"

    @property
    def backend(self) -> str:
        return self._backend

    def extract(self, text: str) -> NERResult:
        t0 = time.perf_counter()
        nlp = _get_nlp()
        if nlp is None:
            return NERResult(text=text, entities=[], latency_ms=(time.perf_counter() - t0) * 1000,
                             backend="none")

        doc = nlp(text)
        entities = []
        by_label: dict[str, list[Entity]] = {}

        for ent in doc.ents:
            e = Entity(
                text=ent.text,
                label=ent.label_,
                confidence=0.9,
                start=ent.start_char,
                end=ent.end_char,
                kb_id=getattr(ent, 'kb_id_', ''),
            )
            entities.append(e)
            by_label.setdefault(ent.label_, []).append(e)

        return NERResult(
            text=text,
            entities=entities,
            by_label=by_label,
            latency_ms=(time.perf_counter() - t0) * 1000,
            backend=self._backend,
        )

    def extract_people(self, text: str) -> list[Entity]:
        result = self.extract(text)
        return result.by_label.get("PERSON", [])

    def extract_orgs(self, text: str) -> list[Entity]:
        result = self.extract(text)
        return result.by_label.get("ORG", [])

    def extract_locations(self, text: str) -> list[Entity]:
        result = self.extract(text)
        locs = []
        for label in ("GPE", "LOC", "FAC"):
            locs.extend(result.by_label.get(label, []))
        return locs

    def extract_dates(self, text: str) -> list[Entity]:
        result = self.extract(text)
        return result.by_label.get("DATE", [])

    def extract_entities_by_label(self, text: str, label: str) -> list[Entity]:
        result = self.extract(text)
        return result.by_label.get(label, [])


_default_ner: NEREngine | None = None


def get_ner_engine() -> NEREngine:
    global _default_ner
    if _default_ner is None:
        _default_ner = NEREngine()
    return _default_ner
