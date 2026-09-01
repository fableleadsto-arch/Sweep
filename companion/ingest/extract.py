"""Deterministic knowledge extraction from ingested documents.

This is deliberately **rule-based** — it runs before any LLM call so the
pipeline stays free/cheap (the stronger model is only invoked later, if at
all). It extracts:

* **Entities** — capitalized proper-noun phrases plus software/library tokens.
* **Versions** — ``v1.2.3`` / ``version 5`` style version facts with context.
* **Claims** — entity → property → value triples from release, support,
  acquisition, announcement and partnership sentences.
* **Edges** — directed knowledge-graph relations derived from the same
  sentences (``Entity --released--> v1.2``).
* **Topics** — which of the source's configured topics a document covers.

Extraction never invents facts: only sentences that match a supported pattern
produce claims, and every claim carries the source URL + confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .models import (
    EntityEdge,
    IngestedDocument,
    KnowledgeClaim,
    SourceKind,
    utcnow,
)
from .text import split_sentences

# ─────────────────────────────────────────────────────────────────────────
#  Patterns
# ─────────────────────────────────────────────────────────────────────────

_VERSION_TOKEN = r"(?:v|ver\.?)?\d+(?:\.\d+){1,3}(?:[-+.][A-Za-z0-9]+)*"
VERSION_RE = re.compile(rf"\b{_VERSION_TOKEN}\b", re.IGNORECASE)

# A version that definitely reads as a software version.
_SOFT_VERSION_RE = re.compile(r"\bv?\d+\.\d+(?:\.\d+){1,2}\b")

_PROPERTY_NOUN = r"(?:support|the support|full support|experimental support)"
_SUPPORT_RE = re.compile(
    rf"(?P<subject>.{{2,60}}?)\s+(?:(?:added|gained|introduced|now has)\s+{_PROPERTY_NOUN}\s+for|now\s+supports|supports)\s+(?P<object>.{{2,60}}?)\.?$",
    re.IGNORECASE,
)
_RELEASE_RE = re.compile(
    r"(?P<subject>.{2,60}?)\s+(?:released|has released|announced|launched)\s+"
    rf"(?:version\s+)?(?P<version>{_VERSION_TOKEN})\.?$",
    re.IGNORECASE,
)
# "OpenAI released version 2.0.5 of Whisper API today." — version mid-sentence
# with the subject as a leading capitalized phrase.
_RELEASE_OF_RE = re.compile(
    r"(?P<subject>[A-Z][A-Za-z0-9]+(?:[\s&'-]+[A-Z][A-Za-z0-9]+)*)"
    r"\s+(?:released|has released|announced|launched)\s+version\s+"
    rf"(?P<version>{_VERSION_TOKEN})",
    re.IGNORECASE,
)
_ACQUIRED_RE = re.compile(
    r"(?P<subject>.{2,60}?)\s+(?:has\s+)?acquired\s+(?P<object>.{2,60}?)\.?$",
    re.IGNORECASE,
)
_ACQUIRED_BY_RE = re.compile(
    r"(?P<object>.{2,60}?)\s+(?:was|has been)\s+acquired\s+by\s+(?P<subject>.{2,60}?)\.?$",
    re.IGNORECASE,
)
_ANNOUNCED_RE = re.compile(
    r"(?P<subject>.{2,60}?)\s+(?:announced|has announced)\s+(?P<object>.{2,60}?)\.?$",
    re.IGNORECASE,
)
_LATEST_VERSION_RE = re.compile(
    r"(?:latest|current|most recent)\s+version\s+of\s+(?P<subject>.{2,60}?)"
    rf"\s+(?:is|reached|stands at)\s+(?P<version>{_VERSION_TOKEN})\.?$",
    re.IGNORECASE,
)
_AT_VERSION_RE = re.compile(
    rf"(?P<subject>.{{2,60}}?)\s+(?:is|sits)\s+(?:now\s+)?at\s+(?:version\s+)?(?P<version>{_VERSION_TOKEN})\.?$",
    re.IGNORECASE,
)
_PARTNER_RE = re.compile(
    r"(?P<subject>.{2,60}?)\s+(?:partnered|is partnering)\s+with\s+(?P<object>.{2,60}?)\.?$",
    re.IGNORECASE,
)
_USE_RE = re.compile(
    r"(?P<subject>.{2,60}?)\s+uses?\s+(?P<object>.{2,60}?)\.?$",
    re.IGNORECASE,
)

# Common nouns that are never entities.
_ENTITY_STOP = {
    "the", "a", "an", "this", "that", "these", "those", "it", "they", "we",
    "you", "i", "he", "she", "new", "latest", "next", "first", "last",
    "previous", "version", "versions", "release", "releases", "update",
    "updates", "support", "feature", "features", "bug", "fix", "fixes",
    "api", "sdk", "tool", "tools", "library", "libraries", "package",
    "packages", "framework", "model", "models", "researcher", "researchers",
    "team", "company", "companies", "users", "developers", "app", "application",
    "server", "client", "service", "web", "platform", "announcement", "year",
    "today", "yesterday", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october", "november",
    "december",
}

_CAPITALIZED_PHRASE = re.compile(
    r"(?<![\w.])((?:[A-Z][a-zA-Z0-9]+)(?:\s+(?:[A-Z][a-zA-Z0-9]+|\d+[A-Za-z]*)){0,3})(?!\w)",
)
# Lowercase software/library tokens that only count when paired with a version.
_SOFTWARE_TOKEN = re.compile(r"\b(?:flask|django|react|rust|cargo|pip|npm|torch|tensorflow|transformers|scikit-learn|pandas|numpy|sympy|matplotlib|openai|gemini|kubernetes|docker|terraform|fastapi|node\.js|typescript|python|golang|postgresql|redis)\b", re.IGNORECASE)

_PERSON_TITLES = ("Dr.", "Prof.", "Mr.", "Ms.", "Mrs.")
_ORG_HINTS = ("Inc.", "Ltd.", "LLC", "Corp.", "GmbH", "Labs", "Lab", "University", "Foundation", "Institute", "Group", "Association")


# ─────────────────────────────────────────────────────────────────────────
#  Output
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ExtractionResult:
    claims: list[KnowledgeClaim] = field(default_factory=list)
    entities: list[tuple[str, str, int]] = field(default_factory=list)  # (name, kind, count)
    edges: list[EntityEdge] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────
#  Entity extraction
# ─────────────────────────────────────────────────────────────────────────


def _normalize_entity(name: str) -> str:
    return " ".join(name.split()).strip(" ,;:.")


def _entity_kind(name: str, sentence: str = "") -> str:
    lowered = name.lower()
    if any(hint in lowered for hint in _ORG_HINTS):
        return "org"
    if any(t in name for t in _PERSON_TITLES):
        return "person"
    if _SOFT_VERSION_RE.search(name) or " v" in lowered:
        return "product"
    if sentence and _USE_RE.search(sentence) and lowered in sentence.lower():
        return "product"
    if re.match(r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3}$", name):
        return "org"
    return "product"


def extract_entities(text: str) -> list[tuple[str, str, int]]:
    """Count capitalized proper-noun phrases (deduped, with kinds)."""
    counts: dict[str, tuple[str, int]] = {}
    sentences = split_sentences(text)
    for sentence in sentences:
        for match in _CAPITALIZED_PHRASE.finditer(sentence):
            name = _normalize_entity(match.group(1))
            if not name or len(name) > 48:
                continue
            lowered = name.lower()
            if lowered in _ENTITY_STOP or any(w in lowered for w in ("the ", "of the")):
                continue
            if " " not in name:
                is_known = bool(
                    _SOFTWARE_TOKEN.fullmatch(lowered)
                    or any(h in name for h in _ORG_HINTS)
                    or any(t in name for t in _PERSON_TITLES)
                )
                if not is_known:
                    # Weak single-word signal — only keep it once it recurs.
                    kind, prev_count = counts.get(lowered, (_entity_kind(name, sentence), 0))
                    counts[lowered] = (kind, prev_count + 1)
                    if prev_count + 1 < 3:
                        continue
            kind = _entity_kind(name, sentence)
            prev_kind, prev_count = counts.get(lowered, (kind, 0))
            counts[lowered] = (prev_kind or kind, prev_count + 1)
    return [(name, kind, count) for (name, (kind, count)) in sorted(
        ((n, c) for n, c in counts.items()),
        key=lambda item: item[1][1],
        reverse=True,
    )]


def extract_versions(text: str) -> list[str]:
    """Unique version strings mentioned in the text (stable order)."""
    seen: set[str] = set()
    versions: list[str] = []
    for match in VERSION_RE.finditer(text):
        token = match.group(0).strip()
        if token.lower() not in seen:
            seen.add(token.lower())
            versions.append(token)
    return versions


# ─────────────────────────────────────────────────────────────────────────
#  Claim extraction
# ─────────────────────────────────────────────────────────────────────────


def _clean(value: str) -> str:
    return " ".join(value.split()).strip(" ,;:.")


def _claim(
    *,
    entity: str,
    prop: str,
    value: str,
    subject: str,
    document_id: str,
    source_id: str,
    source_url: str,
    confidence: float,
    authority: float,
) -> KnowledgeClaim:
    import uuid

    return KnowledgeClaim(
        id=uuid.uuid4().hex,
        entity=_normalize_entity(entity),
        property=prop,
        value=_clean(value),
        subject=_clean(subject),
        document_id=document_id,
        source_id=source_id,
        source_url=source_url,
        collected_at=utcnow(),
        confidence=confidence,
        authority=authority,
    )


def extract_claims(
    document: IngestedDocument,
    *,
    source_authority: float = 0.5,
    confidence: float = 0.5,
) -> list[KnowledgeClaim]:
    """Rule-based entity→property→value triples from an ingested document."""
    claims: list[KnowledgeClaim] = []
    seen: set[tuple[str, str, str, str]] = set()
    sentences = split_sentences(document.content)
    for sentence in sentences:
        candidates: list[tuple[str, str, str, str]] = []

        m = _LATEST_VERSION_RE.search(sentence)
        if m:
            candidates.append((m.group("subject"), "latest_version", m.group("version"), m.group("subject")))
        m = _AT_VERSION_RE.search(sentence)
        if m:
            candidates.append((m.group("subject"), "current_version", m.group("version"), m.group("subject")))
        m = _RELEASE_RE.search(sentence)
        if m:
            candidates.append((m.group("subject"), "released_version", m.group("version"), m.group("subject")))
        m = _RELEASE_OF_RE.search(sentence)
        if m:
            candidates.append((m.group("subject"), "released_version", m.group("version"), m.group("subject")))
        m = _SUPPORT_RE.search(sentence)
        if m:
            candidates.append((m.group("subject"), "supports", m.group("object"), m.group("subject")))
        m = _ACQUIRED_RE.search(sentence)
        if m:
            candidates.append((m.group("subject"), "acquired", m.group("object"), m.group("subject")))
        m = _ACQUIRED_BY_RE.search(sentence)
        if m:
            candidates.append((m.group("subject"), "acquired", m.group("object"), m.group("subject")))
        m = _PARTNER_RE.search(sentence)
        if m:
            candidates.append((m.group("subject"), "partners_with", m.group("object"), m.group("subject")))
        m = _ANNOUNCED_RE.search(sentence)
        if m:
            candidates.append((m.group("subject"), "announced", m.group("object"), m.group("subject")))
        m = _USE_RE.search(sentence)
        if m and not any(c[1] == "supports" for c in candidates):
            candidates.append((m.group("subject"), "uses", m.group("object"), m.group("subject")))

        for entity, prop, value, subject in candidates:
            entity = _normalize_entity(entity)
            if not entity or len(entity) < 2 or entity.lower() in _ENTITY_STOP:
                continue
            if len(value) > 80:
                continue
            key = (entity.lower(), prop, value.lower(), subject.lower())
            if key in seen:
                continue
            seen.add(key)
            claims.append(
                _claim(
                    entity=entity,
                    prop=prop,
                    value=value,
                    subject=subject,
                    document_id=document.id,
                    source_id=document.source_id,
                    source_url=document.url,
                    confidence=confidence,
                    authority=source_authority,
                )
            )
    return claims


# ─────────────────────────────────────────────────────────────────────────
#  Edges / topics
# ─────────────────────────────────────────────────────────────────────────


def extract_edges(
    document: IngestedDocument,
    *,
    confidence: float = 0.5,
) -> list[EntityEdge]:
    """Directed relations from the same sentence patterns as claims."""
    import uuid

    edges: list[EntityEdge] = []
    for claim in extract_claims(document, confidence=confidence):
        relation = claim.property
        target = claim.value if relation in ("released_version", "current_version", "latest_version") else claim.value
        if relation in ("released_version", "current_version", "latest_version"):
            edges.append(
                EntityEdge(
                    id=uuid.uuid4().hex,
                    from_entity=claim.entity,
                    to_entity=f"{claim.entity} {claim.value}",
                    relation=relation,
                    source_url=claim.source_url,
                    confidence=confidence,
                )
            )
        elif relation in ("acquired", "partners_with"):
            edges.append(
                EntityEdge(
                    id=uuid.uuid4().hex,
                    from_entity=claim.entity,
                    to_entity=claim.value,
                    relation=relation,
                    source_url=claim.source_url,
                    confidence=confidence,
                )
            )
        elif relation == "supports" and _looks_like_product(claim.value):
            edges.append(
                EntityEdge(
                    id=uuid.uuid4().hex,
                    from_entity=claim.entity,
                    to_entity=claim.value,
                    relation="supports",
                    source_url=claim.source_url,
                    confidence=confidence,
                )
            )
    return edges


def _looks_like_product(value: str) -> bool:
    return bool(_SOFTWARE_TOKEN.search(value) or re.match(r"^[A-Z]", value))


def extract_topics(text: str, topics: list[str]) -> list[str]:
    """Which of the configured topics the document actually mentions."""
    if not topics:
        return []
    lowered = text.lower()
    return [t for t in topics if t and t.lower() in lowered]
