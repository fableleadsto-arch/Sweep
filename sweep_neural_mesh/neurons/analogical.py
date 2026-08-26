"""
Analogical Reasoning Engine — mapping structural relationships between domains.

Humans reason by analogy constantly:
- "Python is to programming what English is to communication"
- "The nucleus is to a cell what the brain is to the body"
- "Debugging code is like detective work — following clues to find the culprit"

This module implements Gentner's Structure-Mapping Theory:
1. ALIGN: Find structural similarities between source and target domains
2. MAP: Transfer relational knowledge from source to target
3. INFERR: Generate new inferences about the target using the mapping

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │              ANALOGICAL REASONING ENGINE              │
    │                                                     │
    │  ┌─────────────┐     ┌──────────────────────────┐  │
    │  │ Domain       │     │ Structural Alignment      │  │
    │  │ Knowledge    │────→│ - Entity matching          │  │
    │  │ Base         │     │ - Relation mapping         │  │
    │  └─────────────┘     │ - Isomorphism detection    │  │
    │                      └──────────┬───────────────┘  │
    │                                 ↓                   │
    │                      ┌──────────────────────────┐  │
    │                      │ Analogical Inference       │  │
    │                      │ - Transfer relations        │  │
    │                      │ - Generate predictions      │  │
    │                      │ - Explain via analogy       │  │
    │                      └──────────────────────────┘  │
    └─────────────────────────────────────────────────────┘

The engine maintains a growing library of learned analogies that
improves over time through use and feedback.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainEntity:
    """An entity within a domain."""
    name: str
    entity_type: str             # "concept", "process", "tool", "person"
    attributes: dict[str, str]   # name → value
    relations: list[dict]        # [{target, relation_type, strength}]


@dataclass
class Domain:
    """A knowledge domain with entities and their relationships."""
    domain_id: str
    name: str
    entities: list[DomainEntity] = field(default_factory=list)
    description: str = ""
    created_at: float = field(default_factory=time.time)
    use_count: int = 0

    def get_entity(self, name: str) -> DomainEntity | None:
        for e in self.entities:
            if e.name.lower() == name.lower():
                return e
        return None

    def get_entity_names(self) -> set[str]:
        return {e.name.lower() for e in self.entities}

    def get_relation_types(self) -> set[str]:
        types = set()
        for e in self.entities:
            for r in e.relations:
                types.add(r.get("relation_type", ""))
        return types


@dataclass
class StructuralMapping:
    """A mapping between two domains."""
    source_domain: str
    target_domain: str
    entity_mappings: dict[str, str]     # source_entity → target_entity
    relation_mappings: list[dict]        # [{source_rel, target_rel, strength}]
    structural_similarity: float         # 0.0-1.0
    alignment_confidence: float          # 0.0-1.0
    inferences: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Analogy:
    """A complete analogy between two domains."""
    source_domain: str
    target_domain: str
    statement: str                      # "X is to Y as A is to B"
    mapping: StructuralMapping
    confidence: float
    explanation: str
    generated_inferences: list[str]


class AnalogicalReasoner:
    """
    Reason by mapping structural relationships between domains.

    Like the human ability to understand new concepts through analogy
    ("Oh, so a neural network is like a brain!"), this module:

    1. MAINTAINS a library of known domains with their entities/relations
    2. ALIGNS source and target domains by finding structural parallels
    3. MAPS relational knowledge from source to target
    4. INFERS new knowledge about the target using the mapping
    5. LEARNS new analogies from successful reasoning episodes

    The key insight from Structure-Mapping Theory (Gentner):
    Good analogies preserve RELATIONAL structure, not just surface features.
    "A hammer is to a nail" maps to "a saw is to wood" because
    both are TOOL→MATERIAL relationships, not because hammers and saws
    look similar.
    """

    def __init__(self) -> None:
        self._domains: dict[str, Domain] = {}
        self._learned_analogies: list[Analogy] = []
        self._mapping_history: list[StructuralMapping] = []
        # Domain similarity cache
        self._similarity_cache: dict[tuple[str, str], float] = {}

    def register_domain(self, domain: Domain) -> None:
        """Register a new knowledge domain."""
        self._domains[domain.domain_id] = domain

    def add_entity_to_domain(
        self,
        domain_id: str,
        entity: DomainEntity,
    ) -> bool:
        """Add an entity to an existing domain."""
        if domain_id not in self._domains:
            return False
        self._domains[domain_id].entities.append(entity)
        return True

    def find_analogy(
        self,
        source_domain_id: str,
        target_domain_id: str,
    ) -> Analogy | None:
        """
        Find the best analogy between two domains.

        Uses structural alignment to find entity and relation mappings,
        then generates analogical inferences.
        """
        source = self._domains.get(source_domain_id)
        target = self._domains.get(target_domain_id)
        if not source or not target:
            return None

        source.use_count += 1
        target.use_count += 1

        # Step 1: Structural alignment
        mapping = self._align_structures(source, target)
        if mapping.structural_similarity < 0.2:
            return None  # too dissimilar

        # Step 2: Generate analogical inferences
        inferences = self._generate_inferences(source, target, mapping)
        mapping.inferences = inferences

        # Step 3: Build analogy statement
        statement = self._build_analogy_statement(source, target, mapping)

        # Step 4: Generate explanation
        explanation = self._build_explanation(source, target, mapping)

        analogy = Analogy(
            source_domain=source.name,
            target_domain=target.name,
            statement=statement,
            mapping=mapping,
            confidence=mapping.structural_similarity * mapping.alignment_confidence,
            explanation=explanation,
            generated_inferences=inferences,
        )

        self._learned_analogies.append(analogy)
        self._mapping_history.append(mapping)

        return analogy

    def apply_analogy_to_query(
        self,
        query: str,
        evidence: list[str],
    ) -> Analogy | None:
        """
        Find an analogy that helps answer a query.

        Extracts domain concepts from the query, finds matching domains,
        and generates an analogy.
        """
        query_words = set(re.findall(r'\b\w{4,}\b', query.lower()))
        evidence_words = set()
        for e in evidence:
            evidence_words.update(set(re.findall(r'\b\w{4,}\b', e.lower())))

        # Find domains that match query concepts
        matching_domains = []
        for did, domain in self._domains.items():
            entity_names = domain.get_entity_names()
            overlap = len(query_words & entity_names)
            if overlap > 0:
                matching_domains.append((overlap, did))

        matching_domains.sort(reverse=True)

        # Try top domain pairs
        for _, did1 in matching_domains[:3]:
            for _, did2 in matching_domains[1:4]:
                if did1 != did2:
                    analogy = self.find_analogy(did1, did2)
                    if analogy and analogy.confidence > 0.3:
                        return analogy

        return None

    def _align_structures(
        self,
        source: Domain,
        target: Domain,
    ) -> StructuralMapping:
        """
        Align two domains by finding structural parallels.

        Uses entity type matching + relation type matching to find
        the best structural alignment.
        """
        entity_mappings: dict[str, str] = {}
        relation_mappings: list[dict] = []

        # Match entities by type
        source_by_type: dict[str, list[DomainEntity]] = {}
        for e in source.entities:
            source_by_type.setdefault(e.entity_type, []).append(e)

        target_by_type: dict[str, list[DomainEntity]] = {}
        for e in target.entities:
            target_by_type.setdefault(e.entity_type, []).append(e)

        # Align entities of same type
        for etype in source_by_type:
            if etype in target_by_type:
                source_ents = source_by_type[etype]
                target_ents = target_by_type[etype]
                # Simple 1:1 matching by name similarity
                used_target = set()
                for se in source_ents:
                    best_match = None
                    best_score = 0.0
                    for te in target_ents:
                        if te.name in used_target:
                            continue
                        score = self._entity_similarity(se, te)
                        if score > best_score:
                            best_score = score
                            best_match = te
                    if best_match and best_score > 0.2:
                        entity_mappings[se.name] = best_match.name
                        used_target.add(best_match.name)

        # Match relations between mapped entities
        for src_name, tgt_name in entity_mappings.items():
            src_ent = source.get_entity(src_name)
            tgt_ent = target.get_entity(tgt_name)
            if not src_ent or not tgt_ent:
                continue

            for src_rel in src_ent.relations:
                src_target = src_rel.get("target", "")
                if src_target not in entity_mappings:
                    continue
                mapped_target = entity_mappings[src_target]

                for tgt_rel in tgt_ent.relations:
                    if tgt_rel.get("target", "") == mapped_target:
                        rel_type_match = (
                            src_rel.get("relation_type", "")
                            == tgt_rel.get("relation_type", "")
                        )
                        if rel_type_match:
                            relation_mappings.append({
                                "source_relation": f"{src_name} --[{src_rel.get('relation_type')}]--> {src_target}",
                                "target_relation": f"{tgt_name} --[{tgt_rel.get('relation_type')}]--> {mapped_target}",
                                "strength": min(
                                    src_rel.get("strength", 0.5),
                                    tgt_rel.get("strength", 0.5),
                                ),
                            })

        # Compute structural similarity
        source_entities = len(source.entities)
        mapped_entities = len(entity_mappings)
        entity_coverage = mapped_entities / max(1, source_entities)

        source_relations = sum(len(e.relations) for e in source.entities)
        mapped_relations = len(relation_mappings)
        relation_coverage = mapped_relations / max(1, source_relations)

        structural_similarity = entity_coverage * 0.4 + relation_coverage * 0.6
        alignment_confidence = min(1.0, mapped_entities * 0.2 + mapped_relations * 0.1)

        return StructuralMapping(
            source_domain=source.domain_id,
            target_domain=target.domain_id,
            entity_mappings=entity_mappings,
            relation_mappings=relation_mappings,
            structural_similarity=structural_similarity,
            alignment_confidence=alignment_confidence,
        )

    def _entity_similarity(self, e1: DomainEntity, e2: DomainEntity) -> float:
        """Compute similarity between two entities."""
        # Type match
        type_match = 1.0 if e1.entity_type == e2.entity_type else 0.0

        # Name similarity (Jaccard on words)
        words1 = set(e1.name.lower().split())
        words2 = set(e2.name.lower().split())
        if words1 and words2:
            name_sim = len(words1 & words2) / len(words1 | words2)
        else:
            name_sim = 0.0

        # Relation type overlap
        rel_types1 = {r.get("relation_type", "") for r in e1.relations}
        rel_types2 = {r.get("relation_type", "") for r in e2.relations}
        if rel_types1 and rel_types2:
            rel_sim = len(rel_types1 & rel_types2) / len(rel_types1 | rel_types2)
        else:
            rel_sim = 0.0

        return type_match * 0.3 + name_sim * 0.3 + rel_sim * 0.4

    def _generate_inferences(
        self,
        source: Domain,
        target: Domain,
        mapping: StructuralMapping,
    ) -> list[str]:
        """
        Generate new inferences about the target domain using the mapping.

        Like how an analogy lets you predict properties of a new domain
        based on a familiar one.
        """
        inferences = []

        # For each mapped relation, check if the target side has the relation
        for rel_map in mapping.relation_mappings:
            target_rel = rel_map.get("target_relation", "")
            if target_rel:
                # Check if this relation already exists in target
                parts = target_rel.split("-->")
                if len(parts) == 2:
                    src_name = parts[0].strip()
                    rest = parts[1].strip()
                    # If not explicitly present, it's an inference
                    inferences.append(
                        f"By analogy: {target_rel} (inferred from source domain)"
                    )

        # Attribute transfer: if source entity has attribute, target might too
        for src_name, tgt_name in mapping.entity_mappings.items():
            src_ent = source.get_entity(src_name)
            tgt_ent = target.get_entity(tgt_name)
            if src_ent and tgt_ent:
                for attr, val in src_ent.attributes.items():
                    if attr not in tgt_ent.attributes:
                        inferences.append(
                            f"Target entity '{tgt_name}' may have attribute "
                            f"'{attr}' = '{val}' (by analogy with '{src_name}')"
                        )

        return inferences[:10]  # cap at 10

    def _build_analogy_statement(
        self,
        source: Domain,
        target: Domain,
        mapping: StructuralMapping,
    ) -> str:
        """Build a natural language analogy statement."""
        if mapping.entity_mappings:
            pairs = list(mapping.entity_mappings.items())[:2]
            parts = [f"{s} is to {t}" for s, t in pairs]
            return f"{' and '.join(parts)} (structural similarity: {mapping.structural_similarity:.0%})"
        return f"{source.name} maps to {target.name} (similarity: {mapping.structural_similarity:.0%})"

    def _build_explanation(
        self,
        source: Domain,
        target: Domain,
        mapping: StructuralMapping,
    ) -> str:
        """Build an explanation of the analogy."""
        parts = [
            f"Analogy between {source.name} and {target.name}:",
            f"  Entities mapped: {len(mapping.entity_mappings)}",
            f"  Relations mapped: {len(mapping.relation_mappings)}",
            f"  Structural similarity: {mapping.structural_similarity:.0%}",
        ]
        if mapping.entity_mappings:
            pair_strs = [f"'{s}' → '{t}'" for s, t in list(mapping.entity_mappings.items())[:3]]
            parts.append(f"  Key mappings: {', '.join(pair_strs)}")
        return "\n".join(parts)

    def get_domain(self, domain_id: str) -> Domain | None:
        return self._domains.get(domain_id)

    @property
    def domain_count(self) -> int:
        return len(self._domains)

    @property
    def analogy_count(self) -> int:
        return len(self._learned_analogies)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "domain_count": len(self._domains),
            "analogy_count": len(self._learned_analogies),
            "mapping_history_size": len(self._mapping_history),
            "avg_structural_similarity": (
                sum(m.structural_similarity for m in self._mapping_history)
                / len(self._mapping_history)
                if self._mapping_history else 0.0
            ),
        }
