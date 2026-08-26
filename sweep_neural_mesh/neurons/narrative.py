"""
Narrative Coherence Engine — structuring evidence into stories.

Humans understand the world through NARRATIVES:
- "Company X launched product Y, which failed because of Z"
- "The developer found a bug, traced it to memory leak, fixed it"
- "The research showed A, which led to B, which caused C"

Evidence isn't just isolated facts — it's a STORY with:
- Characters (entities)
- Goals (what they're trying to do)
- Obstacles (what's in the way)
- Outcomes (what happened)
- Causal chains linking them

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │           NARRATIVE COHERENCE ENGINE                  │
    │                                                     │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Story Arc Detection                          │  │
    │  │  - Setup → Complication → Resolution           │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Entity Tracking                              │  │
    │  │  - Characters, goals, obstacles               │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Coherence Assessment                         │  │
    │  │  - Missing pieces, narrative fallacies         │  │
    │  └──────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class NarrativeEntity:
    """An entity in the narrative (character)."""
    name: str
    entity_type: str             # "person", "organization", "system", "concept"
    role: str                    # "protagonist", "antagonist", "helper", "observer"
    goals: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


@dataclass
class StoryArc:
    """A narrative arc detected in evidence."""
    arc_id: str
    phase: str                   # "setup", "complication", "resolution"
    events: list[str]            # chronological events
    entities: list[NarrativeEntity]
    causal_chain: list[str]      # cause → effect links
    outcome: str                 # what happened in the end
    coherence_score: float       # 0.0-1.0
    missing_elements: list[str]  # what's missing from the story


@dataclass
class NarrativeAssessment:
    """Assessment of narrative coherence in evidence."""
    story_arcs: list[StoryArc]
    overall_coherence: float     # 0.0-1.0
    narrative_fallacies: list[str]  # detected reasoning fallacies
    missing_context: list[str]   # what's missing for a complete story
    recommendations: list[str]   # what to do about it


class NarrativeEngine:
    """
    Structure evidence into coherent narratives.

    Like the human ability to understand events as stories, this module:

    1. DETECTS story arcs in evidence (setup → complication → resolution)
    2. TRACKS entities and their roles (who does what)
    3. IDENTIFIES causal chains (what led to what)
    4. ASSESSES coherence (does the story make sense?)
    5. DETECTS missing elements (what's left out)
    6. IDENTIFIES narrative fallacies (post-hoc reasoning, etc.)

    The key insight: evidence that forms a coherent narrative is more
    convincing than isolated facts, even if the individual facts are
    the same. Stories create meaning through structure.
    """

    def __init__(self) -> None:
        self._assessment_history: list[NarrativeAssessment] = []

    def assess_narrative(
        self,
        evidence: list[str],
        query: str = "",
    ) -> NarrativeAssessment:
        """
        Assess the narrative coherence of evidence.
        """
        # Step 1: Extract entities
        entities = self._extract_entities(evidence)

        # Step 2: Detect story arcs
        arcs = self._detect_arcs(evidence, entities)

        # Step 3: Assess coherence
        coherence = self._assess_coherence(arcs, evidence)

        # Step 4: Detect narrative fallacies
        fallacies = self._detect_fallacies(evidence)

        # Step 5: Identify missing context
        missing = self._identify_missing(arcs, evidence)

        # Step 6: Generate recommendations
        recommendations = self._generate_recommendations(
            coherence, fallacies, missing, arcs
        )

        assessment = NarrativeAssessment(
            story_arcs=arcs,
            overall_coherence=coherence,
            narrative_fallacies=fallacies,
            missing_context=missing,
            recommendations=recommendations,
        )

        self._assessment_history.append(assessment)
        return assessment

    def _extract_entities(self, evidence: list[str]) -> list[NarrativeEntity]:
        """Extract narrative entities from evidence."""
        entities: dict[str, NarrativeEntity] = {}

        for text in evidence:
            # Simple entity extraction: capitalized words, common patterns
            # Person patterns
            person_matches = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', text)
            for name in person_matches:
                if name not in entities:
                    entities[name] = NarrativeEntity(
                        name=name, entity_type="person", role="observer"
                    )

            # Organization patterns
            org_patterns = [
                r'\b(Google|Microsoft|Apple|Amazon|Meta|OpenAI|Anthropic)\b',
                r'\b(\w+ Inc\.?|\w+ Corp\.?|\w+ LLC)\b',
            ]
            for pattern in org_patterns:
                for match in re.findall(pattern, text):
                    if match not in entities:
                        entities[match] = NarrativeEntity(
                            name=match, entity_type="organization", role="observer"
                        )

            # System/technology patterns
            tech_matches = re.findall(r'\b(Python|JavaScript|Rust|Docker|Kubernetes|Linux|Windows)\b', text, re.IGNORECASE)
            for tech in tech_matches:
                if tech not in entities:
                    entities[tech] = NarrativeEntity(
                        name=tech, entity_type="system", role="helper"
                    )

        return list(entities.values())

    def _detect_arcs(
        self,
        evidence: list[str],
        entities: list[NarrativeEntity],
    ) -> list[StoryArc]:
        """Detect story arcs in evidence."""
        arcs: list[StoryArc] = []

        if not evidence:
            return arcs

        # Look for temporal/causal markers
        setup_markers = ["initially", "first", "before", "originally", "started"]
        complication_markers = ["however", "but", "problem", "issue", "failed", "error"]
        resolution_markers = ["therefore", "resolved", "fixed", "result", "concluded", "outcome"]

        setup_events = []
        complication_events = []
        resolution_events = []

        for text in evidence:
            text_lower = text.lower()
            if any(m in text_lower for m in setup_markers):
                setup_events.append(text[:200])
            elif any(m in text_lower for m in complication_markers):
                complication_events.append(text[:200])
            elif any(m in text_lower for m in resolution_markers):
                resolution_events.append(text[:200])

        # Create arc if we have at least 2 phases
        phases_found = sum(bool(e) for e in [setup_events, complication_events, resolution_events])
        if phases_found >= 2:
            all_events = setup_events + complication_events + resolution_events
            causal_chain = self._extract_causal_chain(all_events)

            # Determine final phase
            if resolution_events:
                phase = "resolution"
                outcome = resolution_events[-1]
            elif complication_events:
                phase = "complication"
                outcome = complication_events[-1]
            else:
                phase = "setup"
                outcome = "story still developing"

            arc = StoryArc(
                arc_id=f"arc_{len(arcs)}",
                phase=phase,
                events=all_events,
                entities=entities[:5],
                causal_chain=causal_chain,
                outcome=outcome,
                coherence_score=self._score_arc_coherence(all_events),
                missing_elements=self._find_missing_elements(
                    setup_events, complication_events, resolution_events
                ),
            )
            arcs.append(arc)
        elif evidence:
            # Single-phase narrative
            arc = StoryArc(
                arc_id=f"arc_{len(arcs)}",
                phase="setup",
                events=[e[:200] for e in evidence[:5]],
                entities=entities[:5],
                causal_chain=[],
                outcome="incomplete narrative",
                coherence_score=0.3,
                missing_elements=["resolution", "complication"],
            )
            arcs.append(arc)

        return arcs

    def _extract_causal_chain(self, events: list[str]) -> list[str]:
        """Extract causal chain from events."""
        chain = []
        causal_markers = [
            "because", "therefore", "caused", "led to", "resulted in",
            "due to", "since", "thus", "consequently",
        ]
        for event in events:
            for marker in causal_markers:
                if marker in event.lower():
                    chain.append(event[:150])
                    break
        return chain

    def _score_arc_coherence(self, events: list[str]) -> float:
        """Score the coherence of a story arc."""
        if not events:
            return 0.0

        score = 0.5  # base

        # More events = more complete story
        if len(events) > 5:
            score += 0.2
        elif len(events) > 2:
            score += 0.1

        # Causal language improves coherence
        causal_count = sum(
            1 for e in events
            for m in ["because", "therefore", "led to", "caused"]
            if m in e.lower()
        )
        score += min(0.3, causal_count * 0.1)

        return min(1.0, score)

    def _find_missing_elements(
        self,
        setup: list[str],
        complication: list[str],
        resolution: list[str],
    ) -> list[str]:
        """Find missing story elements."""
        missing = []
        if not setup:
            missing.append("setup (no background context)")
        if not complication:
            missing.append("complication (no conflict or problem)")
        if not resolution:
            missing.append("resolution (no outcome or conclusion)")
        return missing

    def _assess_coherence(
        self,
        arcs: list[StoryArc],
        evidence: list[str],
    ) -> float:
        """Assess overall narrative coherence."""
        if not arcs:
            return 0.2

        arc_scores = [a.coherence_score for a in arcs]
        avg_score = sum(arc_scores) / len(arc_scores)

        # Penalty for missing elements
        total_missing = sum(len(a.missing_elements) for a in arcs)
        penalty = min(0.3, total_missing * 0.1)

        return max(0.0, avg_score - penalty)

    def _detect_fallacies(self, evidence: list[str]) -> list[str]:
        """Detect narrative fallacies in evidence."""
        fallacies = []

        for text in evidence:
            text_lower = text.lower()

            # Post-hoc ergo propter hoc: "after X, therefore because of X"
            if re.search(r'(after|since|following).*(therefore|thus|so|consequently)', text_lower):
                fallacies.append("Possible post-hoc reasoning: temporal sequence ≠ causation")

            # Appeal to narrative: "it's a story, so it must be true"
            if re.search(r'(story|narrative|journey|path).*(proves?|shows?|demonstrates?)', text_lower):
                fallacies.append("Possible appeal to narrative: story structure ≠ truth")

            # Incomplete narrative: drawing conclusions from partial story
            if re.search(r'(clearly|obviously|undoubtedly|clearly shows)', text_lower):
                if not re.search(r'(evidence|data|research|study)', text_lower):
                    fallacies.append("Possible premature conclusion: strong claims without evidence")

        return fallacies

    def _identify_missing(
        self,
        arcs: list[StoryArc],
        evidence: list[str],
    ) -> list[str]:
        """Identify missing context for a complete narrative."""
        missing = []

        # Check for missing perspective
        all_text = " ".join(evidence).lower()
        if not re.search(r'(however|but|although|despite|on the other hand)', all_text):
            missing.append("No opposing perspective presented")

        # Check for missing data
        if not re.search(r'\d+', all_text):
            missing.append("No quantitative data provided")

        # Check for missing source attribution
        if not re.search(r'(according to|research by|study from|reported by)', all_text):
            missing.append("No source attribution for claims")

        # Check for missing timeline
        if not re.search(r'(before|after|during|since|until|timeline|history)', all_text):
            missing.append("No temporal context provided")

        return missing

    def _generate_recommendations(
        self,
        coherence: float,
        fallacies: list[str],
        missing: list[str],
        arcs: list[StoryArc],
    ) -> list[str]:
        """Generate recommendations to improve narrative coherence."""
        recs = []

        if coherence < 0.4:
            recs.append("Evidence lacks narrative structure — consider organizing by story arc")

        if fallacies:
            recs.append(f"Detected {len(fallacies)} potential narrative fallacy(ies) — verify causal claims")

        if len(missing) > 2:
            recs.append(f"Missing {len(missing)} narrative elements — seek additional context")

        for arc in arcs:
            if arc.missing_elements:
                recs.append(f"Story arc missing: {', '.join(arc.missing_elements)}")

        return recs

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "assessments": len(self._assessment_history),
            "avg_coherence": (
                sum(a.overall_coherence for a in self._assessment_history)
                / len(self._assessment_history)
                if self._assessment_history else 0.0
            ),
            "total_fallacies_detected": sum(
                len(a.narrative_fallacies) for a in self._assessment_history
            ),
        }
