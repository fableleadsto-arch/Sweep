"""
Hard Negative Training — §16

Create difficult examples where:
- Two answers look almost identical
- One source is subtly wrong
- Dates conflict
- Names are similar
- Locations have similar names
- Documents contain misleading information
- Evidence is incomplete
- Multiple hypotheses are plausible

Train Sweep to recognize uncertainty rather than forcing a conclusion.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HardNegative:
    """A single hard negative training example."""
    hard_negative_id: str
    category: str
    original_task: str
    adversarial_task: str
    original_answer: str
    adversarial_answer: str
    difficulty_delta: float  # how much harder the negative is
    subtle_difference: str  # description of what changed
    correct_response: str  # what the model should say
    metadata: dict[str, Any] = field(default_factory=dict)


class HardNegativeGenerator:
    """
    §16: Creates adversarial examples that test whether Sweep can
    distinguish between similar but different inputs.

    These are NOT simple paraphrases. They are designed to be
    confusingly similar while requiring different answers.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._counter = 0
        self._negatives: list[HardNegative] = []

    def generate_all(self, count: int = 50) -> list[HardNegative]:
        """Generate a balanced set of hard negatives across all categories."""
        generators = [
            self._gen_similar_names,
            self._gen_conflicting_dates,
            self._gen_subtly_wrong_source,
            self._gen_similar_locations,
            self._gen_incomplete_evidence,
            self._gen_multiple_hypotheses,
            self._gen_misleading_context,
            self._gen_near_identical_answers,
        ]
        per_category = max(1, count // len(generators))
        negatives = []
        for gen in generators:
            negatives.extend(gen(per_category))
        self._rng.shuffle(negatives)
        self._negatives.extend(negatives)
        return negatives

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY: Similar Names
    # ══════════════════════════════════════════════════════════════════

    def _gen_similar_names(self, count: int) -> list[HardNegative]:
        pairs = [
            ("John Smith", "Jon Smith", "different first name spelling"),
            ("Michael Johnson", "Michelle Johnson", "gender/first name variant"),
            ("Dr. Zhang Wei", "Dr. Zhang Wei", "same name, different person"),
            ("Apple Inc.", "Apple Corps", "similar name, different entity"),
            ("George Washington", "George Washington Carver", "partial name match"),
            ("New York", "Newark", "similar location names"),
            ("Columbia University", "Columbia, SC", "name ambiguity"),
            ("J. Robert Oppenheimer", "Robert Oppenheimer", "formal vs informal"),
        ]
        results = []
        for i in range(min(count, len(pairs))):
            a, b, diff = pairs[i]
            self._counter += 1
            results.append(HardNegative(
                hard_negative_id=f"HN-SN-{self._counter:04d}",
                category="similar_names",
                original_task=f"Are '{a}' and '{b}' the same entity?",
                adversarial_task=f"Compare the entities '{a}' and '{b}'.",
                original_answer="depends_on_context",
                adversarial_answer="requires_further_investigation",
                difficulty_delta=0.3,
                subtle_difference=diff,
                correct_response="UNKNOWN",
                metadata={"entity_a": a, "entity_b": b},
            ))
        return results

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY: Conflicting Dates
    # ══════════════════════════════════════════════════════════════════

    def _gen_conflicting_dates(self, count: int) -> list[HardNegative]:
        conflicts = [
            ("WWII ended in 1945", "WWII ended in 1944", "year difference"),
            ("Einstein published relativity in 1905", "Einstein published relativity in 1915", "10-year gap"),
            ("The telephone was invented in 1876", "The telephone was invented in 1874", "2-year gap"),
            ("Python was created in 1991", "Python was created in 1989", "creation vs release"),
            ("The Internet started in 1969", "The Internet started in 1983", "ARPANET vs TCP/IP"),
        ]
        results = []
        for i in range(min(count, len(conflicts))):
            a, b, diff = conflicts[i]
            self._counter += 1
            results.append(HardNegative(
                hard_negative_id=f"HN-CD-{self._counter:04d}",
                category="conflicting_dates",
                original_task=f"Source A says: {a}. Source B says: {b}. Which is correct?",
                adversarial_task=f"Given these conflicting claims:\n1. {a}\n2. {b}\nDetermine the correct date.",
                original_answer="requires_verification",
                adversarial_answer="requires_source_evaluation",
                difficulty_delta=0.4,
                subtle_difference=diff,
                correct_response="SOURCES_DISAGREE",
                metadata={"claim_a": a, "claim_b": b},
            ))
        return results

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY: Subtly Wrong Source
    # ══════════════════════════════════════════════════════════════════

    def _gen_subtly_wrong_source(self, count: int) -> list[HardNegative]:
        examples = [
            (
                "Water boils at 100°C at sea level.",
                "Water boils at 90°C at sea level.",
                "10-degree error in well-known fact",
            ),
            (
                "The speed of light is approximately 3×10⁸ m/s.",
                "The speed of light is approximately 3×10⁶ m/s.",
                "order of magnitude error",
            ),
            (
                "DNA has a double helix structure.",
                "DNA has a triple helix structure.",
                "structural error",
            ),
            (
                "The Earth orbits the Sun.",
                "The Sun orbits the Earth.",
                "fundamental reversal",
            ),
        ]
        results = []
        for i in range(min(count, len(examples))):
            correct, wrong, diff = examples[i]
            self._counter += 1
            results.append(HardNegative(
                hard_negative_id=f"HN-SS-{self._counter:04d}",
                category="subtly_wrong_source",
                original_task=f"Evaluate this claim: {wrong}",
                adversarial_task=f"Is this statement accurate? '{wrong}'",
                original_answer="INCORRECT",
                adversarial_answer="INCORRECT",
                difficulty_delta=0.2,
                subtle_difference=diff,
                correct_response="INCORRECT",
                metadata={"correct": correct, "wrong": wrong},
            ))
        return results

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY: Similar Locations
    # ══════════════════════════════════════════════════════════════════

    def _gen_similar_locations(self, count: int) -> list[HardNegative]:
        pairs = [
            ("Springfield, IL", "Springfield, MA", "different states"),
            ("Cambridge, UK", "Cambridge, MA", "different countries"),
            ("San Jose, CA", "San José, Costa Rica", "different countries"),
            ("Guinea", "Guinea-Bissau", "similar country names"),
            ("Georgia (US state)", "Georgia (country)", "same name, different entities"),
        ]
        results = []
        for i in range(min(count, len(pairs))):
            a, b, diff = pairs[i]
            self._counter += 1
            results.append(HardNegative(
                hard_negative_id=f"HN-SL-{self._counter:04d}",
                category="similar_locations",
                original_task=f"Are '{a}' and '{b}' the same place?",
                adversarial_task=f"Compare locations: {a} vs {b}.",
                original_answer="different_places",
                adversarial_answer="different_places",
                difficulty_delta=0.3,
                subtle_difference=diff,
                correct_response="DIFFERENT",
                metadata={"loc_a": a, "loc_b": b},
            ))
        return results

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY: Incomplete Evidence
    # ══════════════════════════════════════════════════════════════════

    def _gen_incomplete_evidence(self, count: int) -> list[HardNegative]:
        examples = [
            (
                "Who was the first person on the Moon?",
                ["Neil Armstrong stepped on the Moon."],
                "One source only, no corroboration",
            ),
            (
                "What caused the fire?",
                ["The fire started in the kitchen."],
                "No evidence of cause, only location",
            ),
            (
                "Is this investment safe?",
                ["The company reported profits last quarter."],
                "Single data point, insufficient for conclusion",
            ),
        ]
        results = []
        for i in range(min(count, len(examples))):
            q, ev, diff = examples[i]
            self._counter += 1
            results.append(HardNegative(
                hard_negative_id=f"HN-IE-{self._counter:04d}",
                category="incomplete_evidence",
                original_task=f"Question: {q}\nEvidence: {ev}",
                adversarial_task=f"Based on the evidence, answer: {q}",
                original_answer="insufficient_evidence",
                adversarial_answer="insufficient_evidence",
                difficulty_delta=0.3,
                subtle_difference=diff,
                correct_response="INSUFFICIENT",
                metadata={"question": q, "evidence": ev},
            ))
        return results

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY: Multiple Hypotheses
    # ══════════════════════════════════════════════════════════════════

    def _gen_multiple_hypotheses(self, count: int) -> list[HardNegative]:
        examples = [
            (
                "Why did the stock drop?",
                ["Poor earnings report", "Market-wide selloff", "CEO resignation"],
                "Three plausible causes",
            ),
            (
                "What species is this?",
                ["Could be a fox or a coyote", "Similar ear shape to both"],
                "Ambiguous identification",
            ),
            (
                "Who wrote this email?",
                ["The writing style matches Alice", "But it was sent from Bob's account"],
                "Conflicting evidence about authorship",
            ),
        ]
        results = []
        for i in range(min(count, len(examples))):
            q, hyps, diff = examples[i]
            self._counter += 1
            results.append(HardNegative(
                hard_negative_id=f"HN-MH-{self._counter:04d}",
                category="multiple_hypotheses",
                original_task=f"{q}\nHypotheses: {', '.join(hyps)}",
                adversarial_task=f"Evaluate these hypotheses for '{q}': {', '.join(hyps)}",
                original_answer="multiple_plausible",
                adversarial_answer="multiple_plausible",
                difficulty_delta=0.4,
                subtle_difference=diff,
                correct_response="MULTIPLE_PLAUSIBLE",
                metadata={"hypotheses": hyps},
            ))
        return results

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY: Misleading Context
    # ══════════════════════════════════════════════════════════════════

    def _gen_misleading_context(self, count: int) -> list[HardNegative]:
        examples = [
            (
                "The study was funded by the sugar industry and found sugar is healthy.",
                "The study's funding source may bias the results.",
                "Funding bias",
            ),
            (
                "A website with many ads claims this product cures cancer.",
                "Commercial motivation undermines credibility.",
                "Commercial bias",
            ),
            (
                "An anonymous source claims the CEO committed fraud.",
                "Anonymous claims require corroboration.",
                "Anonymous source",
            ),
        ]
        results = []
        for i in range(min(count, len(examples))):
            ctx, diff_label, diff_desc = examples[i]
            self._counter += 1
            results.append(HardNegative(
                hard_negative_id=f"HN-MC-{self._counter:04d}",
                category="misleading_context",
                original_task=f"Source: {ctx}\nIs this reliable?",
                adversarial_task=f"Evaluate the reliability of: {ctx}",
                original_answer="low_reliability",
                adversarial_answer="low_reliability",
                difficulty_delta=0.3,
                subtle_difference=diff_desc,
                correct_response="LOW_RELIABILITY",
                metadata={"context": ctx},
            ))
        return results

    # ══════════════════════════════════════════════════════════════════
    # CATEGORY: Near-Identical Answers
    # ══════════════════════════════════════════════════════════════════

    def _gen_near_identical_answers(self, count: int) -> list[HardNegative]:
        pairs = [
            ("The meeting is at 3:00 PM", "The meeting is at 3:30 PM", "30-minute difference"),
            ("Revenue grew 5.2%", "Revenue grew 5.3%", "0.1% difference"),
            ("Population is 1.2 million", "Population is 1.3 million", "small absolute difference"),
            ("The deadline is Friday", "The deadline is Saturday", "one day difference"),
            ("Temperature is 72°F", "Temperature is 73°F", "one degree difference"),
        ]
        results = []
        for i in range(min(count, len(pairs))):
            a, b, diff = pairs[i]
            self._counter += 1
            results.append(HardNegative(
                hard_negative_id=f"HN-NA-{self._counter:04d}",
                category="near_identical_answers",
                original_task=f"Source A: {a}. Source B: {b}. Which is correct?",
                adversarial_task=f"Compare: '{a}' vs '{b}'. Are they the same?",
                original_answer="close_but_different",
                adversarial_answer="close_but_different",
                difficulty_delta=0.5,
                subtle_difference=diff,
                correct_response="NEAR_MATCH_BUT_DIFFERENT",
                metadata={"answer_a": a, "answer_b": b},
            ))
        return results

    @property
    def generated_count(self) -> int:
        return len(self._negatives)

    def summary(self) -> dict[str, Any]:
        by_category: dict[str, int] = {}
        for hn in self._negatives:
            by_category[hn.category] = by_category.get(hn.category, 0) + 1
        return {
            "total_generated": len(self._negatives),
            "categories": by_category,
            "avg_difficulty_delta": (
                sum(h.difficulty_delta for h in self._negatives) / max(1, len(self._negatives))
            ),
        }
