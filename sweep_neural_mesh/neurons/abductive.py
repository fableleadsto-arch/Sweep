"""
Abductive Reasoning Center — inference to the best explanation.

Unlike deduction (rules → conclusion) or induction (cases → rule),
abduction starts with OBSERVATIONS and seeks the BEST EXPLANATION.

Examples:
- "The ground is wet" → "It probably rained" (best explanation)
- "The server is slow" → "Database connection pool is exhausted" (best explanation)
- "Multiple sources agree" → "The claim is likely true" (best explanation)

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │            ABDUCTIVE REASONING CENTER                │
    │                                                     │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Observations                                │  │
    │  │  (evidence items from the reasoning pipeline) │  │
    │  └──────────────────┬───────────────────────────┘  │
    │                     ↓                               │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Hypothesis Generation                        │  │
    │  │  (generate candidate explanations)             │  │
    │  └──────────────────┬───────────────────────────┘  │
    │                     ↓                               │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Hypothesis Scoring                           │  │
    │  │  - Simplicity (Occam's razor)                 │  │
    │  │  - Scope (how much it explains)               │  │
    │  │  - Coherence (internal consistency)           │  │
    │  │  - Novelty (doesn't require ad hoc fixes)     │  │
    │  └──────────────────┬───────────────────────────┘  │
    │                     ↓                               │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Best Explanation Selection                   │  │
    │  │  (ranked hypotheses with confidence)           │  │
    │  └──────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hypothesis:
    """A candidate explanation for observed evidence."""
    hypothesis_id: str
    explanation: str
    # Scoring dimensions
    simplicity: float = 0.5        # Occam's razor: simpler = better
    scope: float = 0.5             # how much evidence it explains
    coherence: float = 0.5         # internal consistency
    novelty: float = 0.5           # doesn't require ad hoc assumptions
    # Overall
    overall_score: float = 0.0
    confidence: float = 0.0
    # Supporting evidence
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    # Metadata
    generation_method: str = ""    # how this hypothesis was generated
    timestamp: float = field(default_factory=time.time)


@dataclass
class AbductiveResult:
    """Result of abductive reasoning."""
    observations: list[str]
    hypotheses: list[Hypothesis]    # ranked by score
    best_explanation: Hypothesis | None
    reasoning_chain: list[str]      # step-by-step reasoning
    confidence_in_best: float
    alternative_count: int          # how many alternatives were considered


class AbductiveReasoner:
    """
    Generate and evaluate candidate explanations for observations.

    Like a doctor diagnosing symptoms or a detective solving a case,
    this module:

    1. GENERATES multiple candidate explanations (hypotheses)
    2. SCORES each hypothesis on simplicity, scope, coherence, novelty
    3. RANKS hypotheses by overall score
    4. SELECTS the best explanation
    5. LEARNS from outcomes to improve future hypothesis generation

    Scoring criteria (from philosophy of science):
    - SIMPLICITY: Fewer assumptions = better (Occam's razor)
    - SCOPE: Explains more observations = better
    - COHERENCE: Internally consistent = better
    - NOVELTY: Doesn't require ad hoc patches = better

    This is different from the CausalLinker (which finds relationships
    BETWEEN evidence items). Abductive reasoning finds the BEST
    EXPLANATION FOR all the evidence.
    """

    def __init__(self) -> None:
        self._hypothesis_history: list[AbductiveResult] = []
        self._learned_patterns: dict[str, list[str]] = {}

    def reason(
        self,
        observations: list[str],
        context: dict[str, Any] | None = None,
        max_hypotheses: int = 5,
    ) -> AbductiveResult:
        """
        Generate and evaluate hypotheses for the given observations.

        This is the main entry point for abductive reasoning.
        """
        reasoning_chain: list[str] = []
        reasoning_chain.append(f"Analyzing {len(observations)} observations")

        # Step 1: Generate hypotheses
        hypotheses = self._generate_hypotheses(observations, context)
        reasoning_chain.append(f"Generated {len(hypotheses)} candidate hypotheses")

        # Step 2: Score each hypothesis
        for hyp in hypotheses:
            self._score_hypothesis(hyp, observations)
        reasoning_chain.append("Scored all hypotheses on simplicity, scope, coherence, novelty")

        # Step 3: Rank by overall score
        hypotheses.sort(key=lambda h: h.overall_score, reverse=True)
        hypotheses = hypotheses[:max_hypotheses]

        # Step 4: Select best explanation
        best = hypotheses[0] if hypotheses else None
        if best:
            reasoning_chain.append(
                f"Best explanation: '{best.explanation[:80]}...' "
                f"(score: {best.overall_score:.2f})"
            )

        result = AbductiveResult(
            observations=observations,
            hypotheses=hypotheses,
            best_explanation=best,
            reasoning_chain=reasoning_chain,
            confidence_in_best=best.overall_score if best else 0.0,
            alternative_count=len(hypotheses),
        )

        self._hypothesis_history.append(result)
        return result

    def _generate_hypotheses(
        self,
        observations: list[str],
        context: dict[str, Any] | None,
    ) -> list[Hypothesis]:
        """Generate candidate explanations for observations."""
        hypotheses: list[Hypothesis] = []
        hyp_id = 0

        # Strategy 1: Common pattern matching
        for obs in observations:
            obs_lower = obs.lower()
            # Generate hypotheses based on common patterns
            if re.search(r'(slow|delay|timeout|lag)', obs_lower):
                hypotheses.append(Hypothesis(
                    hypothesis_id=f"h_{hyp_id}",
                    explanation="System resource exhaustion or bottleneck",
                    generation_method="pattern_matching",
                ))
                hyp_id += 1

            if re.search(r'(error|fail|crash|exception)', obs_lower):
                hypotheses.append(Hypothesis(
                    hypothesis_id=f"h_{hyp_id}",
                    explanation="Code defect or configuration issue",
                    generation_method="pattern_matching",
                ))
                hyp_id += 1

            if re.search(r'(increase|decrease|change|shift)', obs_lower):
                hypotheses.append(Hypothesis(
                    hypothesis_id=f"h_{hyp_id}",
                    explanation="Environmental or input distribution change",
                    generation_method="pattern_matching",
                ))
                hyp_id += 1

            # Generic hypothesis: "there's a causal relationship"
            if len(observations) > 1:
                hypotheses.append(Hypothesis(
                    hypothesis_id=f"h_{hyp_id}",
                    explanation=f"Observations are related through a common cause",
                    generation_method="common_cause",
                ))
                hyp_id += 1

        # Strategy 2: Combination hypothesis
        if len(observations) > 2:
            hypotheses.append(Hypothesis(
                hypothesis_id=f"h_{hyp_id}",
                explanation="Multiple factors are interacting to produce these observations",
                generation_method="combination",
            ))
            hyp_id += 1

        # Strategy 3: Null hypothesis
        hypotheses.append(Hypothesis(
            hypothesis_id=f"h_{hyp_id}",
            explanation="Observations are coincidental and not causally related",
            generation_method="null",
        ))
        hyp_id += 1

        return hypotheses

    def _score_hypothesis(
        self,
        hypothesis: Hypothesis,
        observations: list[str],
    ) -> None:
        """Score a hypothesis on all four dimensions."""
        obs_text = " ".join(observations).lower()
        hyp_words = set(re.findall(r'\b\w{4,}\b', hypothesis.explanation.lower()))
        obs_words = set(re.findall(r'\b\w{4,}\b', obs_text))

        # SIMPLICITY: shorter explanations are simpler
        word_count = len(hypothesis.explanation.split())
        if word_count < 5:
            hypothesis.simplicity = 0.9
        elif word_count < 10:
            hypothesis.simplicity = 0.7
        elif word_count < 20:
            hypothesis.simplicity = 0.5
        else:
            hypothesis.simplicity = 0.3

        # Ad hoc assumptions reduce simplicity
        if re.search(r'(except|unless|but only|special case)', hypothesis.explanation.lower()):
            hypothesis.simplicity *= 0.7

        # SCOPE: how many observations does it explain?
        overlap = len(hyp_words & obs_words)
        hypothesis.scope = min(1.0, overlap / max(1, len(obs_words) * 0.3))

        # COHERENCE: is it internally consistent?
        # Simple heuristic: no contradictions in the explanation
        negations = len(re.findall(r'\b(not|never|no|doesn.t)\b', hypothesis.explanation.lower()))
        hypothesis.coherence = max(0.3, 1.0 - negations * 0.15)

        # Does it align with common sense?
        if re.search(r'(coincidence|random|chance)', hypothesis.explanation.lower()):
            hypothesis.coherence *= 0.8  # coincidence explanations are less coherent

        # NOVELTY: doesn't require special pleading
        hypothesis.novelty = 0.7  # base
        if re.search(r'(ad hoc|special|except|unless)', hypothesis.explanation.lower()):
            hypothesis.novelty *= 0.6

        # Check for learned patterns
        for pattern, explanations in self._learned_patterns.items():
            if pattern in obs_text:
                if hypothesis.explanation in explanations:
                    hypothesis.novelty = min(1.0, hypothesis.novelty + 0.2)

        # OVERALL: weighted combination
        hypothesis.overall_score = (
            hypothesis.simplicity * 0.25
            + hypothesis.scope * 0.35
            + hypothesis.coherence * 0.25
            + hypothesis.novelty * 0.15
        )

        hypothesis.confidence = hypothesis.overall_score

    def learn_from_outcome(
        self,
        hypothesis: str,
        observations: list[str],
        was_correct: bool,
    ) -> None:
        """
        Learn from a reasoning outcome to improve future hypothesis generation.
        """
        obs_key = " ".join(sorted(observations)[:3])[:100]
        if obs_key not in self._learned_patterns:
            self._learned_patterns[obs_key] = []

        if was_correct and hypothesis not in self._learned_patterns[obs_key]:
            self._learned_patterns[obs_key].append(hypothesis)

    @property
    def hypothesis_count(self) -> int:
        return sum(len(r.hypotheses) for r in self._hypothesis_history)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "reasoning_sessions": len(self._hypothesis_history),
            "total_hypotheses": self.hypothesis_count,
            "learned_patterns": len(self._learned_patterns),
            "avg_hypotheses_per_session": (
                self.hypothesis_count / len(self._hypothesis_history)
                if self._hypothesis_history else 0.0
            ),
        }
