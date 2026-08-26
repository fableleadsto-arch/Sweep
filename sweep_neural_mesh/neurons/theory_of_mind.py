"""
Theory of Mind Module — understanding other agents' beliefs, goals, and intentions.

Humans constantly reason about what OTHER PEOPLE are thinking:
- "The person asking this probably already knows X but is confused about Y"
- "This source has a political bias, so their evidence is skewed"
- "The author is trying to persuade, not inform"

Without Theory of Mind, Sweep treats all evidence as coming from
neutral, omniscient sources. With it, Sweep understands that evidence
comes from PEOPLE with limited knowledge, specific goals, and biases.

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │            THEORY OF MIND MODULE                     │
    │                                                     │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Per-Stakeholder State Tracking               │  │
    │  │  - goal_state: what they're trying to do      │  │
    │  │  - belief_state: what they think they know    │  │
    │  │  - affect_state: their emotional state        │  │
    │  │  - constraint_state: their limitations        │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Intent Inference                             │  │
    │  │  - Persuasion detection                       │  │
    │  │  - Information gap detection                  │  │
    │  │  - Bias detection                             │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Social Reasoning                             │  │
    │  │  - Trust assessment                           │  │
    │  │  - Motivation analysis                        │  │
    │  │  - Credibility adjustment                     │  │
    │  └──────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """The mental state of an observed agent (person/source)."""
    agent_id: str
    name: str
    # Baddeley-inspired state tracking
    goal_state: str = ""             # what they're trying to accomplish
    belief_state: str = ""           # what they think they know
    affect_state: str = "neutral"    # emotional state: neutral, urgent, excited, defensive
    constraint_state: str = ""       # hard boundaries or limitations
    # Derived assessments
    knowledge_level: float = 0.5     # 0.0-1.0: how much do they actually know
    intent_confidence: float = 0.5   # 0.0-1.0: how confident are we about their intent
    trust_score: float = 0.5         # 0.0-1.0: how much should we trust them
    bias_direction: str = ""         # detected bias: "toward", "against", "neutral"
    # Tracking
    interactions: int = 0
    last_seen: float = field(default_factory=time.time)


@dataclass
class IntentAssessment:
    """Assessment of an agent's intent."""
    agent_id: str
    primary_intent: str              # "inform", "persuade", "entertain", "deceive", "ask"
    intent_confidence: float
    evidence_for_intent: list[str]
    bias_detected: str               # "none", "political", "commercial", "ideological"
    should_trust: bool
    adjustment_recommendation: str   # how to adjust credibility


@dataclass
class SocialContext:
    """The social context of a reasoning situation."""
    participants: list[str]          # who is involved
    relationships: dict[str, str]    # participant → relationship description
    power_dynamics: str              # who has authority/expertise
    shared_knowledge: list[str]      # what all participants likely know
    contested_claims: list[str]      # what is being debated


class TheoryOfMind:
    """
    Understand other agents' beliefs, goals, and intentions.

    Like the human ability to reason about other minds, this module:

    1. TRACKS per-agent mental states (goals, beliefs, affect, constraints)
    2. INFERS intent from behavior and context
    3. DETECTS persuasion, deception, and bias
    4. ADJUSTS credibility assessments based on source motivation
    5. RECOGNIZES information gaps (what the source doesn't know)

    The key insight: evidence quality depends not just on WHAT is said,
    but on WHO says it and WHY. A pharmaceutical company saying their
    drug works is different from an independent researcher saying the same.

    State tracking per agent:
    - goal_state: what are they trying to accomplish?
    - belief_state: what do they think they know?
    - affect_state: are they neutral, excited, defensive?
    - constraint_state: what are their limitations?

    When intent inference confidence is LOW, the system defaults to
    explicit clarification rather than guessing.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentState] = {}
        self._intent_history: list[IntentAssessment] = []
        self._social_contexts: list[SocialContext] = []
        self._next_agent_id = 0

    def register_agent(
        self,
        name: str,
        goal: str = "",
        belief: str = "",
        affect: str = "neutral",
        constraint: str = "",
    ) -> AgentState:
        """Register a new agent (person/source) to track."""
        self._next_agent_id += 1
        agent = AgentState(
            agent_id=f"agent_{self._next_agent_id}",
            name=name,
            goal_state=goal,
            belief_state=belief,
            affect_state=affect,
            constraint_state=constraint,
        )
        self._agents[agent.agent_id] = agent
        return agent

    def infer_intent(
        self,
        agent_id: str,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> IntentAssessment:
        """
        Infer the intent of an agent based on their text and context.
        """
        agent = self._agents.get(agent_id)
        if not agent:
            return IntentAssessment(
                agent_id=agent_id,
                primary_intent="unknown",
                intent_confidence=0.0,
                evidence_for_intent=["agent not registered"],
                bias_detected="none",
                should_trust=False,
                adjustment_recommendation="register agent first",
            )

        text_lower = text.lower()
        agent.interactions += 1
        agent.last_seen = time.time()

        # Detect intent
        intent_scores = {
            "inform": 0.0,
            "persuade": 0.0,
            "entertain": 0.0,
            "deceive": 0.0,
            "ask": 0.0,
        }

        # Inform indicators
        if re.search(r'(research|study|data|evidence|according to|published)', text_lower):
            intent_scores["inform"] += 0.3
        if re.search(r'\b(because|therefore|since|thus)\b', text_lower):
            intent_scores["inform"] += 0.2

        # Persuade indicators
        if re.search(r'(should|must|need to|have to|important to)', text_lower):
            intent_scores["persuade"] += 0.3
        if re.search(r'(believe|trust|accept|agree)', text_lower):
            intent_scores["persuade"] += 0.2
        if re.search(r'(best|only|always|never|everyone)', text_lower):
            intent_scores["persuade"] += 0.15  # absolutist language

        # Deceive indicators
        if re.search(r'(click here|buy now|limited time|secret|miracle)', text_lower):
            intent_scores["deceive"] += 0.4
        if re.search(r'(you won.t believe|shocking|unbelievable)', text_lower):
            intent_scores["deceive"] += 0.3

        # Ask indicators
        if re.search(r'\?$', text.strip()):
            intent_scores["ask"] += 0.3
        if re.search(r'(how|what|why|when|where|can you|could you)', text_lower):
            intent_scores["ask"] += 0.2

        # Determine primary intent
        primary_intent = max(intent_scores, key=intent_scores.get)
        intent_confidence = intent_scores[primary_intent]

        # Detect bias
        bias = self._detect_bias(text_lower, primary_intent)

        # Trust assessment
        should_trust = (
            primary_intent == "inform"
            and intent_confidence > 0.3
            and bias == "none"
        )

        # Build evidence list
        evidence = []
        for intent, score in intent_scores.items():
            if score > 0.1:
                evidence.append(f"{intent}: {score:.2f}")

        # Adjustment recommendation
        if primary_intent == "persuade":
            adj = "Reduce trust: source is trying to persuade, not inform"
        elif primary_intent == "deceive":
            adj = "Significantly reduce trust: source shows deception patterns"
        elif bias != "none":
            adj = f"Reduce trust: detected {bias} bias"
        else:
            adj = "No adjustment needed: source appears to be informing"

        # Update agent state
        agent.knowledge_level = min(1.0, agent.knowledge_level + intent_confidence * 0.1)
        agent.intent_confidence = intent_confidence
        agent.trust_score = 0.8 if should_trust else 0.3
        agent.bias_direction = bias

        assessment = IntentAssessment(
            agent_id=agent_id,
            primary_intent=primary_intent,
            intent_confidence=intent_confidence,
            evidence_for_intent=evidence,
            bias_detected=bias,
            should_trust=should_trust,
            adjustment_recommendation=adj,
        )
        self._intent_history.append(assessment)
        return assessment

    def _detect_bias(self, text: str, intent: str) -> str:
        """Detect bias in text."""
        # Commercial bias
        if re.search(r'(buy|purchase|discount|sale|offer|deal|price)', text):
            return "commercial"

        # Political bias
        if re.search(r'(liberal|conservative|democrat|republican|left|right|progressive)', text):
            return "political"

        # Ideological bias
        if re.search(r'(always|never|everyone|nobody|the only|must|fundamental)', text):
            if intent == "persuade":
                return "ideological"

        return "none"

    def update_agent_state(
        self,
        agent_id: str,
        goal: str | None = None,
        belief: str | None = None,
        affect: str | None = None,
        constraint: str | None = None,
    ) -> bool:
        """Update an agent's mental state."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False
        if goal is not None:
            agent.goal_state = goal
        if belief is not None:
            agent.belief_state = belief
        if affect is not None:
            agent.affect_state = affect
        if constraint is not None:
            agent.constraint_state = constraint
        return True

    def get_agent(self, agent_id: str) -> AgentState | None:
        return self._agents.get(agent_id)

    def get_agent_by_name(self, name: str) -> AgentState | None:
        for agent in self._agents.values():
            if agent.name.lower() == name.lower():
                return agent
        return None

    def assess_credibility_adjustment(
        self,
        source_name: str,
        base_credibility: float,
    ) -> tuple[float, str]:
        """
        Adjust credibility based on Theory of Mind assessment.

        Returns (adjusted_credibility, reason).
        """
        agent = self.get_agent_by_name(source_name)
        if not agent:
            return base_credibility, "no agent data available"

        # Adjust based on trust score
        trust_factor = agent.trust_score
        adjusted = base_credibility * 0.6 + trust_factor * 0.4

        # Bias penalty
        if agent.bias_direction != "none":
            adjusted *= 0.8

        # Intent adjustment
        if agent.intent_confidence > 0.5:
            # We're confident about their intent
            if agent.bias_direction == "none":
                adjusted = min(1.0, adjusted + 0.1)

        reason = (
            f"trust={agent.trust_score:.2f}, bias={agent.bias_direction}, "
            f"intent_confidence={agent.intent_confidence:.2f}"
        )

        return max(0.0, min(1.0, adjusted)), reason

    def detect_information_gaps(
        self,
        text: str,
        agent_id: str,
    ) -> list[str]:
        """
        Detect what the agent DOESN'T know.

        Like recognizing that a person's question reveals their
        knowledge gaps.
        """
        gaps = []
        text_lower = text.lower()

        # Question marks indicate knowledge gaps
        if "?" in text:
            gaps.append("agent is asking — may lack knowledge on this topic")

        # Hedging language indicates uncertainty
        if re.search(r'(maybe|perhaps|might|could be|not sure|uncertain)', text_lower):
            gaps.append("agent shows uncertainty — knowledge may be incomplete")

        # Superlatives without evidence indicate overconfidence
        if re.search(r'(best|worst|always|never|everyone|nobody)', text_lower):
            if not re.search(r'(research|study|data|evidence)', text_lower):
                gaps.append("agent makes strong claims without evidence — may lack depth")

        # Missing specific details
        if len(text.split()) > 50 and not re.search(r'\d+', text):
            gaps.append("agent provides no specific data — may lack detailed knowledge")

        return gaps

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "agent_count": len(self._agents),
            "intent_assessments": len(self._intent_history),
            "avg_trust": (
                sum(a.trust_score for a in self._agents.values())
                / len(self._agents)
                if self._agents else 0.5
            ),
            "bias_distribution": {
                bias: sum(1 for a in self._agents.values() if a.bias_direction == bias)
                for bias in ["none", "commercial", "political", "ideological"]
            },
        }
