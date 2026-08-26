"""
Cortex-Basal Ganglia-Thalamus Loop — action selection and execution.

Modeled after the biological cortico-basal ganglia-thalamo-cortical loop:

    ┌─────────────────────────────────────────────────────────┐
    │                     CORTEX                              │
    │  Proposes actions: "we should trust this evidence",     │
    │  "we should escalate processing", "this is sufficient"  │
    │                         ↓ (proposals)                   │
    │                   BASAL GANGLIA                         │
    │  Decides Go/NoGo via reinforcement learning:            │
    │  "has this type of proposal worked before?"             │
    │                         ↓ (Go signal)                   │
    │                      THALAMUS                           │
    │  Relays the selected action back to cortex:             │
    │  "proceed with this plan"                               │
    │                         ↓                               │
    │                    CORTEX (again)                       │
    │  Executes the selected action                           │
    └─────────────────────────────────────────────────────────┘

Key hypothesis (Arakawa 2024, arxiv:2402.13275):
  The cerebral cortex predicts/actions, while the basal ganglia
  use reinforcement learning to decide whether to perform them.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .signal import Signal, SignalType

logger = logging.getLogger("sweep.neurons.basal_ganglia")


class ActionType(Enum):
    """Types of actions the cortex can propose."""
    ESCALATE_CREDIBILITY = "escalate_credibility"
    ESCALATE_TEMPORAL = "escalate_temporal"
    ESCALATE_CAUSAL = "escalate_causal"
    ESCALATE_CONTRADICTION = "escalate_contradiction"
    TRUST_EVIDENCE = "trust_evidence"
    REJECT_EVIDENCE = "reject_evidence"
    INCREASE_CONFIDENCE = "increase_confidence"
    DECREASE_CONFIDENCE = "decrease_confidence"
    PROCEED_TO_CONSENSUS = "proceed_to_consensus"
    REQUEST_MORE_EVIDENCE = "request_more_evidence"


@dataclass
class ActionProposal:
    """A proposed action from the cortex."""
    action_type: ActionType
    confidence: float           # cortex's confidence in this proposal
    reasoning: str              # why this action is proposed
    evidence_ids: list[str]     # which evidence items are involved
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionDecision:
    """The basal ganglia's decision on a proposed action."""
    proposal: ActionProposal
    go: bool                    # True = execute, False = suppress
    confidence: float           # BG's confidence in its decision
    reasoning: str              # why this Go/NoGo decision
    learning_delta: float = 0.0 # weight change from RL
    # ── Reward Prediction Error ──
    expected_value: float = 0.0   # what BG expected before seeing reward
    actual_reward: float = 0.0    # what actually happened
    prediction_error: float = 0.0 # actual - expected (RPE)


@dataclass
class ThalamusRelay:
    """The thalamus relay output: selected actions to execute."""
    selected_actions: list[ActionDecision]
    rejected_actions: list[ActionDecision]
    relay_time_ms: float
    total_go: int
    total_nogo: int


class BasalGanglia:
    """
    Action selection via reinforcement learning.

    Like the biological basal ganglia:
    1. Receives state information from cortex (current evidence state)
    2. Receives action proposals from cortex (what to do next)
    3. Uses learned Go/NoGo policy to decide each proposal
    4. Outputs a Go signal for accepted proposals
    5. Learns from outcomes (reward/punishment)

    The BG learns which types of proposals work well in which
    contexts, building a policy over time through reinforcement.
    """

    def __init__(self) -> None:
        # State-Action → Go/NoGo policy (learned through RL)
        # Key: (action_type, context_bucket) → value
        self._policy: dict[tuple[str, str], float] = {}

        # Learning parameters
        self._learning_rate = 0.15
        self._discount_factor = 0.9

        # Statistics
        self._decisions: list[ActionDecision] = []
        self._rewards: list[float] = []

        # Exploration rate (epsilon-greedy)
        self._exploration_rate = 0.1

    def decide(
        self,
        proposals: list[ActionProposal],
        context: dict[str, Any],
    ) -> list[ActionDecision]:
        """
        Evaluate each proposed action and decide Go/NoGo.

        Args:
            proposals: Actions proposed by the cortex.
            context: Current state (confidence, evidence count, etc.)

        Returns:
            List of decisions (Go or NoGo for each proposal).
        """
        decisions: list[ActionDecision] = []

        for proposal in proposals:
            # Get context bucket for policy lookup
            ctx_bucket = self._bucket_context(context)

            # Look up learned policy value
            policy_key = (proposal.action_type.value, ctx_bucket)
            policy_value = self._policy.get(policy_key, 0.0)

            # Combine cortex confidence with learned policy
            combined_score = (
                proposal.confidence * 0.5 +    # cortex's assessment
                policy_value * 0.3 +           # learned from past
                0.2                            # prior (slightly positive)
            )

            # Epsilon-greedy exploration
            import random
            if random.random() < self._exploration_rate:
                go = combined_score > 0.4  # exploratory threshold
            else:
                go = combined_score > 0.45  # exploitation threshold

            reasoning = (
                f"policy={policy_value:.2f}, cortex_conf={proposal.confidence:.2f}, "
                f"combined={combined_score:.2f} → {'Go' if go else 'NoGo'}"
            )

            decision = ActionDecision(
                proposal=proposal,
                go=go,
                confidence=abs(combined_score - 0.5) * 2,  # distance from boundary
                reasoning=reasoning,
            )
            decisions.append(decision)
            self._decisions.append(decision)

        go_count = sum(1 for d in decisions if d.go)
        logger.debug(f"BasalGanglia: {len(decisions)} proposals → {go_count} Go, {len(decisions)-go_count} NoGo")
        return decisions

    def learn(
        self,
        decisions: list[ActionDecision],
        reward: float,
    ) -> list[float]:
        """
        Update the policy based on the outcome.

        Like dopamine-mediated reinforcement learning in the
        biological basal ganglia, this strengthens or weakens
        action-context associations based on reward.

        Returns the list of prediction errors (RPEs) per decision.
        """
        self._rewards.append(reward)
        rpe_list: list[float] = []

        for decision in decisions:
            if not decision.go:
                decision.expected_value = 0.0
                decision.actual_reward = reward
                decision.prediction_error = 0.0
                rpe_list.append(0.0)
                continue

            ctx_bucket = self._bucket_context({
                "confidence": decision.proposal.metadata.get("confidence", 0.5),
                "evidence_count": decision.proposal.metadata.get("evidence_count", 0),
            })
            policy_key = (decision.proposal.action_type.value, ctx_bucket)

            # TD error: reward - expected value
            old_value = self._policy.get(policy_key, 0.0)
            td_error = reward - old_value

            # Update policy
            new_value = old_value + self._learning_rate * td_error
            self._policy[policy_key] = max(-1.0, min(1.0, new_value))

            # Record learning delta for trace
            decision.learning_delta = new_value - old_value

            # Record RPE on the decision
            decision.expected_value = old_value
            decision.actual_reward = reward
            decision.prediction_error = td_error
            rpe_list.append(td_error)

        logger.debug(f"BasalGanglia learn: reward={reward:.3f}, {len(rpe_list)} RPEs computed")
        return rpe_list

    def get_rpe_stats(self) -> dict[str, Any]:
        """Get reward prediction error statistics across all decisions."""
        rpes = [d.prediction_error for d in self._decisions if d.go]
        if not rpes:
            return {"total_rpe_events": 0, "avg_rpe": 0.0, "max_rpe": 0.0, "min_rpe": 0.0}
        return {
            "total_rpe_events": len(rpes),
            "avg_rpe": round(sum(rpes) / len(rpes), 6),
            "max_rpe": round(max(rpes), 6),
            "min_rpe": round(min(rpes), 6),
            "last_rpe": round(rpes[-1], 6),
            "positive_rpe_count": sum(1 for r in rpes if r > 0),
            "negative_rpe_count": sum(1 for r in rpes if r < 0),
        }

    def _bucket_context(self, context: dict[str, Any]) -> str:
        """
        Discretize continuous context into buckets for policy lookup.

        Like the striatal medium spiny neurons that classify
        input vectors into discrete action patterns.
        """
        confidence = context.get("confidence", 0.5)
        evidence_count = context.get("evidence_count", 0)

        # Confidence bucket
        if confidence > 0.7:
            conf_bucket = "high"
        elif confidence > 0.4:
            conf_bucket = "mid"
        else:
            conf_bucket = "low"

        # Evidence count bucket
        if evidence_count > 10:
            ev_bucket = "many"
        elif evidence_count > 3:
            ev_bucket = "some"
        else:
            ev_bucket = "few"

        return f"{conf_bucket}_{ev_bucket}"

    @property
    def stats(self) -> dict[str, Any]:
        """Return learning statistics."""
        go_count = sum(1 for d in self._decisions if d.go)
        nogo_count = sum(1 for d in self._decisions if not d.go)
        avg_reward = (
            sum(self._rewards) / len(self._rewards)
            if self._rewards else 0.0
        )
        return {
            "total_decisions": len(self._decisions),
            "go_count": go_count,
            "nogo_count": nogo_count,
            "avg_reward": round(avg_reward, 4),
            "policy_size": len(self._policy),
            "exploration_rate": self._exploration_rate,
        }


class Thalamus:
    """
    Relay station between basal ganglia and cortex.

    Like the biological thalamus:
    1. Receives Go signals from basal ganglia
    2. Relays them back to cortex for execution
    3. Gates which actions actually get executed
    4. Controls signal timing and sequencing

    The thalamus prevents noise from reaching cortex:
    only well-validated Go signals pass through.
    """

    def __init__(self) -> None:
        self._relay_history: list[ThalamusRelay] = []

    def relay(
        self,
        decisions: list[ActionDecision],
        min_confidence: float = 0.3,
    ) -> ThalamusRelay:
        """
        Relay Go decisions from basal ganglia to cortex.

        Filters out low-confidence Go signals to prevent
        noise from triggering actions.
        """
        t0 = time.perf_counter()

        selected: list[ActionDecision] = []
        rejected: list[ActionDecision] = []

        for decision in decisions:
            if decision.go and decision.confidence >= min_confidence:
                selected.append(decision)
            elif decision.go:
                # Go but low confidence → suppress
                decision.go = False
                decision.reasoning += " [thalamus: suppressed low-confidence Go]"
                rejected.append(decision)
            else:
                rejected.append(decision)

        elapsed = (time.perf_counter() - t0) * 1000
        relay = ThalamusRelay(
            selected_actions=selected,
            rejected_actions=rejected,
            relay_time_ms=elapsed,
            total_go=len(selected),
            total_nogo=len(rejected),
        )
        self._relay_history.append(relay)
        logger.debug(f"Thalamus relay: {relay.total_go} selected, {relay.total_nogo} rejected")
        return relay

    @property
    def relay_count(self) -> int:
        return len(self._relay_history)
