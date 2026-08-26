"""
Advanced Mathematical Reasoning — Bayesian inference, probability, and decision theory.

Implements the mathematical backbone for Sweep's reasoning:

    Bayesian Inference:
        P(H|E) = P(E|H) * P(H) / P(E)

    Decision Theory:
        EU(action) = Σ P(outcome_i | action) * utility(outcome_i)

    Hypothesis Testing:
        Bayes Factor = P(E|H1) / P(E|H0)

    Prior Updating:
        posterior = normalize(likelihood * prior)

All computations are logged for full traceability.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("sweep.math.bayesian")


# ════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════

@dataclass
class Hypothesis:
    """A hypothesis with prior probability and supporting evidence."""
    name: str
    prior: float = 0.5
    posterior: float = 0.5
    evidence_count: int = 0
    log_odds: float = 0.0  # log(posterior / (1 - posterior))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BayesianUpdate:
    """Result of a single Bayesian update step."""
    hypothesis_name: str
    prior: float
    likelihood: float
    marginal: float
    posterior: float
    log_odds_before: float
    log_odds_after: float
    information_bits: float  # how many bits of evidence this was


@dataclass
class DecisionOption:
    """An option in a decision-theoretic framework."""
    name: str
    outcomes: dict[str, float]    # outcome_name → probability
    utilities: dict[str, float]   # outcome_name → utility
    expected_utility: float = 0.0
    certainty_equivalent: float = 0.0
    risk_penalty: float = 0.0


@dataclass
class BayesTest:
    """Result of a Bayesian hypothesis test."""
    h0_name: str
    h1_name: str
    bayes_factor: float
    interpretation: str
    h0_posterior: float
    h1_posterior: float
    evidence_strength: str  # 'anecdotal', 'substantial', 'strong', 'decisive'


# ════════════════════════════════════════════════════════════════
# BAYESIAN REASONER
# ════════════════════════════════════════════════════════════════

class BayesianReasoner:
    """
    Bayesian inference engine for hypothesis updating.

    Maintains a collection of hypotheses and updates them
    as evidence arrives, computing posterior probabilities
    via Bayes' theorem with full logging of each update.

    Usage:
        br = BayesianReasoner()
        br.add_hypothesis("h1", prior=0.3)
        br.add_hypothesis("h2", prior=0.7)
        update = br.update("h1", likelihood=0.9, marginal_evidence=0.5)
    """

    def __init__(self) -> None:
        self._hypotheses: dict[str, Hypothesis] = {}
        self._updates: list[BayesianUpdate] = []
        self._total_updates = 0
        logger.info("BayesianReasoner initialized")

    def add_hypothesis(
        self, name: str, prior: float = 0.5, metadata: dict[str, Any] | None = None,
    ) -> Hypothesis:
        """Add a hypothesis with its prior probability."""
        prior = max(0.001, min(0.999, prior))  # clamp to avoid log(0)
        h = Hypothesis(
            name=name, prior=prior, posterior=prior,
            log_odds=math.log(prior / (1.0 - prior)),
            metadata=metadata or {},
        )
        self._hypotheses[name] = h
        logger.info(f"Added hypothesis '{name}' prior={prior:.4f}")
        return h

    def update(
        self,
        hypothesis_name: str,
        likelihood: float,
        marginal_evidence: float | None = None,
    ) -> BayesianUpdate:
        """
        Perform a Bayesian update on a single hypothesis.

        P(H|E) = P(E|H) * P(H) / P(E)

        Args:
            hypothesis_name: which hypothesis to update
            likelihood: P(E|H) — how likely is the evidence given this hypothesis
            marginal_evidence: P(E) — overall evidence probability. If None, computed
                               from all hypotheses using total probability theorem.

        Returns:
            BayesianUpdate with full details of the computation.
        """
        h = self._hypotheses.get(hypothesis_name)
        if h is None:
            raise KeyError(f"Hypothesis '{hypothesis_name}' not found")

        prior = h.posterior  # use current posterior as prior for sequential updates

        # Compute marginal evidence P(E) using total probability theorem
        if marginal_evidence is None:
            marginal_evidence = sum(
                hyp.posterior * likelihood
                for name, hyp in self._hypotheses.items()
                if name == hypothesis_name
            ) + sum(
                hyp.posterior * 0.5  # assume neutral likelihood for other hypotheses
                for name, hyp in self._hypotheses.items()
                if name != hypothesis_name
            )
            marginal_evidence = max(1e-10, marginal_evidence)  # avoid division by zero

        # Bayes' theorem
        posterior = (likelihood * prior) / marginal_evidence
        posterior = max(0.001, min(0.999, posterior))  # clamp

        # Log-odds form (more numerically stable)
        log_odds_before = h.log_odds
        log_likelihood_ratio = math.log(likelihood / max(1e-10, marginal_evidence))
        log_odds_after = log_odds_before + log_likelihood_ratio

        # Information content: bits of evidence this update provided
        information_bits = abs(log_likelihood_ratio) / math.log(2)

        # Update hypothesis
        h.posterior = posterior
        h.log_odds = log_odds_after
        h.evidence_count += 1

        update = BayesianUpdate(
            hypothesis_name=hypothesis_name,
            prior=prior,
            likelihood=likelihood,
            marginal=marginal_evidence,
            posterior=posterior,
            log_odds_before=log_odds_before,
            log_odds_after=log_odds_after,
            information_bits=information_bits,
        )
        self._updates.append(update)
        self._total_updates += 1

        logger.info(
            f"Bayesian update '{hypothesis_name}': "
            f"prior={prior:.4f} → posterior={posterior:.4f} "
            f"(+{information_bits:.2f} bits)"
        )
        return update

    def batch_update(
        self,
        hypothesis_name: str,
        likelihoods: list[float],
    ) -> list[BayesianUpdate]:
        """Apply multiple updates sequentially to a hypothesis."""
        updates = []
        for lh in likelihoods:
            u = self.update(hypothesis_name, lh)
            updates.append(u)
        return updates

    def compare_hypotheses(self) -> list[tuple[str, float]]:
        """Return hypotheses sorted by posterior (most likely first)."""
        items = [(h.name, h.posterior) for h in self._hypotheses.values()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    def log_odds_ratio(self, h1_name: str, h2_name: str) -> float:
        """Compute log-odds ratio between two hypotheses."""
        h1 = self._hypotheses[h1_name]
        h2 = self._hypotheses[h2_name]
        return h1.log_odds - h2.log_odds

    def brier_score(self, outcomes: dict[str, bool]) -> float:
        """
        Compute Brier score: measures calibration of probabilistic predictions.

        Brier = (1/N) * Σ (forecast - actual)²
        Lower is better (0.0 = perfect).
        """
        if not outcomes:
            return 0.0
        total = 0.0
        for name, actual in outcomes.items():
            h = self._hypotheses.get(name)
            if h is None:
                continue
            forecast = h.posterior
            total += (forecast - (1.0 if actual else 0.0)) ** 2
        return total / len(outcomes)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the Bayesian reasoner."""
        return {
            "hypothesis_count": len(self._hypotheses),
            "total_updates": self._total_updates,
            "avg_information_bits": (
                sum(u.information_bits for u in self._updates)
                / max(1, len(self._updates))
            ),
            "hypotheses": {
                name: {
                    "posterior": round(h.posterior, 4),
                    "evidence_count": h.evidence_count,
                    "log_odds": round(h.log_odds, 4),
                }
                for name, h in self._hypotheses.items()
            },
        }


# ════════════════════════════════════════════════════════════════
# DECISION THEORIST
# ════════════════════════════════════════════════════════════════

class DecisionTheorist:
    """
    Expected utility maximization with risk assessment.

    Given a set of options with outcomes and utilities,
    computes the optimal decision under uncertainty.

    EU(a) = Σ P(o_i | a) * U(o_i)

    Risk penalty: variance of the utility distribution
    Certainty equivalent: EU - risk_penalty (for risk-averse agents)
    """

    def __init__(self, risk_aversion: float = 0.5) -> None:
        self._risk_aversion = risk_aversion
        self._decisions: list[dict[str, Any]] = []
        logger.info(f"DecisionTheorist initialized (risk_aversion={risk_aversion})")

    def evaluate_option(
        self,
        name: str,
        outcomes: dict[str, float],
        utilities: dict[str, float],
    ) -> DecisionOption:
        """Evaluate a single decision option."""
        # Expected utility
        eu = sum(
            outcomes.get(o, 0.0) * utilities.get(o, 0.0)
            for o in set(outcomes) | set(utilities)
        )

        # Variance of utility
        variance = sum(
            outcomes.get(o, 0.0) * (utilities.get(o, 0.0) - eu) ** 2
            for o in set(outcomes) | set(utilities)
        )

        # Risk penalty (variance * risk aversion)
        risk_penalty = variance * self._risk_aversion

        # Certainty equivalent
        ce = eu - risk_penalty

        option = DecisionOption(
            name=name,
            outcomes=outcomes,
            utilities=utilities,
            expected_utility=eu,
            certainty_equivalent=ce,
            risk_penalty=risk_penalty,
        )
        logger.info(
            f"Option '{name}': EU={eu:.4f} CE={ce:.4f} risk={risk_penalty:.4f}"
        )
        return option

    def decide(self, options: list[DecisionOption]) -> DecisionOption:
        """Select the best option by certainty equivalent."""
        best = max(options, key=lambda o: o.certainty_equivalent)
        self._decisions.append({
            "chosen": best.name,
            "certainty_equivalent": best.certainty_equivalent,
            "options_evaluated": len(options),
        })
        logger.info(
            f"Decision: '{best.name}' "
            f"(CE={best.certainty_equivalent:.4f})"
        )
        return best

    def minimax(self, options: list[DecisionOption]) -> DecisionOption:
        """Minimax decision: minimize worst-case loss."""
        worst_cases = []
        for opt in options:
            if opt.outcomes:
                worst = min(
                    opt.utilities.get(o, 0.0) * opt.outcomes.get(o, 0.0)
                    for o in opt.outcomes
                )
            else:
                worst = 0.0
            worst_cases.append((worst, opt))
        best = max(worst_cases, key=lambda x: x[0])[1]
        logger.info(f"Minimax decision: '{best.name}'")
        return best

    def regret_matrix(
        self,
        options: list[DecisionOption],
        scenario_utilities: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """
        Compute minimax regret for each option.

        Regret(o, s) = max_o'(U(o', s)) - U(o, s)
        """
        regrets: dict[str, float] = {}
        for opt_name, scenario_utils in scenario_utilities.items():
            max_util = max(scenario_utils.values()) if scenario_utils else 0.0
            opt_util = sum(
                scenario_utils.get(s, 0.0) for s in scenario_utils
            ) / max(1, len(scenario_utils))
            regrets[opt_name] = max_util - opt_util
        return regrets


# ════════════════════════════════════════════════════════════════
# BAYESIAN HYPOTHESIS TESTING
# ════════════════════════════════════════════════════════════════

class BayesFactorTest:
    """
    Bayesian hypothesis testing via Bayes Factors.

    BF = P(E|H1) / P(E|H0)

    Interpretation (Jeffreys scale):
        BF < 1:       supports H0
        1 < BF < 3:   anecdotal for H1
        3 < BF < 10:  substantial for H1
        10 < BF < 30: strong for H1
        30 < BF < 100: very strong for H1
        BF > 100:     decisive for H1
    """

    JEFFREYS_THRESHOLDS = [
        (1.0, "anecdotal"),
        (3.0, "substantial"),
        (10.0, "strong"),
        (30.0, "very strong"),
        (100.0, "decisive"),
    ]

    def test(
        self,
        likelihood_h1: float,
        likelihood_h0: float,
        prior_h1: float = 0.5,
        prior_h0: float = 0.5,
        h0_name: str = "H0",
        h1_name: str = "H1",
    ) -> BayesTest:
        """Run a Bayesian hypothesis test."""
        bf = likelihood_h1 / max(1e-10, likelihood_h0)

        # Posterior via Bayes' theorem
        marginal = likelihood_h1 * prior_h1 + likelihood_h0 * prior_h0
        h1_post = (likelihood_h1 * prior_h1) / max(1e-10, marginal)
        h0_post = 1.0 - h1_post

        # Interpret strength
        interpretation = "anecdotal"
        for threshold, label in self.JEFFREYS_THRESHOLDS:
            if bf >= threshold:
                interpretation = label

        test = BayesTest(
            h0_name=h0_name,
            h1_name=h1_name,
            bayes_factor=bf,
            interpretation=interpretation,
            h0_posterior=h0_post,
            h1_posterior=h1_post,
            evidence_strength=interpretation,
        )
        logger.info(
            f"Bayes Factor test: BF={bf:.2f} → {interpretation} "
            f"for {h1_name} (P(H1|E)={h1_post:.4f})"
        )
        return test

    def sequential_test(
        self,
        likelihoods_h1: list[float],
        likelihoods_h0: list[float],
        prior_h1: float = 0.5,
        prior_h0: float = 0.5,
        stop_threshold: float = 10.0,
    ) -> list[BayesTest]:
        """
        Run sequential Bayesian testing until evidence is decisive.

        Returns list of tests at each step.
        """
        tests = []
        running_prior_h1 = prior_h1
        running_prior_h0 = prior_h0

        for i, (lh1, lh0) in enumerate(zip(likelihoods_h1, likelihoods_h0)):
            t = self.test(lh1, lh0, running_prior_h1, running_prior_h0)
            tests.append(t)

            # Update priors for next step
            running_prior_h1 = t.h1_posterior
            running_prior_h0 = t.h0_posterior

            if t.bayes_factor >= stop_threshold:
                logger.info(f"Sequential test stopped at step {i+1}: {t.evidence_strength}")
                break

        return tests


# ════════════════════════════════════════════════════════════════
# PROBABILITY UTILITIES
# ════════════════════════════════════════════════════════════════

def normalize_distribution(dist: dict[str, float]) -> dict[str, float]:
    """Normalize a probability distribution to sum to 1.0."""
    total = sum(dist.values())
    if total == 0:
        return {k: 1.0 / len(dist) for k in dist}
    return {k: v / total for k, v in dist.items()}


def joint_probability(
    p_a: float, p_b_given_a: float, independent: bool = False, p_b: float = 0.5,
) -> float:
    """
    Compute joint probability P(A, B).

    If independent: P(A, B) = P(A) * P(B)
    Otherwise: P(A, B) = P(A) * P(B|A)
    """
    if independent:
        return p_a * p_b
    return p_a * p_b_given_a


def conditional_probability(p_a_and_b: float, p_b: float) -> float:
    """Compute P(A|B) = P(A,B) / P(B)."""
    if p_b == 0:
        return 0.0
    return p_a_and_b / p_b


def total_probability(
    p_e_given_h: dict[str, float],
    p_h: dict[str, float],
) -> float:
    """
    Compute P(E) = Σ P(E|H_i) * P(H_i) — total probability theorem.
    """
    return sum(
        p_e_given_h.get(h, 0.0) * p_h.get(h, 0.0)
        for h in set(p_e_given_h) | set(p_h)
    )


def odds(p: float) -> float:
    """Convert probability to odds: p / (1-p)."""
    return p / max(1e-10, 1.0 - p)


def probability_from_odds(o: float) -> float:
    """Convert odds to probability: o / (1+o)."""
    return o / (1.0 + o)


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """
    KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

    Measures how different distribution Q is from distribution P.
    """
    kl = 0.0
    for x in set(p) | set(q):
        px = p.get(x, 1e-10)
        qx = q.get(x, 1e-10)
        if px > 0:
            kl += px * math.log(px / qx)
    return kl


def js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """
    Jensen-Shannon divergence — symmetric version of KL divergence.

    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)
    """
    all_keys = set(p) | set(q)
    m = {}
    for x in all_keys:
        m[x] = 0.5 * (p.get(x, 0.0) + q.get(x, 0.0))

    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
