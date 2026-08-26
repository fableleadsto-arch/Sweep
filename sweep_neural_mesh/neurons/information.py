"""
Information Theory — entropy, mutual information, information gain.

Implements the information-theoretic backbone for Sweep's reasoning:

    Shannon Entropy:
        H(X) = -Σ P(x) * log₂(P(x))

    Conditional Entropy:
        H(X|Y) = -Σ P(x,y) * log₂(P(x|y))

    Mutual Information:
        I(X;Y) = H(X) - H(X|Y)

    Information Gain:
        IG(S,A) = H(S) - Σ (|S_v|/|S|) * H(S_v)

    Cross-Entropy:
        H(p,q) = -Σ p(x) * log₂(q(x))

    Perplexity:
        PP = 2^H(p,q)

Used for:
- Evidence quality measurement (entropy of evidence distribution)
- Feature selection (information gain for ranking evidence)
- Anomaly detection (low probability = high surprise)
- Model comparison (cross-entropy between predicted and actual)

All computations are logged.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("sweep.math.information")


@dataclass
class EntropyResult:
    """Result of an entropy computation."""
    entropy: float           # in bits (log base 2)
    entropy_nats: float      # in nats (natural log)
    max_possible: float      # log₂(N) for N outcomes
    normalized: float        # H / H_max (0 to 1)
    distribution_size: int   # number of outcomes


@dataclass
class MutualInfoResult:
    """Result of mutual information computation."""
    mi_xy: float             # I(X;Y) in bits
    mi_normalized: float     # normalized MI (0 to 1)
    h_x: float              # H(X)
    h_y: float              # H(Y)
    h_x_given_y: float      # H(X|Y)
    h_y_given_x: float      # H(Y|X)
    interpretation: str


@dataclass
class InformationGainResult:
    """Result of information gain computation."""
    parent_entropy: float
    weighted_child_entropy: float
    information_gain: float
    gain_ratio: float        # IG / split_info
    split_attribute: str
    child_entropies: dict[str, float]
    child_sizes: dict[str, int]


class InformationTheory:
    """
    Information theory engine for Sweep's reasoning system.

    Computes entropy, mutual information, and information gain
    to measure evidence quality, feature relevance, and
    information content of reasoning processes.
    """

    def __init__(self) -> None:
        self._entropy_cache: dict[str, float] = {}
        self._total_computations = 0
        logger.info("InformationTheory engine initialized")

    # ════════════════════════════════════════════════════════════════
    # ENTROPY
    # ════════════════════════════════════════════════════════════════

    def shannon_entropy(self, probabilities: dict[str, float]) -> EntropyResult:
        """
        Compute Shannon entropy of a distribution.

        H(X) = -Σ P(x) * log₂(P(x))

        Maximum entropy = log₂(N) (uniform distribution)
        Zero entropy = deterministic (one outcome has probability 1)
        """
        # Normalize
        total = sum(probabilities.values())
        if total == 0:
            return EntropyResult(0.0, 0.0, 0.0, 0.0, 0)

        probs = [p / total for p in probabilities.values() if p > 0]
        n = len(probs)

        # Shannon entropy in bits
        h_bits = -sum(p * math.log2(p) for p in probs if p > 0)

        # In nats
        h_nats = -sum(p * math.log(p) for p in probs if p > 0)

        # Maximum possible entropy
        h_max = math.log2(n) if n > 0 else 0.0

        # Normalized entropy
        normalized = h_bits / h_max if h_max > 0 else 0.0

        result = EntropyResult(
            entropy=h_bits, entropy_nats=h_nats,
            max_possible=h_max, normalized=normalized,
            distribution_size=n,
        )

        self._total_computations += 1
        logger.info(
            f"Entropy: H={h_bits:.4f} bits (max={h_max:.4f}, "
            f"norm={normalized:.4f}, n={n})"
        )
        return result

    def conditional_entropy(
        self,
        joint: dict[tuple[str, str], float],
        marginal_y: dict[str, float],
    ) -> float:
        """
        Compute conditional entropy H(X|Y).

        H(X|Y) = -Σ P(x,y) * log₂(P(x|y))
        """
        h_xy = 0.0
        total = sum(joint.values())
        if total == 0:
            return 0.0

        for (x, y), p_xy in joint.items():
            if p_xy <= 0:
                continue
            p_y = marginal_y.get(y, 0.0)
            if p_y <= 0:
                continue
            p_x_given_y = p_xy / p_y
            if p_x_given_y > 0:
                h_xy -= p_xy * math.log2(p_x_given_y)

        logger.info(f"Conditional entropy H(X|Y)={h_xy:.4f} bits")
        return h_xy

    def cross_entropy(
        self,
        actual: dict[str, float],
        predicted: dict[str, float],
    ) -> float:
        """
        Cross-entropy: H(actual, predicted) = -Σ actual(x) * log₂(predicted(x))

        Measures how different predicted distribution is from actual.
        Lower is better. Equal to entropy when predicted = actual.
        """
        total_a = sum(actual.values())
        total_p = sum(predicted.values())
        if total_a == 0 or total_p == 0:
            return 0.0

        ce = 0.0
        for x in actual:
            pa = actual[x] / total_a
            pp = predicted.get(x, 1e-10) / total_p
            if pa > 0 and pp > 0:
                ce -= pa * math.log2(pp)

        logger.info(f"Cross-entropy H(p,q)={ce:.4f} bits")
        return ce

    def perplexity(
        self,
        actual: dict[str, float],
        predicted: dict[str, float],
    ) -> float:
        """
        Perplexity = 2^H(actual, predicted)

        Lower perplexity = better model.
        PP = 1 means perfect prediction.
        """
        ce = self.cross_entropy(actual, predicted)
        pp = 2.0 ** ce
        logger.info(f"Perplexity={pp:.4f}")
        return pp

    # ════════════════════════════════════════════════════════════════
    # MUTUAL INFORMATION
    # ════════════════════════════════════════════════════════════════

    def mutual_information(
        self,
        joint: dict[tuple[str, str], float],
        marginal_x: dict[str, float],
        marginal_y: dict[str, float],
    ) -> MutualInfoResult:
        """
        Compute mutual information I(X;Y).

        I(X;Y) = H(X) + H(Y) - H(X,Y)
                = H(X) - H(X|Y)
                = H(Y) - H(Y|X)

        Measures how much knowing Y reduces uncertainty about X.
        """
        h_x = self.shannon_entropy(marginal_x).entropy
        h_y = self.shannon_entropy(marginal_y).entropy

        # Joint entropy H(X,Y)
        total = sum(joint.values())
        h_xy = 0.0
        if total > 0:
            for p in joint.values():
                if p > 0:
                    p_norm = p / total
                    h_xy -= p_norm * math.log2(p_norm)

        # Conditional entropies
        h_x_given_y = h_xy - h_y
        h_y_given_x = h_xy - h_x

        # Mutual information
        mi = h_x - h_x_given_y
        mi = max(0.0, mi)  # MI is non-negative

        # Normalized MI (0 to 1)
        min_entropy = min(h_x, h_y)
        mi_normalized = mi / min_entropy if min_entropy > 0 else 0.0

        # Interpretation
        if mi_normalized < 0.1:
            interpretation = "negligible"
        elif mi_normalized < 0.3:
            interpretation = "weak"
        elif mi_normalized < 0.5:
            interpretation = "moderate"
        elif mi_normalized < 0.7:
            interpretation = "strong"
        else:
            interpretation = "very strong"

        result = MutualInfoResult(
            mi_xy=mi, mi_normalized=mi_normalized,
            h_x=h_x, h_y=h_y,
            h_x_given_y=h_x_given_y, h_y_given_x=h_y_given_x,
            interpretation=interpretation,
        )

        self._total_computations += 1
        logger.info(
            f"MI(X;Y)={mi:.4f} bits (normalized={mi_normalized:.4f}, "
            f"{interpretation})"
        )
        return result

    # ════════════════════════════════════════════════════════════════
    # INFORMATION GAIN
    # ════════════════════════════════════════════════════════════════

    def information_gain(
        self,
        parent_labels: list[str],
        child_groups: dict[str, list[str]],
        attribute_name: str = "attribute",
    ) -> InformationGainResult:
        """
        Compute information gain for splitting data.

        IG(S,A) = H(S) - Σ (|S_v|/|S|) * H(S_v)

        Used for feature selection: which evidence attribute
        provides the most information for classification?
        """
        # Parent entropy
        parent_dist = self._label_distribution(parent_labels)
        h_parent = self.shannon_entropy(parent_dist).entropy

        # Weighted child entropy
        total_size = len(parent_labels)
        weighted_h_child = 0.0
        child_entropies: dict[str, float] = {}
        child_sizes: dict[str, int] = {}

        for group_name, group_labels in child_groups.items():
            if not group_labels:
                continue
            group_dist = self._label_distribution(group_labels)
            h_group = self.shannon_entropy(group_dist).entropy
            weight = len(group_labels) / max(1, total_size)
            weighted_h_child += weight * h_group
            child_entropies[group_name] = h_group
            child_sizes[group_name] = len(group_labels)

        ig = h_parent - weighted_h_child

        # Split information (for gain ratio)
        split_sizes = {k: len(v) for k, v in child_groups.items() if v}
        split_dist = {k: v / max(1, total_size) for k, v in split_sizes.items()}
        split_info_result = self.shannon_entropy(split_dist)
        split_info = split_info_result.entropy

        gain_ratio = ig / split_info if split_info > 0 else 0.0

        result = InformationGainResult(
            parent_entropy=h_parent,
            weighted_child_entropy=weighted_h_child,
            information_gain=ig,
            gain_ratio=gain_ratio,
            split_attribute=attribute_name,
            child_entropies=child_entropies,
            child_sizes=child_sizes,
        )

        self._total_computations += 1
        logger.info(
            f"Information Gain({attribute_name}): IG={ig:.4f} "
            f"(parent H={h_parent:.4f}, child H={weighted_h_child:.4f}, "
            f"gain_ratio={gain_ratio:.4f})"
        )
        return result

    # ════════════════════════════════════════════════════════════════
    # SURPRISE / ANOMALY DETECTION
    # ════════════════════════════════════════════════════════════════

    def surprise(self, probability: float) -> float:
        """
        Self-information (surprise) of an event.

        I(x) = -log₂(P(x))

        Low probability = high surprise.
        """
        if probability <= 0:
            return float('inf')
        return -math.log2(probability)

    def surprisal_rank(
        self, events: dict[str, float],
    ) -> list[tuple[str, float]]:
        """Rank events by surprise (most surprising first)."""
        ranked = [(name, self.surprise(prob)) for name, prob in events.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def is_surprising(
        self, probability: float, threshold_bits: float = 3.0,
    ) -> bool:
        """
        Is this event surprising? (more than threshold_bits of surprise)

        threshold_bits=3.0 means probability < 1/8 = 12.5%
        """
        return self.surprise(probability) > threshold_bits

    # ════════════════════════════════════════════════════════════════
    # UTILITIES
    # ════════════════════════════════════════════════════════════════

    def _label_distribution(self, labels: list[str]) -> dict[str, float]:
        """Convert a list of labels to a probability distribution."""
        counts: dict[str, float] = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1.0
        return counts

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about information theory computations."""
        return {
            "total_computations": self._total_computations,
        }
