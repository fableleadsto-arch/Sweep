"""
Fuzzy Logic Reasoning — multi-valued logic for uncertain evidence.

Implements fuzzy set operations and fuzzy reasoning:

    Membership functions:
        μ_A(x) ∈ [0, 1] — degree to which x belongs to set A

    Fuzzy operators:
        AND (intersection): μ(x) = min(μ_A(x), μ_B(x))
        OR  (union):        μ(x) = max(μ_A(x), μ_B(x))
        NOT (complement):   μ(x) = 1 - μ_A(x)
        IMPLIES:            μ(x) = min(1, 1 - μ_A(x) + μ_B(x))

    Defuzzification:
        Centroid: x* = Σ μ(x_i) * x_i / Σ μ(x_i)

    Fuzzy rules:
        IF evidence IS strong AND source IS reliable
        THEN conclusion IS trustworthy

Used for:
- Handling vague/uncertain evidence ("strongly suggests", "weakly contradicts")
- Combining multiple uncertain signals
- Grading evidence with fuzzy categories

All operations are logged.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("sweep.math.fuzzy")


@dataclass
class FuzzySet:
    """A fuzzy set with membership function."""
    name: str
    membership: dict[str, float]  # element → membership degree [0,1]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FuzzyRule:
    """A fuzzy IF-THEN rule."""
    name: str
    antecedents: list[tuple[str, str, str]]  # (variable, operator, fuzzy_set_name)
    consequent: tuple[str, str]  # (variable, fuzzy_set_name)
    weight: float = 1.0
    confidence: float = 1.0


@dataclass
class FuzzyResult:
    """Result of fuzzy inference."""
    defuzzified: float
    membership_degrees: dict[str, float]
    activated_rules: list[str]
    reasoning_trace: list[str]


# ════════════════════════════════════════════════════════════════
# MEMBERSHIP FUNCTIONS
# ════════════════════════════════════════════════════════════════

def triangular_mf(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function.

    μ(x) = max(0, min((x-a)/(b-a), (c-x)/(c-b)))

    Parameters:
        a: left foot
        b: peak
        c: right foot
    """
    if x <= a or x >= c:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if (b - a) > 0 else 0.0
    return (c - x) / (c - b) if (c - b) > 0 else 0.0


def trapezoidal_mf(x: float, a: float, b: float, c: float, d: float) -> float:
    """
    Trapezoidal membership function.

    μ(x) = max(0, min((x-a)/(b-a), 1, (d-x)/(d-c)))
    """
    if x <= a or x >= d:
        return 0.0
    if x <= b:
        return (x - a) / (b - a) if (b - a) > 0 else 0.0
    if x <= c:
        return 1.0
    return (d - x) / (d - c) if (d - c) > 0 else 0.0


def gaussian_mf(x: float, mean: float, sigma: float) -> float:
    """
    Gaussian membership function.

    μ(x) = exp(-(x - mean)² / (2σ²))
    """
    if sigma <= 0:
        return 1.0 if x == mean else 0.0
    return math.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def sigmoid_mf(x: float, center: float, steepness: float) -> float:
    """
    Sigmoid membership function.

    μ(x) = 1 / (1 + exp(-steepness * (x - center)))
    """
    z = -steepness * (x - center)
    # Clip to prevent overflow
    if z > 500:
        return 0.0
    if z < -500:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def singleton_mf(x: float, value: float) -> float:
    """Singleton (crisp) membership function."""
    return 1.0 if x == value else 0.0


# ════════════════════════════════════════════════════════════════
# FUZZY OPERATORS
# ════════════════════════════════════════════════════════════════

def fuzzy_and(a: float, b: float) -> float:
    """Fuzzy AND (minimum t-norm)."""
    return min(a, b)


def fuzzy_or(a: float, b: float) -> float:
    """Fuzzy OR (maximum t-conorm)."""
    return max(a, b)


def fuzzy_not(a: float) -> float:
    """Fuzzy NOT (complement)."""
    return 1.0 - a


def fuzzy_implies(a: float, b: float) -> float:
    """Fuzzy IMPLIES (Łukasiewicz)."""
    return min(1.0, 1.0 - a + b)


def fuzzy_xor(a: float, b: float) -> float:
    """Fuzzy XOR."""
    return max(fuzzy_and(a, fuzzy_not(b)), fuzzy_and(fuzzy_not(a), b))


def probabilistic_and(a: float, b: float) -> float:
    """Probabilistic AND (product t-norm)."""
    return a * b


def probabilistic_or(a: float, b: float) -> float:
    """Probabilistic OR (probabilistic sum)."""
    return a + b - a * b


def bounded_and(a: float, b: float) -> float:
    """Bounded AND (drastic t-norm)."""
    return max(0.0, a + b - 1.0)


def bounded_or(a: float, b: float) -> float:
    """Bounded OR."""
    return min(1.0, a + b)


# ════════════════════════════════════════════════════════════════
# FUZZY INFERENCE ENGINE
# ════════════════════════════════════════════════════════════════

class FuzzyReasoner:
    """
    Mamdani-style fuzzy inference engine.

    Process:
        1. Fuzzify inputs (convert crisp values to membership degrees)
        2. Evaluate rules (apply fuzzy operators)
        3. Aggregate rule outputs
        4. Defuzzify (convert fuzzy output to crisp value)
    """

    def __init__(self) -> None:
        self._fuzzy_sets: dict[str, dict[str, FuzzySet]] = {}  # variable → set_name → FuzzySet
        self._rules: list[FuzzyRule] = []
        self._inference_count = 0
        logger.info("FuzzyReasoner initialized")

    def add_fuzzy_set(
        self,
        variable: str,
        set_name: str,
        membership_dict: dict[str, float],
    ) -> None:
        """Add a fuzzy set for a variable."""
        if variable not in self._fuzzy_sets:
            self._fuzzy_sets[variable] = {}
        self._fuzzy_sets[variable][set_name] = FuzzySet(
            name=set_name, membership=membership_dict,
        )
        logger.debug(f"Added fuzzy set '{set_name}' for variable '{variable}'")

    def add_rule(self, rule: FuzzyRule) -> None:
        """Add a fuzzy rule."""
        self._rules.append(rule)
        logger.debug(f"Added rule: {rule.name}")

    def fuzzify(
        self,
        variable: str,
        value: float,
        membership_functions: dict[str, tuple[str, tuple[float, ...]]] | None = None,
    ) -> dict[str, float]:
        """
        Fuzzify a crisp input value.

        Returns dict of set_name → membership_degree.
        """
        result: dict[str, float] = {}

        # If explicit membership functions provided
        if membership_functions:
            for set_name, (mf_type, params) in membership_functions.items():
                if mf_type == "triangular":
                    result[set_name] = triangular_mf(value, *params)
                elif mf_type == "trapezoidal":
                    result[set_name] = trapezoidal_mf(value, *params)
                elif mf_type == "gaussian":
                    result[set_name] = gaussian_mf(value, *params)
                elif mf_type == "sigmoid":
                    result[set_name] = sigmoid_mf(value, *params)
        else:
            # Use stored fuzzy sets
            sets = self._fuzzy_sets.get(variable, {})
            for set_name, fs in sets.items():
                if str(value) in fs.membership:
                    result[set_name] = fs.membership[str(value)]
                else:
                    # Interpolate from nearest values
                    result[set_name] = self._interpolate_membership(
                        fs.membership, value,
                    )

        logger.info(f"Fuzzify({variable}={value}): {result}")
        return result

    def evaluate_rule(
        self,
        rule: FuzzyRule,
        inputs: dict[str, dict[str, float]],
    ) -> tuple[float, list[str]]:
        """
        Evaluate a single fuzzy rule.

        Returns (rule_strength, reasoning_trace).
        """
        trace: list[str] = []
        strengths: list[float] = []

        for var, op, set_name in rule.antecedents:
            membership = inputs.get(var, {}).get(set_name, 0.0)
            strengths.append(membership)
            trace.append(f"  {var} IS {set_name} → μ={membership:.4f}")

        # Combine antecedents
        if not strengths:
            return 0.0, trace

        result = strengths[0]
        for i in range(1, len(strengths)):
            if op == "AND" or op == "and":
                result = fuzzy_and(result, strengths[i])
            elif op == "OR" or op == "or":
                result = fuzzy_or(result, strengths[i])
            else:
                result = fuzzy_and(result, strengths[i])

        # Apply rule weight
        result *= rule.weight

        trace_str = f"Rule '{rule.name}': strength={result:.4f}"
        trace.append(trace_str)
        logger.debug(trace_str)

        return result, trace

    def infer(
        self,
        inputs: dict[str, dict[str, float]],
    ) -> FuzzyResult:
        """
        Run full fuzzy inference on inputs.

        Args:
            inputs: dict of variable_name → {set_name: membership_degree}

        Returns:
            FuzzyResult with defuzzified output and reasoning trace.
        """
        rule_strengths: dict[str, float] = {}
        all_trace: list[str] = []

        for rule in self._rules:
            strength, trace = self.evaluate_rule(rule, inputs)
            rule_strengths[rule.name] = strength
            all_trace.extend(trace)

            # Aggregate consequent
            consequent_var, consequent_set = rule.consequent
            logger.debug(f"Rule '{rule.name}': {consequent_var} IS {consequent_set} (μ={strength:.4f})")

        # Defuzzify using centroid-like method
        # For simplicity, compute weighted average of rule strengths
        active_rules = {k: v for k, v in rule_strengths.items() if v > 0}
        if active_rules:
            defuzzified = sum(active_rules.values()) / len(active_rules)
        else:
            defuzzified = 0.0

        self._inference_count += 1
        logger.info(f"Fuzzy inference #{self._inference_count}: defuzzified={defuzzified:.4f}")

        return FuzzyResult(
            defuzzified=defuzzified,
            membership_degrees=rule_strengths,
            activated_rules=list(active_rules.keys()),
            reasoning_trace=all_trace,
        )

    def _interpolate_membership(
        self, membership: dict[str, float], value: float,
    ) -> float:
        """Interpolate membership degree from nearest stored values."""
        if not membership:
            return 0.0

        try:
            numeric_items = [(float(k), v) for k, v in membership.items()]
        except ValueError:
            return 0.0

        numeric_items.sort(key=lambda x: x[0])

        # Find bounding interval
        for i in range(len(numeric_items) - 1):
            x0, y0 = numeric_items[i]
            x1, y1 = numeric_items[i + 1]
            if x0 <= value <= x1:
                if x1 - x0 == 0:
                    return y0
                t = (value - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)

        # Outside range
        if value < numeric_items[0][0]:
            return numeric_items[0][1]
        return numeric_items[-1][1]


# ════════════════════════════════════════════════════════════════
# FUZZY EVIDENCE GRADER
# ════════════════════════════════════════════════════════════════

class FuzzyEvidenceGrader:
    """
    Grade evidence using fuzzy logic instead of crisp thresholds.

    Instead of:
        if confidence > 0.7: grade = "high"
    Uses:
        μ_high(0.7) = 0.67
        μ_medium(0.7) = 0.33
        grade = weighted combination

    More natural handling of borderline cases.
    """

    def __init__(self) -> None:
        self._reasoner = FuzzyReasoner()
        self._setup_evidence_sets()
        logger.info("FuzzyEvidenceGrader initialized")

    def _setup_evidence_sets(self) -> None:
        """Set up fuzzy sets for evidence grading."""
        # Evidence strength fuzzy sets
        for var, sets in [
            ("strength", {
                "weak": {"0.0": 1.0, "0.2": 1.0, "0.3": 0.5, "0.4": 0.0},
                "moderate": {"0.2": 0.0, "0.3": 0.5, "0.5": 1.0, "0.7": 0.5, "0.8": 0.0},
                "strong": {"0.6": 0.0, "0.7": 0.5, "0.8": 1.0, "1.0": 1.0},
            }),
            ("reliability", {
                "unreliable": {"0.0": 1.0, "0.3": 1.0, "0.4": 0.5, "0.5": 0.0},
                "moderate": {"0.3": 0.0, "0.4": 0.5, "0.6": 1.0, "0.8": 0.5, "0.9": 0.0},
                "reliable": {"0.7": 0.0, "0.8": 0.5, "0.9": 1.0, "1.0": 1.0},
            }),
            ("coherence", {
                "incoherent": {"0.0": 1.0, "0.3": 1.0, "0.4": 0.5, "0.5": 0.0},
                "partial": {"0.3": 0.0, "0.4": 0.5, "0.6": 1.0, "0.8": 0.5, "0.9": 0.0},
                "coherent": {"0.7": 0.0, "0.8": 0.5, "0.9": 1.0, "1.0": 1.0},
            }),
        ]:
            for set_name, members in sets.items():
                self._reasoner.add_fuzzy_set(var, set_name, members)

        # Add rules
        self._reasoner.add_rule(FuzzyRule(
            name="high_quality",
            antecedents=[("strength", "and", "strong"), ("reliability", "and", "reliable")],
            consequent=("quality", "high"),
            weight=1.0,
        ))
        self._reasoner.add_rule(FuzzyRule(
            name="medium_quality",
            antecedents=[("strength", "and", "moderate"), ("reliability", "and", "moderate")],
            consequent=("quality", "medium"),
            weight=0.7,
        ))
        self._reasoner.add_rule(FuzzyRule(
            name="low_quality",
            antecedents=[("strength", "and", "weak"), ("reliability", "and", "unreliable")],
            consequent=("quality", "low"),
            weight=0.4,
        ))

    def grade(
        self,
        strength: float,
        reliability: float,
        coherence: float = 0.5,
    ) -> FuzzyResult:
        """
        Grade evidence using fuzzy logic.

        Args:
            strength: evidence strength (0-1)
            reliability: source reliability (0-1)
            coherence: internal coherence (0-1)

        Returns:
            FuzzyResult with graded quality.
        """
        inputs = {
            "strength": {"strong": strength, "moderate": strength, "weak": strength},
            "reliability": {"reliable": reliability, "moderate": reliability, "unreliable": reliability},
        }

        result = self._reasoner.infer(inputs)
        logger.info(
            f"FuzzyGrade: strength={strength:.2f} reliability={reliability:.2f} "
            f"→ quality={result.defuzzified:.4f}"
        )
        return result
