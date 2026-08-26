"""
Common Sense Knowledge Base — default assumptions about the world.

Humans never state the obvious, but we all know:
- Objects fall when dropped
- People generally tell the truth
- Wet floors are slippery
- Fires are hot
- If you cut yourself, it hurts

Without common sense, AI treats all claims as equally plausible
and can't catch obviously absurd conclusions.

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │           COMMON SENSE KNOWLEDGE BASE                │
    │                                                     │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Physical Defaults                            │  │
    │  │  (gravity, thermodynamics, material properties)│  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Social Defaults                              │  │
    │  │  (people have goals, actions imply intentions)│  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Temporal Defaults                            │  │
    │  │  (events follow sequences, causes precede)    │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Causal Defaults                              │  │
    │  │  (actions have consequences, effort → results)│  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Biological Defaults                          │  │
    │  │  (organisms, health, survival instincts)      │  │
    │  └──────────────────────────────────────────────┘  │
    │  ┌──────────────────────────────────────────────┐  │
    │  │  Mathematical Defaults                        │  │
    │  │  (logic, quantities, axioms)                  │  │
    │  └──────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CommonSenseRule:
    """A single common sense rule/assumption."""
    rule_id: str
    category: str               # "physical", "social", "temporal", "causal", "biological", "mathematical"
    premise: str                # "when X happens"
    conclusion: str             # "then Y typically follows"
    confidence: float           # 0.0-1.0: how reliable is this default
    exceptions: list[str]       # known exceptions to this rule
    source: str = "learned"     # "innate", "learned", "inferred"
    use_count: int = 0
    last_used: float = field(default_factory=time.time)


@dataclass
class CommonSenseCheck:
    """Result of checking a claim against common sense."""
    claim: str
    is_plausible: bool
    violated_rules: list[str]   # rules that this claim violates
    supported_rules: list[str]  # rules that support this claim
    plausibility_score: float   # 0.0-1.0
    reasoning: str


# ─── Synonym groups for semantic matching ───
_SYNONYM_GROUPS: list[list[str]] = [
    ["fall", "drop", "descend", "sink", "plummet"],
    ["rise", "increase", "grow", "climb", "ascend", "expand"],
    ["decrease", "decline", "reduce", "shrink", "fall", "diminish"],
    ["break", "shatter", "crack", "fracture", "snap"],
    ["melt", "dissolve", "liquefy", "thaw"],
    ["burn", "heat", "scorch", "ignite", "combust"],
    ["freeze", "solidify", "harden", "crystallize"],
    ["flow", "pour", "stream", "trickle", "gush"],
    ["cause", "lead", "produce", "result", "generate", "create"],
    ["prevent", "block", "stop", "hinder", "impede", "inhibit"],
    ["help", "assist", "support", "aid", "facilitate", "enable"],
    ["hurt", "harm", "damage", "injure", "wound"],
    ["alive", "living", "viable", "animate"],
    ["dead", "deceased", "lifeless", "nonviable", "extinct"],
    ["eat", "consume", "ingest", "feed"],
    ["sleep", "rest", "slumber", "nap"],
    ["move", "travel", "go", "walk", "run", "navigate"],
    ["think", "reason", "consider", "ponder", "deliberate"],
    ["speak", "say", "tell", "state", "declare", "assert"],
    ["buy", "purchase", "acquire", "obtain"],
    ["sell", "trade", "exchange", "barter"],
    ["money", "currency", "cash", "funds", "capital"],
    ["fast", "quick", "rapid", "swift", "speedy"],
    ["slow", "gradual", "delayed", "sluggish"],
    ["big", "large", "huge", "massive", "enormous", "vast"],
    ["small", "tiny", "little", "minute", "微型"],
    ["good", "positive", "beneficial", "favorable", "advantageous"],
    ["bad", "negative", "harmful", "detrimental", "disadvantageous"],
]

# Build reverse lookup: word → group index
_SYNONYM_LOOKUP: dict[str, int] = {}
for _i, _group in enumerate(_SYNONYM_GROUPS):
    for _word in _group:
        _SYNONYM_LOOKUP[_word] = _i

# Category-specific base confidence weights
_CATEGORY_CONFIDENCE: dict[str, float] = {
    "physical": 0.95,
    "biological": 0.90,
    "mathematical": 0.99,
    "temporal": 0.90,
    "causal": 0.85,
    "social": 0.75,
}

# Extended negation patterns
_NEGATION_PATTERNS: list[str] = [
    "not", "never", "no", "doesn't", "isn't", "won't", "can't",
    "cannot", "unable", "lack", "lacks", "lacking", "absent",
    "absence", "fails", "failed", "without", "neither", "nor",
    "impossible", "nobody", "nothing", "nowhere", "hardly",
    "barely", "seldom", "rarely", "deny", "denies", "refuse",
]


class CommonSense:
    """
    A knowledge base of default assumptions about the world.

    Like the vast background knowledge that humans take for granted,
    this module maintains default rules that act as priors for reasoning:

    1. PHYSICAL: Objects fall, fire burns, water flows downhill
    2. SOCIAL: People have goals, actions imply intentions, trust is earned
    3. TEMPORAL: Causes precede effects, events follow sequences
    4. CAUSAL: Effort leads to results, actions have consequences
    5. BIOLOGICAL: Organisms need food/water, injuries cause pain, sleep is needed
    6. MATHEMATICAL: Numbers follow logic, parts sum to wholes, zero means none

    When processing evidence, common sense acts as a FAST PRE-FILTER:
    - "This claim says water flows uphill" → VIOLATES physical defaults
    - "This person has no reason to lie" → SUPPORTS social defaults
    - "The effect happened before the cause" → VIOLATES temporal defaults

    Common sense is gradually EXPANDED through learning:
    - Episodes that confirm defaults strengthen them
    - Episodes that contradict defaults create exceptions
    - Frequently confirmed defaults become "innate" (high confidence)
    """

    def __init__(self) -> None:
        self._rules: dict[str, CommonSenseRule] = {}
        self._check_history: list[CommonSenseCheck] = []
        self._next_rule_id = 0
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        """Initialize with expanded common sense rules — 55+ rules across 6 categories."""
        defaults = [
            # ── Physical (12 rules) ──
            ("physical", "objects are dropped", "they fall down", 0.99, ["in space", "in water"]),
            ("physical", "fire touches skin", "it causes burns", 0.99, ["special protective gear"]),
            ("physical", "ice is heated", "it melts", 0.98, ["extremely low pressure"]),
            ("physical", "a container is overfilled", "it spills", 0.97, ["elastic containers"]),
            ("physical", "glass is hit with force", "it breaks", 0.90, ["tempered glass", "flexible glass"]),
            ("physical", "metal is heated to high temperature", "it expands", 0.95, ["invar alloys"]),
            ("physical", "a liquid cools below its freezing point", "it solidifies", 0.97, ["supercooled liquids"]),
            ("physical", "light hits an opaque object", "it casts a shadow", 0.98, ["diffuse light"]),
            ("physical", "a force is applied to a mass", "it accelerates", 0.99, ["counterbalanced forces"]),
            ("physical", "sound travels through air", "it loses intensity with distance", 0.95, ["focused beams"]),
            ("physical", "two objects occupy the same space", "they cannot coexist", 0.99, ["quantum effects"]),
            ("physical", "energy is converted between forms", "some is lost as heat", 0.95, ["superconductors"]),

            # ── Social (10 rules) ──
            ("social", "a person is asked a question", "they attempt to answer honestly", 0.70, ["deception", "ignorance"]),
            ("social", "someone spends years learning", "they become skilled", 0.85, ["poor teaching", "learning disability"]),
            ("social", "a company releases a product", "they want to sell it", 0.90, ["open source", "research"]),
            ("social", "a person makes a claim", "they believe it to be true", 0.75, ["lying", "mistake", "sarcasm"]),
            ("social", "someone recommends something", "they have tried it", 0.60, ["hearsay", "sponsorship"]),
            ("social", "a person works hard at something", "they expect recognition or reward", 0.75, ["altruism", "intrinsic motivation"]),
            ("social", "someone shares personal information", "they trust the listener", 0.80, ["accidental disclosure", "manipulation"]),
            ("social", "an authority figure gives an instruction", "others tend to comply", 0.80, ["rebellion", "disagreement"]),
            ("social", "a contract is signed by both parties", "both parties are bound", 0.90, ["unenforceable contracts", "fraud"]),
            ("social", "a community faces a shared threat", "they cooperate", 0.70, ["tragedy of the commons", "panic"]),

            # ── Temporal (8 rules) ──
            ("temporal", "cause occurs", "effect follows after", 0.95, ["retrocausality claims"]),
            ("temporal", "a document is dated 2024", "it was written in or before 2024", 0.98, ["backdating", "errors"]),
            ("temporal", "technology improves", "newer versions are generally better", 0.70, ["regressions", "feature removal"]),
            ("temporal", "research is published", "findings were observed before publication", 0.95, ["preprints"]),
            ("temporal", "a person is born", "they did not exist before", 0.99, ["reincarnation beliefs"]),
            ("temporal", "an event is recorded", "it happened before the recording", 0.98, ["retroactive editing"]),
            ("temporal", "seasons change", "temperature follows a predictable cycle", 0.90, ["climate change", "local anomalies"]),
            ("temporal", "a machine is used repeatedly", "it wears down over time", 0.95, ["self-repairing systems"]),

            # ── Causal (8 rules) ──
            ("causal", "more effort is applied", "more results are achieved", 0.75, ["diminishing returns", "wrong direction"]),
            ("causal", "a system has many components", "it is more complex to maintain", 0.85, ["modular design"]),
            ("causal", "data is from a trusted source", "it is more likely accurate", 0.80, ["even trusted sources can err"]),
            ("causal", "multiple sources agree", "the claim is more likely true", 0.80, ["shared bias", "common source"]),
            ("causal", "a claim lacks evidence", "it should be treated with caution", 0.90, ["absence of evidence ≠ evidence of absence"]),
            ("causal", "a person practices regularly", "their skill improves", 0.85, ["wrong practice", "plateau"]),
            ("causal", "a problem is ignored", "it typically gets worse", 0.80, ["self-correcting systems"]),
            ("causal", "resources are allocated efficiently", "outcomes improve", 0.80, ["misallocation", "unintended consequences"]),

            # ── Biological (8 rules) ──
            ("biological", "an organism has no food", "it weakens and may die", 0.95, ["hibernation", "dormancy"]),
            ("biological", "an organism has no water", "it dehydrates", 0.98, ["desert-adapted species"]),
            ("biological", "a wound is untreated", "it may become infected", 0.80, ["clean wounds", "strong immune system"]),
            ("biological", "a person does not sleep", "their cognitive function declines", 0.90, ["short-term compensation"]),
            ("biological", "a virus enters a host", "the immune system responds", 0.85, ["immunocompromised", "novel virus"]),
            ("biological", "a person exercises regularly", "their fitness improves", 0.85, ["overtraining", "injury"]),
            ("biological", "a child is exposed to language", "they learn to speak", 0.90, ["language deprivation", "hearing impairment"]),
            ("biological", "a predator hunts prey", "the prey attempts to escape", 0.90, ["camouflage", "defense mechanisms"]),

            # ── Mathematical (9 rules) ──
            ("mathematical", "a number is added to zero", "the result equals the original number", 0.99, []),
            ("mathematical", "a number is multiplied by one", "the result equals the original number", 0.99, []),
            ("mathematical", "a number is divided by zero", "the result is undefined", 0.99, []),
            ("mathematical", "if A equals B and B equals C", "then A equals C", 0.99, ["fuzzy logic", "approximate equality"]),
            ("mathematical", "a whole is divided into parts", "the parts sum to the whole", 0.99, ["rounding errors"]),
            ("mathematical", "a negative number is multiplied by a negative", "the result is positive", 0.99, []),
            ("mathematical", "probability exceeds one", "it is invalid", 0.99, ["non-standard probability"]),
            ("mathematical", "a sequence follows a pattern", "the next element follows the same pattern", 0.90, ["pattern breaks", "noise"]),
            ("mathematical", "an angle exceeds 360 degrees", "it wraps around", 0.95, ["non-Euclidean geometry"]),
        ]

        for category, premise, conclusion, confidence, exceptions in defaults:
            self.add_rule(category, premise, conclusion, confidence, exceptions, "innate")

    def add_rule(
        self,
        category: str,
        premise: str,
        conclusion: str,
        confidence: float = 0.7,
        exceptions: list[str] | None = None,
        source: str = "learned",
    ) -> CommonSenseRule:
        """Add a new common sense rule."""
        self._next_rule_id += 1
        rule = CommonSenseRule(
            rule_id=f"cs_{self._next_rule_id}",
            category=category,
            premise=premise,
            conclusion=conclusion,
            confidence=confidence,
            exceptions=exceptions or [],
            source=source,
        )
        self._rules[rule.rule_id] = rule
        return rule

    def check_claim(self, claim: str) -> CommonSenseCheck:
        """
        Check a claim against common sense rules.

        Returns which rules support or violate the claim, and an
        overall plausibility score.
        """
        claim_lower = claim.lower()
        violated = []
        supported = []

        for rule in self._rules.values():
            if self._claim_related_to_rule(claim_lower, rule):
                if self._claim_violates_rule(claim_lower, rule):
                    violated.append(f"{rule.premise} → {rule.conclusion}")
                elif self._claim_supports_rule(claim_lower, rule):
                    supported.append(f"{rule.premise} → {rule.conclusion}")

        # Weighted plausibility: each violation/support is weighted by rule confidence
        if violated:
            # Find the rules that were violated and weight by their confidence
            violating_rules = [
                r for r in self._rules.values()
                if self._claim_related_to_rule(claim_lower, r)
                and self._claim_violates_rule(claim_lower, r)
            ]
            avg_violation_conf = sum(r.confidence for r in violating_rules) / len(violating_rules)
            plausibility = max(0.0, 0.5 - len(violated) * 0.12 * avg_violation_conf)
        elif supported:
            supporting_rules = [
                r for r in self._rules.values()
                if self._claim_related_to_rule(claim_lower, r)
                and self._claim_supports_rule(claim_lower, r)
            ]
            avg_support_conf = sum(r.confidence for r in supporting_rules) / len(supporting_rules)
            plausibility = min(1.0, 0.5 + len(supported) * 0.08 * avg_support_conf)
        else:
            plausibility = 0.5

        is_plausible = plausibility > 0.3

        reasoning_parts = []
        if violated:
            reasoning_parts.append(f"violates {len(violated)} common sense rule(s)")
        if supported:
            reasoning_parts.append(f"supported by {len(supported)} common sense rule(s)")
        if not reasoning_parts:
            reasoning_parts.append("no common sense signal")

        check = CommonSenseCheck(
            claim=claim,
            is_plausible=is_plausible,
            violated_rules=violated,
            supported_rules=supported,
            plausibility_score=plausibility,
            reasoning="; ".join(reasoning_parts),
        )

        self._check_history.append(check)
        return check

    def _get_synonym_group(self, word: str) -> int | None:
        """Return the synonym group index for a word, or None."""
        return _SYNONYM_LOOKUP.get(word)

    def _words_match(self, word_a: str, word_b: str) -> bool:
        """Check if two words match directly or via synonym groups."""
        if word_a == word_b:
            return True
        group_a = self._get_synonym_group(word_a)
        group_b = self._get_synonym_group(word_b)
        if group_a is not None and group_b is not None:
            return group_a == group_b
        return False

    def _extract_meaningful_words(self, text: str) -> set[str]:
        """Extract meaningful words (3+ chars, not stop words)."""
        stop = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can",
            "had", "her", "was", "one", "our", "out", "that", "this",
            "with", "have", "from", "they", "been", "said", "each",
            "which", "their", "will", "would", "there", "what", "about",
            "than", "into", "some", "them", "other", "when", "could",
            "more", "very", "also", "after", "being", "where", "those",
            "only", "then", "these", "first", "any", "most", "its",
            "over", "such", "make", "like", "just", "your", "does",
        }
        words = set(re.findall(r'\b[a-z]{3,}\b', text))
        return words - stop

    def _claim_related_to_rule(self, claim: str, rule: CommonSenseRule) -> bool:
        """Check if a claim is related to a rule using synonym-aware matching."""
        premise_words = self._extract_meaningful_words(rule.premise.lower())
        conclusion_words = self._extract_meaningful_words(rule.conclusion.lower())
        claim_words = self._extract_meaningful_words(claim)

        rule_words = premise_words | conclusion_words
        if not rule_words or not claim_words:
            return False

        # Direct overlap
        direct_overlap = len(claim_words & rule_words)
        if direct_overlap >= 2:
            return True

        # Synonym-aware overlap: for each claim word, check if any rule word is in the same synonym group
        synonym_overlap = 0
        claim_synonyms_matched: set[int] = set()
        for cw in claim_words:
            cw_group = self._get_synonym_group(cw)
            if cw_group is None:
                continue
            for rw in rule_words:
                rw_group = self._get_synonym_group(rw)
                if rw_group is not None and cw_group == rw_group and cw_group not in claim_synonyms_matched:
                    synonym_overlap += 1
                    claim_synonyms_matched.add(cw_group)
                    break

        # Combined threshold: 2 direct, or 1 direct + 1 synonym, or 2 synonyms
        if direct_overlap + synonym_overlap >= 2:
            return True

        # Lower threshold: if there are 3+ direct overlaps with just 1 word matching
        if direct_overlap >= 1 and len(claim_words) >= 3 and len(rule_words) >= 3:
            return True

        return False

    def _has_negation(self, claim: str) -> bool:
        """Check if the claim contains any negation pattern."""
        for neg in _NEGATION_PATTERNS:
            if neg in claim:
                return True
        return False

    def _claim_violates_rule(self, claim: str, rule: CommonSenseRule) -> bool:
        """Check if a claim violates a rule."""
        has_negation = self._has_negation(claim)

        conclusion_words = self._extract_meaningful_words(rule.conclusion.lower())
        claim_words = self._extract_meaningful_words(claim)

        # Direct overlap between claim and conclusion
        direct_overlap = len(claim_words & conclusion_words)

        # Synonym-aware overlap
        synonym_overlap = 0
        for cw in claim_words:
            cw_group = self._get_synonym_group(cw)
            if cw_group is None:
                continue
            for rw in conclusion_words:
                rw_group = self._get_synonym_group(rw)
                if rw_group is not None and cw_group == rw_group:
                    synonym_overlap += 1
                    break

        total_overlap = direct_overlap + synonym_overlap

        # Negation + overlap with conclusion → violation
        if has_negation and total_overlap >= 1:
            for exception in rule.exceptions:
                if exception.lower() in claim:
                    return False
            return True

        # Explicit opposite words (e.g., "increase" vs rule conclusion "decrease")
        claim_has_opposite = False
        for cw in claim_words:
            cw_group = self._get_synonym_group(cw)
            if cw_group is None:
                continue
            for rw in conclusion_words:
                rw_group = self._get_synonym_group(rw)
                if rw_group is not None and cw_group != rw_group:
                    # Check if they are antonym-like (opposite groups that commonly oppose)
                    if self._are_opposite_groups(cw_group, rw_group):
                        claim_has_opposite = True
                        break
            if claim_has_opposite:
                break

        if claim_has_opposite and total_overlap >= 1:
            for exception in rule.exceptions:
                if exception.lower() in claim:
                    return False
            return True

        return False

    def _are_opposite_groups(self, group_a: int, group_b: int) -> bool:
        """Check if two synonym groups represent opposites."""
        # Pairs of group indices that are antonyms
        _OPPOSITE_PAIRS: list[tuple[int, int]] = [
            (1, 2),   # rise ↔ decrease
            (7, 0),   # flow/move ↔ fall
            (24, 25), # big ↔ small
            (26, 27), # good ↔ bad
            (13, 12), # dead ↔ alive
        ]
        for a, b in _OPPOSITE_PAIRS:
            if (group_a == a and group_b == b) or (group_a == b and group_b == a):
                return True
        return False

    def _claim_supports_rule(self, claim: str, rule: CommonSenseRule) -> bool:
        """Check if a claim supports a rule."""
        conclusion_words = self._extract_meaningful_words(rule.conclusion.lower())
        claim_words = self._extract_meaningful_words(claim)

        direct_overlap = len(claim_words & conclusion_words)
        if direct_overlap >= 2:
            return True

        # Synonym-aware support
        synonym_overlap = 0
        for cw in claim_words:
            cw_group = self._get_synonym_group(cw)
            if cw_group is None:
                continue
            for rw in conclusion_words:
                rw_group = self._get_synonym_group(rw)
                if rw_group is not None and cw_group == rw_group:
                    synonym_overlap += 1
                    break

        return direct_overlap + synonym_overlap >= 2

    def learn_from_episode(
        self,
        claim: str,
        outcome: str,
        was_plausible: bool,
    ) -> None:
        """
        Learn from a reasoning episode to improve common sense.

        If a claim was plausible and turned out true → strengthen supporting rules
        If a claim was implausible and turned out true → create exception
        """
        check = self.check_claim(claim)
        for rule_id, rule in self._rules.items():
            if self._claim_related_to_rule(claim.lower(), rule):
                if was_plausible:
                    if rule.rule_id in [r.split(" →")[0] for r in check.supported_rules]:
                        rule.confidence = min(0.99, rule.confidence + 0.02)
                else:
                    if rule.rule_id in [r.split(" →")[0] for r in check.violated_rules]:
                        if outcome not in rule.exceptions:
                            rule.exceptions.append(outcome[:100])

    def get_rules_by_category(self, category: str) -> list[CommonSenseRule]:
        """Get all rules in a category."""
        return [r for r in self._rules.values() if r.category == category]

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    @property
    def stats(self) -> dict[str, Any]:
        categories: dict[str, int] = {}
        for rule in self._rules.values():
            categories[rule.category] = categories.get(rule.category, 0) + 1
        return {
            "total_rules": len(self._rules),
            "by_category": categories,
            "total_checks": len(self._check_history),
            "avg_confidence": (
                sum(r.confidence for r in self._rules.values()) / len(self._rules)
                if self._rules else 0.0
            ),
        }
