"""
Causal Handler — cause-effect reasoning and chain analysis.

Handles:
  - Chain reasoning: multi-step cause → effect → cause → effect
  - Effect prediction: given a cause, predict effects
  - Root cause analysis: trace effects back to root causes
  - Causal classification: distinguish correlation from causation
  - Counterfactual reasoning: what would happen if X changed?
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class CausalResult:
    """Structured result from causal reasoning."""
    answer: str
    confidence: float
    method: str
    chain: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Causal chain database (expanded) ──────────────────────────
CAUSAL_CHAINS: list[tuple[str, list[str], float]] = [
    # Earth science
    ("rain", [
        "Water evaporates from oceans and lakes",
        "Water vapor rises and cools in the atmosphere",
        "Water vapor condenses into cloud droplets",
        "Cloud droplets combine and grow heavier",
        "Gravity pulls water droplets to Earth as rain",
    ], 0.95),
    ("earthquake", [
        "Tectonic plates are in constant motion",
        "Plates get stuck at their boundaries",
        "Stress builds up over time",
        "Stress exceeds rock strength",
        "Plates suddenly slip, releasing energy as seismic waves",
    ], 0.95),
    ("volcano", [
        "Magma forms deep in Earth's mantle",
        "Magma rises due to lower density",
        "Pressure builds in underground chambers",
        "Magma reaches the surface through vents",
        "Eruption releases lava, ash, and gases",
    ], 0.95),
    ("season", [
        "Earth is tilted 23.5° on its axis",
        "Different hemispheres receive different sunlight angles",
        "Summer hemisphere gets more direct sunlight",
        "Winter hemisphere gets less direct sunlight",
        "This creates seasonal temperature changes",
    ], 0.95),
    ("tide", [
        "Moon's gravity pulls on Earth's water",
        "Water bulges toward the Moon",
        "A second bulge forms on the opposite side",
        "Earth rotates through these bulges",
        "Coastal areas experience two high tides per day",
    ], 0.95),
    ("sky blue", [
        "Sunlight contains all colors",
        "Sunlight enters Earth's atmosphere",
        "Gas molecules scatter short-wavelength light",
        "Blue light scatters more than other colors",
        "Scattered blue light reaches our eyes from all directions",
    ], 0.95),
    ("rainbow", [
        "Sunlight enters raindrops",
        "Light refracts (bends) entering the water",
        "Different colors bend at different angles",
        "Light reflects off the back of the raindrop",
        "Light refracts again exiting the raindrop, separating colors",
    ], 0.95),
    ("thunder", [
        "Lightning heats the air around it to 30,000°C",
        "Heated air expands rapidly",
        "Rapid expansion creates a shockwave",
        "Shockwave travels as sound waves",
        "We hear thunder as the sound reaches us",
    ], 0.95),
    ("wind", [
        "Sun heats Earth's surface unevenly",
        "Heated air rises, creating low pressure",
        "Cooler air moves in to fill the gap",
        "This movement of air is wind",
        "Differences in pressure determine wind speed and direction",
    ], 0.95),
    ("snow", [
        "Water vapor rises into cold clouds",
        "Temperature is below freezing (0°C)",
        "Water vapor freezes into ice crystals",
        "Ice crystals grow and form snowflakes",
        "Snowflakes fall when they become heavy enough",
    ], 0.95),

    # Biology
    ("photosynthesis", [
        "Chlorophyll absorbs sunlight energy",
        "Light energy splits water molecules",
        "Carbon dioxide is captured from air",
        "Energy converts CO2 and H2O into glucose",
        "Oxygen is released as a byproduct",
    ], 0.95),
    ("evolution", [
        "Genetic mutations create variation in a population",
        "Environmental pressures select for certain traits",
        "Organisms with favorable traits survive and reproduce",
        "Favorable traits are passed to offspring",
        "Over many generations, the population changes",
    ], 0.95),
    ("dna replication", [
        "DNA helix unwinds at replication fork",
        "Helicase separates the two strands",
        "Primase adds RNA primers",
        "DNA polymerase builds new complementary strands",
        "Ligase seals gaps between Okazaki fragments",
    ], 0.95),
    ("disease", [
        "Pathogen enters the body",
        "Pathogen attaches to host cells",
        "Pathogen multiplies inside host cells",
        "Immune system detects the infection",
        "Immune response fights the pathogen",
    ], 0.90),
    ("muscle soreness", [
        "Intense exercise causes micro-tears in muscle fibers",
        "Inflammatory response sends immune cells to repair",
        "Repair process produces lactic acid and other metabolites",
        "Nerve endings detect the chemical changes",
        "Brain registers pain信号信号 as muscle soreness",
    ], 0.85),

    # Technology
    ("internet slow", [
        "High network traffic causes congestion",
        "Routers buffer excess packets",
        "Packet delays increase",
        "Retransmissions add more traffic",
        "Effective throughput decreases",
    ], 0.90),
    ("computer crash", [
        "Software bug causes unexpected behavior",
        "Program accesses invalid memory",
        "Operating system detects the error",
        "OS terminates the offending process",
        "System may display error message or restart",
    ], 0.90),
    ("battery drain", [
        "Apps run background processes",
        "Screen brightness uses power",
        "Wireless radios (WiFi, Bluetooth) consume energy",
        "CPU and GPU process data",
        "Battery charge decreases",
    ], 0.90),

    # Chemistry
    ("rust", [
        "Iron is exposed to water and oxygen",
        "Iron atoms lose electrons (oxidation)",
        "Iron oxide forms on the surface",
        "Iron oxide is porous, allowing more exposure",
        "Process continues, weakening the metal",
    ], 0.95),
    ("fire", [
        "Fuel is present (wood, paper, etc.)",
        "Heat source raises fuel to ignition temperature",
        "Oxygen from air supports combustion",
        "Chemical reaction releases heat and light",
        "Heat sustains the reaction until fuel or oxygen runs out",
    ], 0.95),

    # Everyday
    ("ice slippery", [
        "Pressure on ice melts a thin layer",
        "Thin water layer reduces friction",
        "Water acts as a lubricant between surfaces",
        "This makes ice slippery",
    ], 0.90),
    ("leaf brown", [
        "Leaves contain chlorophyll which makes them green",
        "As days shorten, chlorophyll production slows",
        "Existing chlorophyll breaks down",
        "Other pigments (carotenoids, anthocyanins) become visible",
        "Leaves appear yellow, orange, or red",
    ], 0.90),
    ("onion cry", [
        "Cutting onion releases syn-propanethial-S-oxide",
        "Gas reaches the eyes",
        "Gas reacts with water in tears",
        "Irritation signals are sent to the brain",
        "Brain triggers tear production to wash away the irritant",
    ], 0.90),
    ("caffeine awake", [
        "Caffeine molecules are similar to adenosine",
        "Caffeine binds to adenosine receptors",
        "Adenosine cannot bind and signal sleepiness",
        "Dopamine and norepinephrine increase",
        "Brain remains alert and focused",
    ], 0.90),
    ("exercise muscle", [
        "Exercise creates micro-tears in muscle fibers",
        "Body sends satellite cells to repair damage",
        "Satellite cells fuse with muscle fibers",
        "Protein synthesis builds new muscle tissue",
        "Muscle becomes stronger and larger",
    ], 0.90),
]


class CausalHandler:
    """Handles causal reasoning tasks."""

    def process(self, query: str, evidence: list[str] | None = None) -> CausalResult:
        t0 = time.perf_counter()
        q = query.strip()
        ev = evidence or []

        result = self._try_causal_chain(q, t0)
        if result:
            return result

        result = self._try_effect_prediction(q, t0)
        if result:
            return result

        result = self._try_root_cause(q, t0)
        if result:
            return result

        result = self._try_counterfactual(q, ev, t0)
        if result:
            return result

        result = self._try_causal_from_evidence(q, ev, t0)
        if result:
            return result

        return CausalResult(
            answer="", confidence=0.0, method="none",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ── Causal Chain ─────────────────────────────────────

    def _try_causal_chain(self, q: str, t0: float) -> CausalResult | None:
        """Answer 'why' questions with a causal chain."""
        q_lower = q.lower()

        # Extract the topic from the question
        topic_match = re.search(
            r"why\s+(?:does?|do|is|are|did|has|can|would)\s+(.+?)(?:\?)?$",
            q_lower,
        )
        if not topic_match:
            topic_match = re.search(
                r"what\s+(?:causes?|makes?|leads?\s+to)\s+(.+?)(?:\?)?$",
                q_lower,
            )

        if not topic_match:
            return None

        topic = topic_match.group(1).strip()

        # Search for matching causal chain
        for chain_topic, chain_steps, confidence in CAUSAL_CHAINS:
            if self._topic_matches(topic, chain_topic):
                return CausalResult(
                    answer=chain_steps[-1],  # Final effect
                    confidence=confidence,
                    method="causal_chain",
                    chain=[f"Step {i+1}: {step}" for i, step in enumerate(chain_steps)],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    metadata={"topic": chain_topic, "chain_length": len(chain_steps)},
                )

        return None

    # ── Effect Prediction ────────────────────────────────

    def _try_effect_prediction(self, q: str, t0: float) -> CausalResult | None:
        """Predict effects of a given cause."""
        q_lower = q.lower()

        predict_match = re.search(
            r"(?:what|what)\s+(?:happens?|would\s+happen|are?\s+the\s+effects?)\s+(?:if|when)\s+(.+?)(?:\?)?$",
            q_lower,
        )
        if predict_match:
            scenario = predict_match.group(1).strip()

            # Search for matching cause in causal chains
            for chain_topic, chain_steps, confidence in CAUSAL_CHAINS:
                if self._topic_matches(scenario, chain_topic):
                    # Return effects (steps after the cause)
                    effects = chain_steps[1:] if len(chain_steps) > 1 else chain_steps
                    return CausalResult(
                        answer=effects[0] if effects else "No known effects",
                        confidence=confidence * 0.85,
                        method="effect_prediction",
                        chain=[f"Effect {i+1}: {e}" for i, e in enumerate(effects)],
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )

            # Generic prediction based on keywords
            if "increase" in scenario or "raise" in scenario:
                return CausalResult(
                    answer="Upstream effects propagate through connected systems",
                    confidence=0.50,
                    method="effect_prediction",
                    chain=["Cause propagates through causal links"],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )

        return None

    # ── Root Cause Analysis ──────────────────────────────

    def _try_root_cause(self, q: str, t0: float) -> CausalResult | None:
        """Trace an effect back to its root cause."""
        q_lower = q.lower()

        root_match = re.search(
            r"(?:what\s+is\s+the\s+)?root\s+cause\s+(?:of|for)\s+(.+?)(?:\?)?$",
            q_lower,
        )
        if root_match:
            effect = root_match.group(1).strip()

            for chain_topic, chain_steps, confidence in CAUSAL_CHAINS:
                if self._topic_matches(effect, chain_topic):
                    root = chain_steps[0] if chain_steps else "Unknown"
                    return CausalResult(
                        answer=root,
                        confidence=confidence,
                        method="root_cause",
                        chain=[f"Root → {chain_steps[0]}"],
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )

        return None

    # ── Counterfactual ───────────────────────────────────

    def _try_counterfactual(self, q: str, ev: list[str], t0: float) -> CausalResult | None:
        """Handle 'what if' questions."""
        q_lower = q.lower()

        cf_match = re.search(
            r"what\s+if\s+(.+?)(?:\?)?$",
            q_lower,
        )
        if cf_match:
            scenario = cf_match.group(1).strip()

            # Check if this relates to a known causal chain
            for chain_topic, chain_steps, confidence in CAUSAL_CHAINS:
                if self._topic_matches(scenario, chain_topic):
                    # Analyze the chain and suggest what would change
                    return CausalResult(
                        answer=f"If {scenario}, the causal chain would be affected: {' → '.join(chain_steps[:3])}",
                        confidence=0.65,
                        method="counterfactual",
                        chain=[f"Scenario: {scenario}"] + [f"  {s}" for s in chain_steps],
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )

            # Generic counterfactual response
            return CausalResult(
                answer=f"Analyzing what-if scenario: {scenario}",
                confidence=0.40,
                method="counterfactual",
                chain=[f"Counterfactual: {scenario}"],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        return None

    # ── Causal from Evidence ─────────────────────────────

    def _try_causal_from_evidence(self, q: str, ev: list[str], t0: float) -> CausalResult | None:
        """Extract causal relationships from evidence."""
        q_lower = q.lower()

        if not any(w in q_lower for w in ("cause", "reason", "why", "because", "leads to", "results in")):
            return None

        if not ev:
            return None

        # Look for causal language in evidence
        causal_patterns = [
            r"(.+)\s+causes?\s+(.+)",
            r"(.+)\s+leads?\s+to\s+(.+)",
            r"(.+)\s+results?\s+in\s+(.+)",
            r"(.+)\s+because\s+(.+)",
            r"due\s+to\s+(.+?),\s+(.+)",
            r"(.+)\s+is\s+caused\s+by\s+(.+)",
        ]

        found_chains = []
        for e in ev:
            for pattern in causal_patterns:
                match = re.search(pattern, e.lower())
                if match:
                    cause = match.group(1).strip()
                    effect = match.group(2).strip()
                    found_chains.append((cause, effect, e))

        if found_chains:
            best = found_chains[0]
            return CausalResult(
                answer=f"{best[0]} → {best[1]}",
                confidence=0.75,
                method="evidence_causal",
                chain=[f"Cause: {best[0]}", f"Effect: {best[1]}"],
                latency_ms=(time.perf_counter() - t0) * 1000,
                metadata={"chains_found": len(found_chains)},
            )

        return None

    # ── Helpers ──────────────────────────────────────────

    def _topic_matches(self, query_topic: str, chain_topic: str) -> bool:
        """Check if a query topic matches a causal chain topic."""
        q_words = set(re.findall(r"\b\w{3,}\b", query_topic.lower()))
        c_words = set(re.findall(r"\b\w{3,}\b", chain_topic.lower()))

        # Direct match
        if chain_topic.lower() in query_topic.lower():
            return True

        # Word overlap
        overlap = len(q_words & c_words)
        if overlap >= 2:
            return True

        # Check for key causal words
        causal_words = {"why", "cause", "reason", "because", "due", "leads", "results"}
        query_content = q_words - causal_words
        chain_content = c_words - causal_words

        if query_content and chain_content:
            content_overlap = len(query_content & chain_content)
            if content_overlap >= 1:
                return True

        return False
