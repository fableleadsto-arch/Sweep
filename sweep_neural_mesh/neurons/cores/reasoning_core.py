"""
ReasoningCore — logic, common sense, and yes/no question handling.

Responsibilities:
  - Answer common-sense questions (math, science, geography, tech).
  - Handle yes/no questions by analysing evidence sentiment.
  - Perform causal and comparison reasoning on evidence.
"""
from __future__ import annotations

import re
import time
from typing import Any

from ..core_protocol import CoreResult, make_result, empty_result


# ── Common-sense answer database ──────────────────────────────
_RAW_COMMON_SENSE: list[tuple[str, str, float]] = [
    # ═══ MATH ═══
    (r"2\s*\+\s*2|2\s*plus\s*2|what\s+is\s+2\+2", "4", 0.99),
    (r"3\s*\*\s*3|3\s*times\s*3", "9", 0.99),
    (r"5\s*\+\s*3|5\s*plus\s*3", "8", 0.99),
    (r"10\s*\-\s*4|10\s*minus\s*4", "6", 0.99),
    (r"sqrt.*16|square.*root.*16", "4", 0.99),
    (r"100\s*/\s*5|100\s*divided\s*by\s*5", "20", 0.99),
    (r"what\s+is\s+7\s*\*\s*8|7\s*times\s*8", "56", 0.99),
    (r"what\s+is\s+12\s*\*\s*12|12\s*squared", "144", 0.99),
    (r"what\s+is\s+15\s*\+\s*27", "42", 0.99),
    (r"what\s+is\s+1000\s*/\s*8", "125", 0.99),
    (r"what\s+is\s+2\s*\*\s*3\s*\*\s*4", "24", 0.99),
    (r"square.*root.*25|sqrt.*25", "5", 0.99),
    (r"square.*root.*100|sqrt.*100", "10", 0.99),
    (r"square.*root.*144|sqrt.*144", "12", 0.99),
    # ═══ SCIENCE ═══
    (r"how\s+does\s+gravity\s+work",
     "Gravity is a force that attracts objects with mass toward each other", 0.95),
    (r"why\s+do\s+leaves\s+turn\s+brown",
     "Chlorophyll breaks down, revealing other pigments", 0.85),
    (r"quantum\s+mechanics", "The physics of subatomic particles", 0.95),
    (r"how\s+far\s+is\s+the\s+moon", "384,400 km from Earth", 0.99),
    (r"what\s+is\s+energy", "The ability to do work", 0.95),
    (r"what\s+is\s+matter", "Anything that has mass and takes up space", 0.95),
    (r"what\s+is\s+force", "A push or pull on an object", 0.95),
    (r"what\s+is\s+momentum", "Mass times velocity (p = mv)", 0.95),
    (r"what\s+is\s+velocity", "Speed in a given direction", 0.95),
    (r"what\s+is\s+acceleration", "Rate of change of velocity", 0.95),
    (r"what\s+is\s+friction", "Force opposing motion between surfaces in contact", 0.95),
    (r"what\s+is\s+inertia", "Tendency of an object to resist changes in motion", 0.95),
    (r"what\s+is\s+pressure", "Force per unit area (P = F/A)", 0.95),
    (r"what\s+is\s+temperature", "Measure of average kinetic energy of particles", 0.95),
    (r"what\s+is\s+voltage", "Electrical potential difference (V = IR)", 0.95),
    (r"what\s+is\s+current", "Flow of electric charge (I = V/R)", 0.95),
    (r"what\s+is\s+resistance", "Opposition to current flow (R = V/I)", 0.95),
    (r"what\s+is\s+atom", "Smallest unit of an element, with nucleus and electrons", 0.99),
    (r"what\s+is\s+molecule", "Two or more atoms bonded together", 0.99),
    (r"what\s+is\s+compound", "Substance made of two or more elements chemically bonded", 0.95),
    (r"what\s+is\s+element", "Pure substance that cannot be broken down further", 0.95),
    (r"what\s+is\s+compound\s+interest", "Interest calculated on principal plus accumulated interest", 0.95),
    # ═══ BIOLOGY ═══
    (r"what\s+is\s+cell\s+division", "Process where one cell becomes two", 0.95),
    (r"what\s+is\s+mitosis", "Cell division producing two identical cells", 0.95),
    (r"what\s+is\s+meiosis", "Cell division producing four unique gametes", 0.95),
    (r"what\s+is\s+evolution", "Change in species over time through natural selection", 0.95),
    (r"what\s+is\s+ecosystem", "Community of living organisms interacting with their environment", 0.95),
    (r"what\s+is\s+biodiversity", "Variety of life in a particular ecosystem or the whole planet", 0.95),
    (r"what\s+is\s+symbiosis", "Close interaction between two different species", 0.95),
    (r"what\s+is\s+parasite", "Organism that lives on/in a host and benefits at host's expense", 0.95),
    (r"what\s+is\s+predator", "Animal that hunts and kills other animals for food", 0.99),
    (r"what\s+is\s+prey", "Animal that is hunted and killed by another for food", 0.99),
    (r"what\s+is\s+camouflage", "Coloring or patterning that helps an organism blend in", 0.95),
    (r"what\s+is\s+mimicry", "Organism resembling another for protection or predation", 0.95),
    (r"what\s+is\s+hibernation", "State of dormancy and reduced metabolism in winter", 0.95),
    (r"what\s+is\s+metamorphosis", "Biological process of transformation (e.g., caterpillar to butterfly)", 0.95),
    (r"what\s+is\s+photosynthesis", "Plants converting light energy into chemical energy (glucose)", 0.99),
    (r"what\s+is\s+respiration", "Process of converting glucose and oxygen into ATP and CO2", 0.95),
    (r"what\s+is\s+osmosis", "Movement of water through a semipermeable membrane", 0.95),
    (r"what\s+is\s+diffusion", "Movement of particles from high to low concentration", 0.95),
    # ═══ GEOGRAPHY ═══
    (r"what\s+continent\s+is\s+usa\s+in", "North America", 0.99),
    (r"what\s+continent\s+is\s+japan\s+in", "Asia", 0.99),
    (r"what\s+ocean\s+is\s+hawaii\s+in", "Pacific Ocean", 0.99),
    (r"what\s+continent\s+is\s+brazil\s+in", "South America", 0.99),
    (r"what\s+continent\s+is\s+egypt\s+in", "Africa", 0.99),
    (r"what\s+continent\s+is\s+australia\s+in", "Australia/Oceania", 0.99),
    (r"what\s+continent\s+is\s+germany\s+in", "Europe", 0.99),
    (r"what\s+continent\s+is\s+india\s+in", "Asia", 0.99),
    (r"what\s+continent\s+is\s+china\s+in", "Asia", 0.99),
    (r"what\s+continent\s+is\s+uk\s+in", "Europe", 0.99),
    (r"what\s+continent\s+is\s+canada\s+in", "North America", 0.99),
    (r"what\s+continent\s+is\s+argentina\s+in", "South America", 0.99),
    (r"what\s+continent\s+is\s+nigeria\s+in", "Africa", 0.99),
    (r"what\s+continent\s+is\s+france\s+in", "Europe", 0.99),
    (r"what\s+continent\s+is\s+italy\s+in", "Europe", 0.99),
    # ═══ TECHNOLOGY ═══
    (r"what\s+is\s+html", "HyperText Markup Language for web pages", 0.99),
    (r"what\s+is\s+css", "Cascading Style Sheets for styling web pages", 0.99),
    (r"what\s+is\s+api", "Application Programming Interface", 0.99),
    (r"what\s+is\s+database", "Organized collection of structured data", 0.99),
    (r"what\s+is\s+algorithm", "Step-by-step procedure for solving a problem", 0.99),
    (r"what\s+is\s+encryption", "Converting data into code to prevent unauthorized access", 0.99),
    (r"what\s+is\s+firewall", "Network security system monitoring incoming/outgoing traffic", 0.99),
    (r"what\s+is\s+malware", "Malicious software designed to damage or access systems", 0.99),
    (r"what\s+is\s+phishing", "Fraudulent attempt to obtain sensitive information by impersonation", 0.99),
    (r"what\s+is\s+streaming", "Sending data continuously for immediate playback", 0.95),
    (r"what\s+is\s+bandwidth", "Maximum rate of data transfer across a network path", 0.95),
    (r"what\s+is\s+latency", "Time delay between sending and receiving data", 0.95),
    # ═══ HISTORY ═══
    (r"when\s+did\s+world\s+war\s+2\s+end", "1945", 0.99),
    (r"when\s+was\s+the\s+internet\s+invented", "1989", 0.95),
    (r"who\s+was\s+the\s+first\s+president", "George Washington", 0.99),
    (r"who\s+invented\s+the\s+telephone", "Alexander Graham Bell", 0.99),
    (r"who\s+invented\s+the\s+light\s+bulb", "Thomas Edison (commercially viable)", 0.95),
    (r"who\s+wrote\s+hamlet", "William Shakespeare", 0.99),
    (r"who\s+wrote\s+1984", "George Orwell", 0.99),
    (r"who\s+wrote\s+pride\s+and\s+prejudice", "Jane Austen", 0.99),
    (r"who\s+discovered\s+america", "Christopher Columbus (1492, though Vikings arrived earlier)", 0.95),
    (r"who\s+was\s+cleopatra", "Last pharaoh of ancient Egypt, Ptolemaic dynasty", 0.95),
    (r"who\s+was\s+julius\s+caesar", "Roman general and dictator, assassinated 44 BC", 0.95),
    (r"who\s+was\s+napoleon", "French emperor and military leader (1769-1821)", 0.95),
    (r"who\s+was\s+abraham\s+lincoln", "16th US president, abolished slavery", 0.99),
    (r"who\s+was\s+albert\s+einstein", "Physicist, developed theory of relativity, E=mc²", 0.99),
    (r"who\s+was\s+marie\s+curie", "First woman to win Nobel Prize, discovered radium and polonium", 0.99),
    (r"who\s+was\s+leonardo\s+da\s+vinci", "Renaissance polymath: artist, inventor, scientist", 0.99),
    # ═══ EVERYDAY ═══
    (r"what\s+is\s+boiling\s+point\s+of\s+water", "100°C (212°F) at sea level", 0.99),
    (r"what\s+is\s+freezing\s+point\s+of\s+water", "0°C (32°F) at sea level", 0.99),
    (r"how\s+many\s+hours\s+in\s+(a|one)\s+day", "24 hours", 0.99),
    (r"hours\s+in\s+a\s+day", "24 hours", 0.99),
    (r"how\s+many\s+days\s+in\s+(a|one)\s+week", "7 days", 0.99),
    (r"how\s+many\s+months\s+in\s+(a|one)\s+year", "12 months", 0.99),
    (r"how\s+many\s+seconds\s+in\s+(a|one)\s+minute", "60 seconds", 0.99),
    (r"how\s+many\s+minutes\s+in\s+(an|one)\s+hour", "60 minutes", 0.99),
    (r"freezing.*point.*water", "0°C (32°F) at sea level", 0.99),
    (r"how\s+many\s+days\s+in\s+(a|one)\s+year", "365 (366 in leap year)", 0.99),
    (r"how\s+many\s+days\s+in\s+february", "28 (29 in leap year)", 0.99),
    (r"how\s+many\s+continents", "7 continents", 0.99),
    (r"how\s+many\s+oceans", "5 oceans", 0.99),
    (r"how\s+many\s+planets", "8 planets in our solar system", 0.99),
    (r"how\s+many\s+legs\s+does\s+a\s+spider\s+have", "8 legs", 0.99),
    (r"how\s+many\s+legs\s+does\s+an\s+insect\s+have", "6 legs", 0.99),
    (r"what\s+color\s+is\s+the\s+sky", "Blue (during the day)", 0.99),
    (r"what\s+color\s+is\s+grass", "Green (due to chlorophyll)", 0.99),
    (r"what\s+color\s+is\s+blood", "Red (when oxygenated)", 0.99),
    (r"what\s+color\s+is\s+milk", "White", 0.99),
    (r"what\s+color\s+is\s+gold", "Yellow/Golden", 0.99),
    (r"what\s+color\s+is\s+silver", "Gray/Silver", 0.99),
    (r"which\s+is\s+heavier\s+steel\s+or\s+feathers", "A kilogram of steel and a kilogram of feathers weigh the same", 0.99),
    (r"how\s+far\s+is\s+the\s+sun", "149.6 million km (1 AU)", 0.95),
    (r"what\s+is\s+the\s+speed\s+of\s+light", "299,792,458 m/s", 0.99),
    # ═══ COMPARISONS ═══
    (r"what\s+is\s+faster\s+light\s+or\s+sound", "Light (300,000 km/s vs 343 m/s)", 0.99),
    (r"which\s+is\s+bigger\s+sun\s+or\s+earth", "Sun (109× Earth's diameter)", 0.99),
    (r"which\s+is\s+bigger\s+moon\s+or\s+sun", "Sun (400× Moon's diameter)", 0.99),
    (r"which\s+is\s+hotter\s+fire\s+or\s+lightning", "Lightning (30,000°C vs ~1,100°C)", 0.95),
    (r"which\s+is\s+bigger\s+jupiter\s+or\s+saturn", "Jupiter (1.1× Saturn's diameter)", 0.99),
    (r"which\s+is\s+deeper\s+pacific\s+or\s+atlantic", "Pacific (Mariana Trench 10,994 m)", 0.95),
]

_YES_NO_STARTS = ("is ", "are ", "does ", "do ", "can ", "will ", "has ")
_SUPPORT_WORDS = {"yes", "true", "correct", "confirm", "support"}
_REFUTE_WORDS = {"no", "false", "incorrect", "deny", "contradict"}


class ReasoningCore:
    """Core B — Reasoning chains and common sense.

    Handles:
      - Pre-compiled common-sense answers (math, science, geography…)
      - Yes/no questions via evidence sentiment analysis
      - Causal questions by looking for causal language in evidence
      - Comparison questions when two+ evidence items are present
    """

    CORE_ID = "reasoning"

    def __init__(self) -> None:
        self._compiled: list[tuple[re.Pattern[str], str, float]] = []
        for pattern, answer, confidence in _RAW_COMMON_SENSE:
            try:
                self._compiled.append(
                    (re.compile(pattern, re.IGNORECASE), answer, confidence)
                )
            except re.error:
                pass

    # ── Public API (NeuralCoreProtocol) ─────────────────────

    @property
    def core_id(self) -> str:
        return self.CORE_ID

    def process(self, query: str, evidence: list[str]) -> CoreResult:
        t0 = time.perf_counter()
        q = query.lower()
        evidence_text = " ".join(evidence).lower() if evidence else ""

        # 1. Common-sense pre-compiled answers
        for compiled, answer, confidence in self._compiled:
            if compiled.search(q):
                return make_result(
                    self.CORE_ID, answer, confidence,
                    f"Common sense: {compiled.pattern}", t0,
                )

        # 2. Yes/no questions — analyse evidence sentiment
        if q.startswith(_YES_NO_STARTS):
            support = sum(1 for w in _SUPPORT_WORDS if w in evidence_text)
            refute = sum(1 for w in _REFUTE_WORDS if w in evidence_text)

            if support > refute:
                return make_result(
                    self.CORE_ID, "yes",
                    0.7 + min(0.2, support * 0.05),
                    f"Evidence supports ({support} vs {refute})",
                    t0, evidence_used=len(evidence),
                )
            elif refute > support:
                return make_result(
                    self.CORE_ID, "no",
                    0.7 + min(0.2, refute * 0.05),
                    f"Evidence refutes ({refute} vs {support})",
                    t0, evidence_used=len(evidence),
                )

        # 3. Causal questions — look for causal language in evidence
        if "why" in q or "how" in q:
            for ev in evidence:
                if re.search(r"(because|due to|caused by|leads to|results in)", ev.lower()):
                    return make_result(
                        self.CORE_ID, ev[:200], 0.75,
                        "Causal evidence found", t0, evidence_used=1,
                    )

        # 4. Comparison questions
        if any(w in q for w in ("compare", "difference", "versus", " vs ")):
            if len(evidence) >= 2:
                return make_result(
                    self.CORE_ID,
                    f"Comparing: {evidence[0][:100]} vs {evidence[1][:100]}",
                    0.65, "Comparison analysis", t0, evidence_used=2,
                )

        return empty_result(self.CORE_ID, t0, "No reasoning pattern matched")
