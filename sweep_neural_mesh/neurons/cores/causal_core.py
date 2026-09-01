"""
CausalCore — cause-effect reasoning chains.

Responsibilities:
  - Answer "why" / "how" / "what causes" questions.
  - Match queries against a pre-built causal-chain database.
  - Fall back to causal language detection in evidence.
"""
from __future__ import annotations

import re
import time

from ..core_protocol import CoreResult, make_result, empty_result


# ── Causal chain database ─────────────────────────────────────
_RAW_CAUSAL: list[tuple[str, str, float]] = [
    # ═══ WEATHER & EARTH SCIENCE ═══
    (r"why.*rain|cause.*rain", "Water vapor condenses in clouds", 0.95),
    (r"why.*earthquake|cause.*earthquake", "Tectonic plates shift", 0.95),
    (r"why.*volcano|cause.*volcano", "Magma erupts from Earth's interior", 0.95),
    (r"why.*season|cause.*season", "Earth's axial tilt (23.5°)", 0.95),
    (r"why.*tide|cause.*tide", "Moon's gravitational pull on ocean water", 0.95),
    (r"why.*leaf.*brown|leaf.*change.*color|leaves.*brown", "Chlorophyll breaks down revealing other pigments", 0.90),
    (r"why.*sky.*blue|sky.*blue.*cause", "Rayleigh scattering of short-wavelength light", 0.95),
    (r"why.*sun.*hot|sun.*hot.*cause", "Nuclear fusion of hydrogen into helium", 0.95),
    (r"why.*wind|cause.*wind", "Differences in air pressure between regions", 0.95),
    (r"why.*snow|cause.*snow", "Water vapor freezes into ice crystals in clouds", 0.95),
    (r"why.*fog|cause.*fog", "Water vapor condenses near ground level", 0.95),
    (r"why.*thunder|cause.*thunder", "Lightning heats air to 30,000°C, creating shockwave", 0.95),
    (r"why.*rainbow|cause.*rainbow", "Sunlight refracts and reflects inside raindrops", 0.95),
    (r"why.*avalanche|cause.*avalanche", "Snow layers become unstable on steep slopes", 0.90),
    (r"why.*tsunami|cause.*tsunami", "Underwater earthquake displaces large water volume", 0.95),
    (r"why.*drought|cause.*drought", "Prolonged below-normal precipitation", 0.90),
    (r"why.*hurricane|hurricane.*form", "Warm ocean water evaporates, creating rotating storm", 0.95),
    (r"why.*tornado|tornado.*form", "Warm moist air meets cold dry air, creating rotation", 0.95),
    (r"why.*erosion|cause.*erosion", "Wind, water, or ice wears away rock and soil", 0.90),
    (r"why.*cave.*form|cave.*form", "Water dissolves soluble rock like limestone", 0.90),
    (r"why.*coral.*bleach", "Rising water temperature causes coral to expel symbiotic algae", 0.95),
    (r"why.*ocean.*acidif", "CO2 dissolves in seawater forming carbonic acid", 0.95),
    (r"why.*glacier.*retreat", "Warmer temperatures cause ice to melt faster than accumulation", 0.90),
    # ═══ BIOLOGY ═══
    (r"how.*dna.*work|dna.*work", "DNA stores genetic instructions in base pairs", 0.95),
    (r"how.*photosynthesis.*work", "Plants convert light, CO2, and water into glucose and oxygen", 0.95),
    (r"how.*gravity.*work", "Mass curves spacetime (Einstein's general relativity)", 0.95),
    (r"why.*heart.*beat|cause.*heartbeat", "Electrical signals from sinoatrial node", 0.95),
    (r"why.*yawn|cause.*yawn", "Brain cooling or social empathy mechanism", 0.85),
    (r"why.*sleep|cause.*sleep", "Brain restoration, memory consolidation, and energy conservation", 0.90),
    (r"why.*dream|cause.*dreams", "Brain processes emotions and consolidates memories during REM sleep", 0.85),
    (r"why.*age|cause.*aging", "Telomere shortening, DNA damage accumulation, and cellular senescence", 0.90),
    (r"why.*muscle.*sore|sore.*muscles", "Micro-tears in muscle fibers trigger inflammation and repair", 0.95),
    (r"why.*hair.*grow|hair.*growth", "Follicles produce keratin protein in growth cycles", 0.90),
    (r"why.*shiver|shivering.*cause", "Muscles contract to generate heat when body is cold", 0.95),
    (r"why.*blush|blushing.*cause", "Blood vessels dilate due to emotional response", 0.90),
    (r"why.*sneeze|sneezing.*cause", "Nasal irritants trigger reflex to expel particles", 0.95),
    (r"why.*hiccup|hiccup.*cause", "Diaphragm spasms due to irritation of phrenic nerve", 0.90),
    (r"why.*callus.*form", "Repeated friction causes skin to thicken for protection", 0.90),
    (r"why.*fever|fever.*cause", "Immune system raises body temperature to fight infection", 0.95),
    (r"why.*bruise|bruise.*cause", "Blood vessels break under skin, blood pools and discolors", 0.95),
    (r"why.*scar.*form", "Body produces collagen to repair wound, differs from normal skin", 0.90),
    (r"why.*photon.*energy|photon.*energy", "E = hf, energy proportional to frequency", 0.95),
    (r"how.*nerve.*impulse|nerve.*signal", "Sodium/potassium ion exchange propagates electrical signal", 0.95),
    (r"why.*cramp|muscle.*cramp", "Involuntary muscle contraction due to dehydration or fatigue", 0.90),
    (r"why.*eye.*twitch|eye.*twitching", "Fatigue, stress, or caffeine causing eyelid muscle spasm", 0.85),
    # ═══ CHEMISTRY ═══
    (r"why.*rust|cause.*rust", "Iron oxidizes with water and oxygen (electrochemical process)", 0.95),
    (r"why.*metal.*expand|metal.*expand.*heat", "Atoms vibrate faster and occupy more space when heated", 0.95),
    (r"why.*sugar.*dissolve", "Polar water molecules separate sugar molecules (like dissolves like)", 0.95),
    (r"why.*ice.*float", "Water expands when freezing, ice is less dense than liquid water", 0.95),
    (r"why.*fire.*hot", "Exothermic chemical reaction releases energy as heat and light", 0.95),
    (r"why.*glass.*transparent", "Electrons in glass don't absorb visible light photons", 0.90),
    (r"why.*diamond.*hard", "Strong covalent bonds in rigid tetrahedral crystal structure", 0.95),
    (r"why.*lead.*heavy", "Lead atoms are large with many protons and neutrons", 0.90),
    (r"why.*copper.*green.*patina", "Copper oxidizes and reacts with atmospheric compounds", 0.90),
    # ═══ TECHNOLOGY ═══
    (r"how.*internet.*work", "Data packets routed through TCP/IP across interconnected networks", 0.95),
    (r"how.*wifi.*work", "Radio waves (2.4/5 GHz) transmit data between devices and router", 0.95),
    (r"how.*computer.*work", "CPU executes binary instructions, manipulating data in memory", 0.95),
    (r"how.*bluetooth.*work", "Short-range radio waves (2.4 GHz) pair devices wirelessly", 0.95),
    (r"how.*gps.*work", "Trilateration from 4+ satellites measuring signal travel time", 0.95),
    (r"how.*touchscreen.*work", "Capacitive screens detect electrical changes from finger contact", 0.95),
    (r"how.*battery.*work", "Chemical energy converts to electrical energy via redox reactions", 0.95),
    (r"how.*led.*work", "Electrons release photons when crossing semiconductor junction", 0.95),
    (r"how.*hard.*drive.*work", "Magnetic platters store data as binary on spinning disks", 0.95),
    (r"how.*ssd.*work", "Flash memory stores data in floating-gate transistors", 0.95),
    (r"how.*3d.*printer.*work", "Adds material layer by layer from digital model", 0.95),
    (r"how.*laser.*work", "Stimulated emission of coherent light photons", 0.95),
    (r"how.*email.*work", "SMTP sends, IMAP/POP3 receives, servers route messages", 0.95),
    (r"how.*search.*engine.*work", "Crawls web, indexes content, ranks by relevance algorithms", 0.95),
    (r"how.*streaming.*work", "Video compressed, sent in packets, buffered and decoded on device", 0.90),
    (r"how.*ai.*neural.*network.*work", "Layers of artificial neurons adjust weights to minimize prediction error", 0.95),
    # ═══ EVERYDAY ═══
    (r"why.*ice.*slippery", "Pressure on ice melts thin water layer acting as lubricant", 0.90),
    (r"why.*onion.*make.*cry", "Cutting releases syn-propanethial-S-oxide gas that irritates eyes", 0.95),
    (r"why.*coffee.*keep.*awake", "Caffeine blocks adenosine receptors that signal sleepiness", 0.95),
    (r"why.*exercise.*sweat", "Evaporation of sweat cools the body (thermoregulation)", 0.95),
    (r"why.*microwave.*heat.*food", "Microwaves excite water molecules, generating heat via friction", 0.95),
    (r"why.*banana.*slippery", "Smooth waxy surface and moisture reduce friction", 0.90),
    (r"why.*echo.*happen", "Sound waves reflect off hard surfaces and return to listener", 0.95),
    (r"why.*soap.*clean", "Amphiphilic molecules bind to both water and grease", 0.95),
    (r"why.*balloon.*float.*helium", "Helium is lighter than air, creating buoyancy", 0.95),
    (r"why.*paint.*dry", "Solvents evaporate, leaving solid pigment and binder layer", 0.90),
    (r"why.*magnet.*stick.*fridge", "Ferromagnetic material attracted to magnetic field", 0.95),
    (r"why.*popcorn.*pop", "Water inside kernel heats, pressure builds, kernel explodes", 0.95),
    (r"why.*paper.*tear.*easily", "Weak hydrogen bonds between cellulose fibers break under stress", 0.90),
    (r"why.*ketchup.*hard.*pour", "Shear-thinning non-Newtonian fluid needs force to flow", 0.95),
    (r"why.*rain.*smell.*good", "Geosmin released by soil bacteria when rain hits dry ground", 0.90),
    (r"why.*stomach.*growl", "Gas and fluid move through contracting intestines", 0.90),
    (r"why.*static.*electricity", "Electrons transfer between materials through friction", 0.95),
    (r"why.*sky.*red.*sunset", "Longer light path scatters blue light, leaving red/orange", 0.95),
]


class CausalCore:
    """Core E — Causal reasoning and cause-effect analysis.

    Matches "why"/"how" queries against pre-built causal chains.
    Falls back to detecting causal language in evidence.
    """

    CORE_ID = "causal"

    def __init__(self) -> None:
        self._compiled: list[tuple[re.Pattern[str], str, float]] = []
        for pattern, answer, confidence in _RAW_CAUSAL:
            try:
                self._compiled.append(
                    (re.compile(pattern, re.IGNORECASE), answer, confidence)
                )
            except re.error:
                pass

    @property
    def core_id(self) -> str:
        return self.CORE_ID

    def process(self, query: str, evidence: list[str]) -> CoreResult:
        t0 = time.perf_counter()
        q = query.lower()

        # Check for why/how questions
        if q.startswith(("why ", "how ", "what causes ", "what makes ")):
            for compiled, answer, confidence in self._compiled:
                if compiled.search(q):
                    return make_result(
                        self.CORE_ID, answer, confidence,
                        f"Causal chain: {compiled.pattern}", t0,
                    )

        # Fall back to causal language in evidence
        if evidence:
            for ev in evidence:
                if re.search(r"(because|due to|caused by|leads to|results in)", ev.lower()):
                    return make_result(
                        self.CORE_ID, ev[:200], 0.7,
                        "Causal evidence found", t0, evidence_used=1,
                    )

        return empty_result(self.CORE_ID, t0, "No causal pattern matched")
