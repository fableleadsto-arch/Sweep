"""
Hybrid Inference Engine — Combines all available models for maximum accuracy.

Architecture:
    Query -> Intent Detection -> Model Selection -> Inference -> Verification -> Output

Components:
    1. MiniLM Embeddings (pretrained) — semantic similarity
    2. Trained Relay Transformer (200 steps) — token prediction
    3. Rule-based system — factual lookup, logic, evidence scoring
    4. Ensemble voting — combine multiple model outputs
"""
import sys
import os
import re
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

_sweep_dir = Path(__file__).resolve().parent.parent
_sweep_parent = _sweep_dir.parent
sys.path.insert(0, str(_sweep_parent))
sys.path.insert(0, str(_sweep_dir))

logger = logging.getLogger("hybrid")


@dataclass
class HybridResult:
    """Result from the hybrid engine."""
    answer: str
    confidence: float
    method: str
    components_used: list[str]
    latency_ms: float
    evidence_score: float = 0.0
    reasoning_chain: list[str] = field(default_factory=list)


class HybridEngine:
    """
    Combines multiple models for better inference.

    1. MiniLM for semantic understanding
    2. Relay Transformer for token prediction
    3. Rules for factual answers
    4. Ensemble for final decision
    """

    def __init__(self):
        self._miniLM = None
        self._relay_model = None
        self._relay_tokenizer = None
        self._cortex = None
        self._initialized = False

    def initialize(self):
        """Lazy-load all models."""
        if self._initialized:
            return

        # Load MiniLM
        try:
            from neurons.semantic_embeddings import SemanticEmbedder
            self._miniLM = SemanticEmbedder()
            logger.info(f"MiniLM loaded: {self._miniLM.backend}")
        except Exception as e:
            logger.warning(f"MiniLM failed: {e}")

        # Load Relay Transformer
        try:
            import torch
            from companion.neural.training.checkpointing import load_model, load_tokenizer
            relay_dir = str(_sweep_dir / "training" / "relay_small_trained")
            if os.path.exists(os.path.join(relay_dir, "model.safetensors")):
                self._relay_model = load_model(relay_dir)
                self._relay_tokenizer = load_tokenizer(relay_dir)
                self._relay_model.eval()
                logger.info(f"Relay loaded: {self._relay_model.param_count():,} params")
        except Exception as e:
            logger.warning(f"Relay failed: {e}")

        # Load Cortex (rule-based)
        try:
            from neurons.cortex import ReasoningCortex
            self._cortex = ReasoningCortex(enable_ml=False)
            logger.info("Cortex loaded")
        except Exception as e:
            logger.warning(f"Cortex failed: {e}")

        self._initialized = True

    def answer(self, query: str, evidence: list[str] | None = None) -> HybridResult:
        """Get answer using hybrid approach."""
        self.initialize()
        t0 = time.perf_counter()
        evidence = evidence or []

        components_used = []
        answers = []

        # Rule-based factual lookup (highest priority for factual queries)
        rule_answer = self._rule_based_answer(query)
        if rule_answer:
            answers.append(("rules", rule_answer["answer"], rule_answer["confidence"]))
            components_used.append("rules")

        # MiniLM semantic search
        if self._miniLM and evidence:
            sem_answer = self._semantic_evidence_score(query, evidence)
            if sem_answer:
                answers.append(("minilm", sem_answer["answer"], sem_answer["confidence"]))
                components_used.append("minilm")

        # Relay Transformer token prediction
        if self._relay_model and self._relay_tokenizer:
            relay_answer = self._relay_predict(query)
            if relay_answer:
                answers.append(("relay", relay_answer["answer"], relay_answer["confidence"]))
                components_used.append("relay")

        # Cortex rule-based reasoning
        if self._cortex:
            cortex_result = self._cortex.reason(query=query, evidence=evidence)
            cortex_answer = cortex_result.decision
            cortex_conf = cortex_result.confidence
            answers.append(("cortex", cortex_answer, cortex_conf))
            components_used.append("cortex")

        # Ensemble: Combine answers
        if answers:
            weights = {"rules": 1.5, "minilm": 1.2, "relay": 0.8, "cortex": 1.0}
            best = max(answers, key=lambda x: x[2] * weights.get(x[0], 1.0))
            final_answer = best[1]
            final_confidence = best[2]
            final_method = best[0]
        else:
            final_answer = "unknown"
            final_confidence = 0.0
            final_method = "none"

        latency = (time.perf_counter() - t0) * 1000

        return HybridResult(
            answer=final_answer,
            confidence=final_confidence,
            method=final_method,
            components_used=components_used,
            latency_ms=latency,
        )

    def _rule_based_answer(self, query: str) -> dict | None:
        """Fast rule-based factual lookup."""
        q = query.lower().strip()

        # Capital cities (80+ countries)
        capitals = {
            "france": "Paris", "japan": "Tokyo", "germany": "Berlin", "uk": "London",
            "china": "Beijing", "india": "New Delhi", "brazil": "Brasilia",
            "australia": "Canberra", "canada": "Ottawa", "egypt": "Cairo",
            "russia": "Moscow", "south korea": "Seoul", "italy": "Rome",
            "spain": "Madrid", "mexico": "Mexico City", "turkey": "Ankara",
            "thailand": "Bangkok", "vietnam": "Hanoi", "indonesia": "Jakarta",
            "pakistan": "Islamabad", "nigeria": "Abuja", "kenya": "Nairobi",
            "argentina": "Buenos Aires", "colombia": "Bogota", "chile": "Santiago",
            "peru": "Lima", "greece": "Athens", "portugal": "Lisbon",
            "netherlands": "Amsterdam", "belgium": "Brussels", "switzerland": "Bern",
            "austria": "Vienna", "sweden": "Stockholm", "norway": "Oslo",
            "denmark": "Copenhagen", "finland": "Helsinki", "poland": "Warsaw",
            "czech republic": "Prague", "hungary": "Budapest", "romania": "Bucharest",
            "ukraine": "Kyiv", "ireland": "Dublin", "iceland": "Reykjavik",
            "new zealand": "Wellington", "singapore": "Singapore",
            "philippines": "Manila", "malaysia": "Kuala Lumpur",
            "bangladesh": "Dhaka", "sri lanka": "Colombo", "nepal": "Kathmandu",
            "morocco": "Rabat", "ethiopia": "Addis Ababa", "tanzania": "Dodoma",
            "ghana": "Accra", "cameroon": "Yaounde", "senegal": "Dakar",
            "mali": "Bamako", "niger": "Niamey", "chad": "N'Djamena",
            "somalia": "Mogadishu", "sudan": "Khartoum", "libya": "Tripoli",
            "tunisia": "Tunis", "algeria": "Algiers", "iraq": "Baghdad",
            "iran": "Tehran", "saudi arabia": "Riyadh", "uae": "Abu Dhabi",
            "qatar": "Doha", "kuwait": "Kuwait City", "oman": "Muscat",
            "jordan": "Amman", "lebanon": "Beirut", "israel": "Jerusalem",
            "syria": "Damascus", "yemen": "Sanaa", "afghanistan": "Kabul",
            "myanmar": "Naypyidaw", "laos": "Vientiane", "cambodia": "Phnom Penh",
            "mongolia": "Ulaanbaatar", "taiwan": "Taipei",
            "cuba": "Havana", "jamaica": "Kingston", "costa rica": "San Jose",
            "panama": "Panama City", "ecuador": "Quito", "bolivia": "Sucre",
            "paraguay": "Asuncion", "uruguay": "Montevideo", "venezuela": "Caracas",
            "haiti": "Port-au-Prince", "dominican republic": "Santo Domingo",
            "guatemala": "Guatemala City", "honduras": "Tegucigalpa",
            "el salvador": "San Salvador", "nicaragua": "Managua",
        }
        for country, capital in capitals.items():
            if f"capital of {country}" in q:
                return {"answer": capital, "confidence": 0.99}

        # Planets
        planet_facts = {
            "closest planet to the sun": ("Mercury", 0.99),
            "hottest planet": ("Venus", 0.95),
            "red planet": ("Mars", 0.99),
            "largest planet": ("Jupiter", 0.99),
            "planet with rings": ("Saturn", 0.99),
            "farthest planet": ("Neptune", 0.99),
            "smallest planet": ("Mercury", 0.99),
            "has the great red spot": ("Jupiter", 0.99),
            "rotates backwards": ("Venus", 0.99),
            "has the most moons": ("Saturn", 0.95),
            "tilted on its side": ("Uranus", 0.99),
            "ice giant": ("Uranium", 0.80),
            "planet number 1": ("Mercury", 0.99),
            "planet number 2": ("Venus", 0.99),
            "planet number 3": ("Earth", 0.99),
            "planet number 4": ("Mars", 0.99),
            "planet number 5": ("Jupiter", 0.99),
            "planet number 6": ("Saturn", 0.99),
            "planet number 7": ("Uranus", 0.99),
            "planet number 8": ("Neptune", 0.99),
        }
        for pattern, (answer, conf) in planet_facts.items():
            if pattern in q:
                return {"answer": answer, "confidence": conf}

        # Science facts (expanded)
        science = {
            "boiling point of water": ("100 degrees Celsius", 0.99),
            "freezing point of water": ("0 degrees Celsius", 0.99),
            "speed of light": ("299792458 meters per second", 0.99),
            "chemical formula for water": ("H2O", 0.99),
            "chemical symbol for gold": ("Au", 0.99),
            "chemical symbol for silver": ("Ag", 0.99),
            "chemical symbol for iron": ("Fe", 0.99),
            "chemical symbol for copper": ("Cu", 0.99),
            "chemical symbol for oxygen": ("O", 0.99),
            "chemical symbol for hydrogen": ("H", 0.99),
            "chemical symbol for carbon": ("C", 0.99),
            "chemical symbol for nitrogen": ("N", 0.99),
            "chemical symbol for sodium": ("Na", 0.99),
            "chemical symbol for potassium": ("K", 0.99),
            "chemical symbol for calcium": ("Ca", 0.99),
            "number of bones in human body": ("206", 0.99),
            "how many bones": ("206", 0.99),
            "largest ocean": ("Pacific Ocean", 0.99),
            "tallest mountain": ("Mount Everest", 0.99),
            "longest river": ("Nile River", 0.95),
            "largest continent": ("Asia", 0.99),
            "number of planets": ("8", 0.99),
            "how many planets": ("8", 0.99),
            "how many continents": ("7", 0.99),
            "world war ii ended": ("1945", 0.99),
            "world war 2 ended": ("1945", 0.99),
            "wwii ended": ("1945", 0.99),
            "moon landing": ("1969", 0.99),
            "first moon landing": ("1969", 0.99),
            "dna stands for": ("deoxyribonucleic acid", 0.99),
            "what does dna stand for": ("deoxyribonucleic acid", 0.99),
            "photosynthesis": ("converts sunlight into energy", 0.99),
            "largest desert": ("Sahara Desert", 0.99),
            "deepest ocean point": ("Mariana Trench", 0.99),
            "largest rainforest": ("Amazon Rainforest", 0.99),
            "speed of sound": ("343 meters per second", 0.99),
            "absolute zero": ("-273.15 degrees Celsius", 0.99),
            "atomic number of hydrogen": ("1", 0.99),
            "atomic number of carbon": ("6", 0.99),
            "atomic number of oxygen": ("8", 0.99),
            "atomic number of gold": ("79", 0.99),
            "atomic number of silver": ("47", 0.99),
            "atomic number of iron": ("26", 0.99),
            "atomic number of helium": ("2", 0.99),
            "atomic number of uranium": ("92", 0.99),
            "gravity on the moon": ("1/6 of Earth", 0.99),
            "composition of atmosphere": ("78% nitrogen, 21% oxygen", 0.99),
            "heart beats per day": ("100,000 times", 0.95),
            "sound cannot travel through": ("vacuum", 0.99),
            "diamonds are made of": ("carbon", 0.99),
            "gold does not": ("rust or tarnish", 0.99),
            "largest land animal": ("African elephant", 0.99),
            "fastest animal": ("peregrine falcon", 0.99),
            "fastest land animal": ("cheetah", 0.99),
            "largest animal": ("blue whale", 0.99),
            "smallest mammal": ("bumblebee bat", 0.95),
        }
        for pattern, (answer, conf) in science.items():
            if pattern in q:
                return {"answer": answer, "confidence": conf}

        # Discoverers / inventors
        discoverers = {
            "who discovered penicillin": ("Alexander Fleming", 0.99),
            "who discovered gravity": ("Isaac Newton", 0.99),
            "who discovered america": ("Christopher Columbus", 0.99),
            "who invented the telephone": ("Alexander Graham Bell", 0.99),
            "who invented the light bulb": ("Thomas Edison", 0.99),
            "who invented the printing press": ("Johannes Gutenberg", 0.99),
            "who discovered evolution": ("Charles Darwin", 0.99),
            "who wrote the origin of species": ("Charles Darwin", 0.99),
            "who discovered penicillin?": ("Alexander Fleming", 0.99),
            "who proposed relativity": ("Albert Einstein", 0.99),
            "who developed the theory of evolution": ("Charles Darwin", 0.99),
            "who discovered x-rays": ("Wilhelm Roentgen", 0.99),
            "who discovered radioactivity": ("Marie Curie", 0.99),
            "who invented the airplane": ("Wright Brothers", 0.99),
        }
        for pattern, (answer, conf) in discoverers.items():
            if pattern in q:
                return {"answer": answer, "confidence": conf}

        # History dates
        history = {
            "world war ii": ("1939-1945", 0.99),
            "world war 2": ("1939-1945", 0.99),
            "wwii": ("1939-1945", 0.99),
            "french revolution": ("1789", 0.99),
            "american independence": ("1776", 0.99),
            "declaration of independence": ("1776", 0.99),
            "moon landing year": ("1969", 0.99),
            "berlin wall fell": ("1989", 0.99),
            "industrial revolution": ("18th century", 0.95),
            "renaissance": ("14th-17th century", 0.95),
            "fall of roman empire": ("476 AD", 0.99),
            "magna carta": ("1215", 0.99),
        }
        for pattern, (answer, conf) in history.items():
            if pattern in q:
                return {"answer": answer, "confidence": conf}

        # Geography
        geography = {
            "largest island": ("Greenland", 0.99),
            "coldest continent": ("Antarctica", 0.99),
            "lowest point on earth": ("Dead Sea", 0.99),
            "highest point on earth": ("Mount Everest", 0.99),
            "time zones": ("24", 0.99),
            "how many time zones": ("24", 0.99),
            "largest country by area": ("Russia", 0.99),
            "most populous country": ("India", 0.95),
            "most spoken language": ("Mandarin Chinese", 0.95),
            "most spoken language in the world": ("Mandarin Chinese", 0.95),
        }
        for pattern, (answer, conf) in geography.items():
            if pattern in q:
                return {"answer": answer, "confidence": conf}

        # Math (arithmetic)
        math_match = re.search(r'what is (\d+) \+ (\d+)', q)
        if math_match:
            a, b = int(math_match.group(1)), int(math_match.group(2))
            return {"answer": str(a + b), "confidence": 0.99}

        math_match = re.search(r'what is (\d+) \* (\d+)', q)
        if math_match:
            a, b = int(math_match.group(1)), int(math_match.group(2))
            return {"answer": str(a * b), "confidence": 0.99}

        math_match = re.search(r'what is (\d+) - (\d+)', q)
        if math_match:
            a, b = int(math_match.group(1)), int(math_match.group(2))
            return {"answer": str(a - b), "confidence": 0.99}

        math_match = re.search(r'what is (\d+) / (\d+)', q)
        if math_match:
            a, b = int(math_match.group(1)), int(math_match.group(2))
            if b != 0:
                return {"answer": str(a / b), "confidence": 0.99}

        percent_match = re.search(r'(\d+)% of (\d+)', q)
        if percent_match:
            pct, num = int(percent_match.group(1)), int(percent_match.group(2))
            return {"answer": str(int(num * pct / 100)), "confidence": 0.99}

        return None

    def _semantic_evidence_score(self, query: str, evidence: list[str]) -> dict | None:
        """Use MiniLM to score evidence relevance."""
        if not self._miniLM or not evidence:
            return None

        scores = []
        for ev in evidence:
            result = self._miniLM.similarity(query, ev)
            scores.append((ev, result.score))

        if not scores:
            return None

        scores.sort(key=lambda x: x[1], reverse=True)
        best_ev, best_score = scores[0]

        direction = "supports"
        if any(neg in best_ev.lower() for neg in ["not", "no", "never", "fail", "refute"]):
            direction = "refutes"

        return {
            "answer": direction,
            "confidence": min(0.9, best_score * 1.2),
            "best_evidence": best_ev,
        }

    def _relay_predict(self, query: str) -> dict | None:
        """Use trained Relay Transformer for token prediction."""
        if not self._relay_model or not self._relay_tokenizer:
            return None

        try:
            import torch
            prompt = f"Question: {query} Answer:"
            tokens = self._relay_tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long)

            with torch.no_grad():
                logits, _ = self._relay_model(input_ids)

            top_token = logits[0, -1, :].argmax().item()
            predicted = self._relay_tokenizer.decode([top_token])

            return {
                "answer": predicted.strip(),
                "confidence": 0.5,
            }
        except Exception:
            return None


# Standalone test
if __name__ == "__main__":
    print("=" * 70)
    print("SWEEP HYBRID ENGINE — TEST")
    print("=" * 70)

    engine = HybridEngine()

    tests = [
        ("What is the capital of France?", [], "Paris"),
        ("What is 15% of 200?", [], "30"),
        ("What is the boiling point of water?", [], "100"),
        ("Is exercise good for health?", ["Exercise reduces heart disease"], "supports"),
        ("What is the largest planet?", [], "Jupiter"),
        ("What year did WWII end?", [], "1945"),
        ("Who discovered penicillin?", [], "Fleming"),
        ("What does DNA stand for?", [], "deoxyribonucleic acid"),
        ("How many bones in the human body?", [], "206"),
        ("What is the chemical formula for water?", [], "H2O"),
        ("What is the speed of light?", [], "299792458"),
        ("What is 25 * 4?", [], "100"),
        ("What is the capital of Japan?", [], "Tokyo"),
        ("What is the capital of India?", [], "New Delhi"),
        ("What is the largest ocean?", [], "Pacific"),
        ("What is the tallest mountain?", [], "Everest"),
    ]

    correct = 0
    for q, evidence, expected in tests:
        t0 = time.perf_counter()
        result = engine.answer(q, evidence)
        latency = (time.perf_counter() - t0) * 1000

        found = expected.lower() in result.answer.lower()
        if found:
            correct += 1

        print(f"  {'PASS' if found else 'FAIL'}: {q[:45]}...")
        print(f"       Answer: {result.answer} (conf={result.confidence:.2f}, method={result.method})")
        print(f"       Components: {result.components_used}, Latency: {latency:.0f}ms")

    print(f"\nAccuracy: {correct}/{len(tests)} ({correct/len(tests):.1%})")
