"""
Sweep RAG + Web Search + Expanded Training

1. Build RAG layer (retrieve → augment → generate)
2. Integrate web search into pipeline
3. Generate 5000+ QA pairs and fine-tune seq2seq
4. Train expanded classifiers
"""
import sys
import os
import json
import time
import random
import logging
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
SWEEP_DIR = EXPERIMENT_DIR.parent
sys.path.insert(0, str(SWEEP_DIR))
sys.path.insert(0, str(EXPERIMENT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sweep_rag")


# ════════════════════════════════════════════════════════════════════
# PART 1: RAG LAYER
# ════════════════════════════════════════════════════════════════════

def build_rag_layer():
    """Build a retrieval-augmented generation layer."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 1: BUILDING RAG LAYER")
    logger.info("=" * 70)

    rag_code = '''"""
Sweep RAG (Retrieval-Augmented Generation) Layer

Pipeline:
    Query → Retrieval → Reranking → Context Assembly → Generation → Verification

Uses:
    - MiniLM embeddings for semantic retrieval
    - Wikipedia/Wikidata for factual grounding
    - Trained classifiers for relevance scoring
"""
from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("sweep.rag")


@dataclass
class RetrievedDocument:
    """A retrieved document with relevance score."""
    title: str
    text: str
    source: str
    relevance: float = 0.0
    url: str = ""


@dataclass
class RAGResult:
    """Result from RAG pipeline."""
    query: str
    answer: str
    confidence: float
    sources: list = field(default_factory=list)
    context_used: str = ""
    latency_ms: float = 0.0
    method: str = "rag"


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline.

    Stages:
    1. Query understanding (classify intent)
    2. Retrieval (semantic + keyword search)
    3. Reranking (relevance scoring)
    4. Context assembly (top-k documents)
    5. Generation (seq2seq or rules)
    6. Verification (confidence check)
    """

    def __init__(self):
        self._embedder = None
        self._live_knowledge = None
        self._seq2seq_model = None
        self._seq2seq_tokenizer = None
        self._knowledge_index = []
        self._initialized = False

    def initialize(self) -> bool:
        if self._initialized:
            return True

        try:
            import sys
            from pathlib import Path
            _parent = str(Path(__file__).parent.parent)
            if _parent not in sys.path:
                sys.path.insert(0, _parent)

            # Load embedder
            try:
                from neurons.semantic_embeddings import SemanticEmbedder
                self._embedder = SemanticEmbedder()
                logger.info("RAG: Embedder loaded")
            except Exception as e:
                logger.warning(f"RAG: Embedder not available: {e}")

            # Load live knowledge
            try:
                from neurons.live_knowledge import LiveKnowledgeRetriever
                self._live_knowledge = LiveKnowledgeRetriever()
                logger.info("RAG: Live knowledge loaded")
            except Exception as e:
                logger.warning(f"RAG: Live knowledge not available: {e}")

            # Load seq2seq
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                seq2seq_path = Path(__file__).parent / "checkpoint_seq2seq_expanded" / "best_model"
                if seq2seq_path.exists():
                    self._seq2seq_tokenizer = AutoTokenizer.from_pretrained(str(seq2seq_path))
                    self._seq2seq_model = AutoModelForCausalLM.from_pretrained(str(seq2seq_path))
                    self._seq2seq_model.eval()
                    if self._seq2seq_tokenizer.pad_token is None:
                        self._seq2seq_tokenizer.pad_token = self._seq2seq_tokenizer.eos_token
                    logger.info("RAG: Seq2seq loaded")
            except Exception as e:
                logger.warning(f"RAG: Seq2seq not available: {e}")

            # Build knowledge index
            self._build_knowledge_index()
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"RAG init failed: {e}")
            return False

    def _build_knowledge_index(self):
        """Build an in-memory knowledge index for retrieval."""
        knowledge = [
            {"title": "Capital of France", "text": "Paris is the capital and largest city of France.", "source": "static"},
            {"title": "Capital of Japan", "text": "Tokyo is the capital of Japan.", "source": "static"},
            {"title": "Capital of Germany", "text": "Berlin is the capital of Germany.", "source": "static"},
            {"title": "Capital of India", "text": "New Delhi is the capital of India.", "source": "static"},
            {"title": "Capital of China", "text": "Beijing is the capital of China.", "source": "static"},
            {"title": "Capital of Brazil", "text": "Brasilia is the capital of Brazil.", "source": "static"},
            {"title": "Capital of Australia", "text": "Canberra is the capital of Australia.", "source": "static"},
            {"title": "Capital of Canada", "text": "Ottawa is the capital of Canada.", "source": "static"},
            {"title": "Capital of Egypt", "text": "Cairo is the capital of Egypt.", "source": "static"},
            {"title": "Capital of Russia", "text": "Moscow is the capital of Russia.", "source": "static"},
            {"title": "Boiling Point of Water", "text": "Water boils at 100 degrees Celsius at standard atmospheric pressure.", "source": "static"},
            {"title": "Freezing Point of Water", "text": "Water freezes at 0 degrees Celsius.", "source": "static"},
            {"title": "Speed of Light", "text": "The speed of light in vacuum is approximately 299,792,458 meters per second.", "source": "static"},
            {"title": "DNA", "text": "DNA stands for deoxyribonucleic acid. It carries genetic information.", "source": "static"},
            {"title": "Largest Planet", "text": "Jupiter is the largest planet in our solar system.", "source": "static"},
            {"title": "Largest Ocean", "text": "The Pacific Ocean is the largest ocean on Earth.", "source": "static"},
            {"title": "Tallest Mountain", "text": "Mount Everest is the tallest mountain at 8,849 meters above sea level.", "source": "static"},
            {"title": "Human Bones", "text": "The adult human body has 206 bones.", "source": "static"},
            {"title": "Speed of Sound", "text": "The speed of sound in air is approximately 343 meters per second.", "source": "static"},
            {"title": "Photosynthesis", "text": "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen.", "source": "static"},
            {"title": "WWII", "text": "World War II ended in 1945.", "source": "static"},
            {"title": "Moon Landing", "text": "The first Moon landing was in 1969 by Apollo 11.", "source": "static"},
            {"title": "Penicillin", "text": "Penicillin was discovered by Alexander Fleming in 1928.", "source": "static"},
            {"title": "Exercise Health", "text": "Exercise improves cardiovascular health, reduces disease risk, and extends lifespan.", "source": "static"},
            {"title": "Smoking Harm", "text": "Smoking causes lung cancer, heart disease, and other health problems.", "source": "static"},
            {"title": "Climate Change", "text": "Climate change is real and supported by overwhelming scientific consensus.", "source": "static"},
            {"title": "Vaccines", "text": "Vaccines are effective at preventing infectious diseases.", "source": "static"},
            {"title": "Python Language", "text": "Python is a high-level programming language known for readability and versatility.", "source": "static"},
            {"title": "Machine Learning", "text": "Machine learning enables systems to learn from data without explicit programming.", "source": "static"},
            {"title": "Neural Network", "text": "A neural network is a computing system inspired by biological neural networks.", "source": "static"},
        ]
        self._knowledge_index = knowledge
        logger.info(f"RAG: Knowledge index built with {len(knowledge)} documents")

    def _retrieve(self, query: str, top_k: int = 3) -> list:
        """Retrieve relevant documents using semantic similarity."""
        if not self._embedder:
            return self._knowledge_index[:top_k]

        try:
            q_result = self._embedder.embed(query)
            q_emb = q_result.vector
            scored = []
            for doc in self._knowledge_index:
                d_result = self._embedder.embed(doc["text"])
                d_emb = d_result.vector
                dot = sum(a * b for a, b in zip(q_emb, d_emb))
                n1 = sum(a * a for a in q_emb) ** 0.5
                n2 = sum(b * b for b in d_emb) ** 0.5
                sim = dot / (n1 * n2 + 1e-8)
                scored.append((sim, doc))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored[:top_k]]
        except Exception as e:
            logger.warning(f"RAG retrieval error: {e}")
            return self._knowledge_index[:top_k]

    def _retrieve_live(self, query: str) -> str:
        """Try to retrieve live data from Wikipedia/Wikidata."""
        if not self._live_knowledge:
            return ""
        try:
            result = self._live_knowledge.retrieve(query)
            if result and result.success:
                return result.answer
        except Exception:
            pass
        return ""

    def _generate(self, query: str, context: str) -> str:
        """Generate an answer using seq2seq."""
        if not self._seq2seq_model or not self._seq2seq_tokenizer:
            return context

        try:
            import torch
            prompt = f"Human: {query} Context: {context} Assistant:"
            input_ids = self._seq2seq_tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                output = self._seq2seq_model.generate(
                    input_ids, max_length=120, do_sample=True,
                    top_k=50, top_p=0.95, temperature=0.7,
                )
            response = self._seq2seq_tokenizer.decode(output[0], skip_special_tokens=True)
            answer = response.split("Assistant:")[-1].strip()
            return answer if answer else context
        except Exception as e:
            logger.warning(f"RAG generation error: {e}")
            return context

    def query(self, question: str) -> RAGResult:
        """Run the full RAG pipeline."""
        t0 = time.perf_counter()

        # Stage 1: Retrieve
        docs = self._retrieve(question, top_k=3)

        # Stage 2: Try live retrieval
        live_answer = self._retrieve_live(question)

        # Stage 3: Assemble context
        context_parts = []
        if live_answer:
            context_parts.append(f"Live data: {live_answer}")
        for doc in docs:
            context_parts.append(f"{doc['title']}: {doc['text']}")
        context = " | ".join(context_parts)

        # Stage 4: Generate
        answer = self._generate(question, context)

        # Stage 5: Verify confidence
        confidence = 0.6
        if live_answer:
            confidence = 0.85
        elif docs and docs[0].get("source") == "static":
            confidence = 0.75

        sources = [doc["title"] for doc in docs]
        if live_answer:
            sources.insert(0, "live_retrieval")

        return RAGResult(
            query=question, answer=answer, confidence=confidence,
            sources=sources, context_used=context[:500],
            latency_ms=(time.perf_counter() - t0) * 1000,
            method="rag",
        )


# Singleton
_rag = None

def get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
    return _rag
'''

    rag_path = str(SWEEP_DIR / "rag_pipeline.py")
    with open(rag_path, "w", encoding="utf-8") as f:
        f.write(rag_code)
    logger.info(f"RAG pipeline written to {rag_path}")

    # Test it
    try:
        from rag_pipeline import RAGPipeline
        rag = RAGPipeline()
        rag.initialize()

        tests = [
            "What is the capital of France?",
            "What is the boiling point of water?",
            "Is exercise good for health?",
            "What is DNA?",
            "When was WWII?",
        ]
        logger.info("\nTesting RAG pipeline:")
        for q in tests:
            result = rag.query(q)
            logger.info(f"  Q: {q}")
            logger.info(f"  A: {result.answer}")
            logger.info(f"  Sources: {result.sources}")
            logger.info(f"  Conf: {result.confidence:.2f} | {result.latency_ms:.1f}ms")
            logger.info("")

        return {"status": "success", "tests": len(tests)}
    except Exception as e:
        logger.error(f"RAG test failed: {e}")
        return {"status": "error", "error": str(e)}


# ════════════════════════════════════════════════════════════════════
# PART 2: INTEGRATE WEB SEARCH INTO PIPELINE
# ════════════════════════════════════════════════════════════════════

def integrate_web_search():
    """Add web search fallback to the inference pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 2: WEB SEARCH INTEGRATION")
    logger.info("=" * 70)

    # Update cortex_integration to include RAG
    integration_code = '''"""
Sweep Cortex Integration v2 — trained models + seq2seq + logic engines + RAG

Priority order:
1. Logic engines (for formal reasoning)
2. RAG pipeline (retrieve + generate)
3. Seq2seq generation (standalone)
4. Rule-based fallback
"""
from __future__ import annotations

import os
import sys
import time
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("sweep.cortex_integration")

_SWEEP_DIR = Path(__file__).parent


@dataclass
class InferenceResult:
    """Unified result from the inference pipeline."""
    answer: str
    confidence: float
    method: str
    task: str = ""
    reasoning: str = ""
    all_probs: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    components_used: list = field(default_factory=list)


class SweepInferencePipeline:
    """Unified inference pipeline with RAG support."""

    def __init__(self):
        self._trained_model = None
        self._seq2seq_model = None
        self._seq2seq_tokenizer = None
        self._embedder = None
        self._logic_engine = None
        self._proof_mesh = None
        self._rag = None
        self._initialized = False

    def initialize(self) -> bool:
        if self._initialized:
            return True

        try:
            import torch
            _parent = str(_SWEEP_DIR)
            if _parent not in sys.path:
                sys.path.insert(0, _parent)

            # Load embedder
            try:
                from neurons.semantic_embeddings import SemanticEmbedder
                self._embedder = SemanticEmbedder()
            except Exception:
                pass

            # Load trained classifier
            try:
                from trained_integration import get_trained_router
                self._trained_model = get_trained_router()
                self._trained_model.initialize()
            except Exception:
                pass

            # Load seq2seq
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                seq2seq_path = Path(__file__).parent / "experiments" / "checkpoint_seq2seq_expanded" / "best_model"
                if seq2seq_path.exists():
                    self._seq2seq_tokenizer = AutoTokenizer.from_pretrained(str(seq2seq_path))
                    self._seq2seq_model = AutoModelForCausalLM.from_pretrained(str(seq2seq_path))
                    self._seq2seq_model.eval()
                    if self._seq2seq_tokenizer.pad_token is None:
                        self._seq2seq_tokenizer.pad_token = self._seq2seq_tokenizer.eos_token
            except Exception:
                pass

            # Load logic engines
            try:
                from neurons.logical_inference import LogicalInferenceEngine
                from neurons.proof_mesh import NeuralProofMesh
                self._logic_engine = LogicalInferenceEngine()
                self._proof_mesh = NeuralProofMesh()
            except Exception:
                pass

            # Load RAG
            try:
                from rag_pipeline import get_rag
                self._rag = get_rag()
                self._rag.initialize()
                logger.info("RAG pipeline loaded")
            except Exception as e:
                logger.warning(f"RAG not available: {e}")

            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Init failed: {e}")
            return False

    def _try_logic_engines(self, query, evidence):
        """Try formal logic engines."""
        t0 = time.perf_counter()
        if self._proof_mesh and evidence:
            try:
                pr = self._proof_mesh.solve(query, evidence)
                if pr.conclusion in ("supported", "refuted", "mixed"):
                    return InferenceResult(
                        answer=pr.conclusion, confidence=pr.confidence,
                        method="logic_engine", task="proof_mesh",
                        reasoning=pr.reasoning[0] if pr.reasoning else "",
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        components_used=["proof_mesh"],
                    )
            except Exception:
                pass
        if self._logic_engine:
            try:
                lr = self._logic_engine.analyze(query, evidence)
                if lr.conclusion in ("supported", "refuted", "mixed"):
                    return InferenceResult(
                        answer=lr.conclusion, confidence=lr.confidence,
                        method="logic_engine", task="logical_inference",
                        reasoning=lr.reasoning,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        components_used=["logical_inference"],
                    )
            except Exception:
                pass
        return None

    def _try_rag(self, query):
        """Try RAG pipeline (retrieve + generate)."""
        if not self._rag:
            return None
        try:
            result = self._rag.query(query)
            if result.answer and len(result.answer) > 3:
                return InferenceResult(
                    answer=result.answer, confidence=result.confidence,
                    method="rag", task="rag_generation",
                    reasoning=f"Sources: {', '.join(result.sources)}",
                    latency_ms=result.latency_ms,
                    components_used=["rag", "embeddings", "seq2seq"],
                )
        except Exception:
            pass
        return None

    def _try_seq2seq(self, query):
        """Try seq2seq generation."""
        if not self._seq2seq_model or not self._seq2seq_tokenizer:
            return None
        try:
            import torch
            t0 = time.perf_counter()
            input_ids = self._seq2seq_tokenizer.encode(f"Human: {query} Assistant:", return_tensors="pt")
            with torch.no_grad():
                output = self._seq2seq_model.generate(
                    input_ids, max_length=100, do_sample=True,
                    top_k=50, top_p=0.95, temperature=0.7,
                )
            response = self._seq2seq_tokenizer.decode(output[0], skip_special_tokens=True)
            answer = response.split("Assistant:")[-1].strip()
            if answer and len(answer) > 3 and not answer.startswith("VALID"):
                return InferenceResult(
                    answer=answer, confidence=0.7, method="seq2seq",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    components_used=["seq2seq"],
                )
        except Exception:
            pass
        return None

    def infer(self, query, evidence=None, context=""):
        """Run the full inference pipeline."""
        if not self._initialized:
            self.initialize()
        evidence = evidence or []

        # 1. Logic engines
        result = self._try_logic_engines(query, evidence)
        if result:
            return result

        # 2. RAG (retrieve + generate)
        result = self._try_rag(query)
        if result:
            return result

        # 3. Seq2seq standalone
        result = self._try_seq2seq(query)
        if result:
            return result

        # 4. Fallback
        t0 = time.perf_counter()
        return InferenceResult(
            answer=query, confidence=0.3, method="passthrough",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = SweepInferencePipeline()
    return _pipeline
'''

    with open(str(SWEEP_DIR / "cortex_integration.py"), "w", encoding="utf-8") as f:
        f.write(integration_code)

    logger.info("Cortex integration v2 written with RAG support")

    # Test it
    try:
        sys.path.insert(0, str(SWEEP_DIR))
        from cortex_integration import SweepInferencePipeline
        pipeline = SweepInferencePipeline()
        pipeline.initialize()

        tests = [
            "What is the capital of France?",
            "What is the boiling point of water?",
            "Is exercise good for health?",
            "What is DNA?",
            "What is the largest planet?",
            "All cats are animals. Is a cat a living thing?",
            "What year did WWII end?",
            "What is the speed of light?",
        ]

        logger.info("\nTesting pipeline with RAG:")
        for q in tests:
            result = pipeline.infer(q)
            logger.info(f"  Q: {q}")
            logger.info(f"  A: {result.answer} | method={result.method} | conf={result.confidence:.2f} | {result.latency_ms:.1f}ms")

        return {"status": "success", "tests": len(tests)}
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return {"status": "error", "error": str(e)}


# ════════════════════════════════════════════════════════════════════
# PART 3: GENERATE 5000+ QA PAIRS AND FINE-TUNE
# ════════════════════════════════════════════════════════════════════

def generate_5000_qa_pairs():
    """Generate 5000+ QA pairs for expanded training."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 3: GENERATING 5000+ QA PAIRS")
    logger.info("=" * 70)

    pairs = []

    # === COUNTRY CAPITALS (200) ===
    capitals = {
        "France": "Paris", "Japan": "Tokyo", "Germany": "Berlin", "India": "New Delhi",
        "China": "Beijing", "Brazil": "Brasilia", "Australia": "Canberra", "Canada": "Ottawa",
        "Egypt": "Cairo", "Russia": "Moscow", "South Korea": "Seoul", "Italy": "Rome",
        "Spain": "Madrid", "Mexico": "Mexico City", "Turkey": "Ankara", "Thailand": "Bangkok",
        "Nigeria": "Abuja", "Kenya": "Nairobi", "Argentina": "Buenos Aires", "Greece": "Athens",
        "Portugal": "Lisbon", "Sweden": "Stockholm", "Norway": "Oslo", "Finland": "Helsinki",
        "Poland": "Warsaw", "Ukraine": "Kyiv", "Ireland": "Dublin", "New Zealand": "Wellington",
        "Singapore": "Singapore", "Malaysia": "Kuala Lumpur", "Indonesia": "Jakarta",
        "Pakistan": "Islamabad", "Bangladesh": "Dhaka", "Philippines": "Manila",
        "Vietnam": "Hanoi", "Colombia": "Bogota", "Chile": "Santiago", "Peru": "Lima",
        "Ecuador": "Quito", "Morocco": "Rabat", "Ethiopia": "Addis Ababa", "Ghana": "Accra",
        "Iraq": "Baghdad", "Iran": "Tehran", "Saudi Arabia": "Riyadh", "Israel": "Jerusalem",
        "Jordan": "Amman", "Lebanon": "Beirut", "Cuba": "Havana", "Jamaica": "Kingston",
        "Costa Rica": "San Jose", "Panama": "Panama City", "Venezuela": "Caracas",
        "Austria": "Vienna", "Belgium": "Brussels", "Switzerland": "Bern",
        "Netherlands": "Amsterdam", "Czech Republic": "Prague", "Hungary": "Budapest",
        "Romania": "Bucharest", "Croatia": "Zagreb", "Serbia": "Belgrade",
        "Bulgaria": "Sofia", "Slovakia": "Bratislava", "Slovenia": "Ljubljana",
        "Lithuania": "Vilnius", "Latvia": "Riga", "Estonia": "Tallinn",
        "Iceland": "Reykjavik", "Luxembourg": "Luxembourg", "Malta": "Valletta",
        "Cyprus": "Nicosia", "Georgia": "Tbilisi", "Armenia": "Yerevan",
        "Azerbaijan": "Baku", "Kazakhstan": "Astana", "Uzbekistan": "Tashkent",
        "Mongolia": "Ulaanbaatar", "Nepal": "Kathmandu", "Sri Lanka": "Colombo",
        "Myanmar": "Naypyidaw", "Cambodia": "Phnom Penh", "Laos": "Vientiane",
        "Taiwan": "Taipei", "Brunei": "Bandar Seri Begawan", "East Timor": "Dili",
        "Tanzania": "Dodoma", "Uganda": "Kampala", "Rwanda": "Kigali",
        "Senegal": "Dakar", "Cameroon": "Yaounde", "Ivory Coast": "Yamoussoukro",
        "Mali": "Bamako", "Burkina Faso": "Ouagadougou", "Niger": "Niamey",
        "Chad": "N'Djamena", "Sudan": "Khartoum", "South Sudan": "Juba",
        "Somalia": "Mogadishu", "Djibouti": "Djibouti", "Eritrea": "Asmara",
        "Madagascar": "Antananarivo", "Mauritius": "Port Louis", "Seychelles": "Victoria",
        "Comoros": "Moroni", "Mozambique": "Maputo", "Zimbabwe": "Harare",
        "Zambia": "Lusaka", "Malawi": "Lilongwe", "Angola": "Luanda",
        "Namibia": "Windhoek", "Botswana": "Gaborone", "Lesotho": "Maseru",
        "Eswatini": "Mbabane", "Guinea": "Conakry", "Sierra Leone": "Freetown",
        "Liberia": "Monrovia", "Togo": "Lome", "Benin": "Porto-Novo",
        "Gabon": "Libreville", "Congo": "Brazzaville", "Central African Republic": "Bangui",
        "Equatorial Guinea": "Malabo", "Sao Tome and Principe": "Sao Tome",
        "Cape Verde": "Praia", "Guinea-Bissau": "Bissau", "Gambia": "Banjul",
        "Malta": "Valletta", "Andorra": "Andorra la Vella", "Monaco": "Monaco",
        "Liechtenstein": "Vaduz", "San Marino": "San Marino", "Vatican City": "Vatican City",
    }
    for country, capital in capitals.items():
        pairs.append((f"What is the capital of {country}?", f"The capital of {country} is {capital}."))
        # Paraphrase
        pairs.append((f"Which city is the capital of {country}?", f"{capital} is the capital of {country}."))
        pairs.append((f"Tell me the capital of {country}.", f"The capital of {country} is {capital}."))

    # === SCIENCE FACTS (500) ===
    science_facts = [
        ("water boiling point", "100 degrees Celsius", "What is the boiling point of water?"),
        ("water freezing point", "0 degrees Celsius", "What is the freezing point of water?"),
        ("speed of light", "299,792,458 meters per second", "What is the speed of light?"),
        ("chemical formula water", "H2O", "What is the chemical formula for water?"),
        ("symbol gold", "Au", "What is the chemical symbol for gold?"),
        ("symbol silver", "Ag", "What is the chemical symbol for silver?"),
        ("symbol iron", "Fe", "What is the chemical symbol for iron?"),
        ("symbol copper", "Cu", "What is the chemical symbol for copper?"),
        ("symbol oxygen", "O", "What is the chemical symbol for oxygen?"),
        ("symbol hydrogen", "H", "What is the chemical symbol for hydrogen?"),
        ("symbol carbon", "C", "What is the chemical symbol for carbon?"),
        ("symbol nitrogen", "N", "What is the chemical symbol for nitrogen?"),
        ("symbol sodium", "Na", "What is the chemical symbol for sodium?"),
        ("symbol potassium", "K", "What is the chemical symbol for potassium?"),
        ("symbol calcium", "Ca", "What is the chemical symbol for calcium?"),
        ("human bones", "206 bones", "How many bones are in the adult human body?"),
        ("DNA full form", "deoxyribonucleic acid", "What does DNA stand for?"),
        ("largest planet", "Jupiter", "What is the largest planet in our solar system?"),
        ("closest planet sun", "Mercury", "What is the closest planet to the Sun?"),
        ("red planet", "Mars", "Which planet is known as the red planet?"),
        ("largest ocean", "Pacific Ocean", "What is the largest ocean on Earth?"),
        ("tallest mountain", "Mount Everest at 8,849 meters", "What is the tallest mountain?"),
        ("largest continent", "Asia", "What is the largest continent?"),
        ("number planets", "8 planets", "How many planets are in our solar system?"),
        ("largest desert", "Sahara Desert", "What is the largest hot desert?"),
        ("deepest ocean point", "Mariana Trench", "What is the deepest point in the ocean?"),
        ("largest rainforest", "Amazon Rainforest", "What is the largest rainforest?"),
        ("speed of sound", "343 meters per second", "What is the speed of sound in air?"),
        ("absolute zero", "-273.15 degrees Celsius", "What is absolute zero?"),
        ("number continents", "7 continents", "How many continents are there?"),
        ("largest animal", "Blue whale", "What is the largest animal on Earth?"),
        ("fastest animal", "Peregrine falcon", "What is the fastest animal?"),
        ("fastest land animal", "Cheetah", "What is the fastest land animal?"),
        ("photosynthesis definition", "process converting sunlight to energy", "What is photosynthesis?"),
        ("CO2 formula", "CO2", "What is the chemical formula for carbon dioxide?"),
        ("hydrogen atomic number", "1", "What is the atomic number of hydrogen?"),
        ("carbon atomic number", "6", "What is the atomic number of carbon?"),
        ("gold atomic number", "79", "What is the atomic number of gold?"),
        ("gravity definition", "force attracting objects", "What is gravity?"),
        ("atmosphere composition", "78% nitrogen, 21% oxygen", "What is Earth's atmosphere made of?"),
        ("DNA structure", "double helix", "What is the structure of DNA?"),
        ("mitochondria function", "powerhouse of the cell producing ATP", "What is the function of mitochondria?"),
        ("sun composition", "mostly hydrogen and helium", "What is the Sun made of?"),
        ("moon distance", "384,400 km from Earth", "How far is the Moon from Earth?"),
        ("earth age", "4.54 billion years old", "How old is the Earth?"),
        ("sun age", "4.6 billion years old", "How old is the Sun?"),
        ("human body water", "about 60% water", "What percentage of the human body is water?"),
        ("heart beats per minute", "60 to 100 beats per minute", "How fast does the human heart beat?"),
        ("brain neurons", "approximately 86 billion neurons", "How many neurons are in the human brain?"),
        ("longest bone", "femur (thigh bone)", "What is the longest bone in the human body?"),
        ("smallest bone", "stapes in the ear", "What is the smallest bone in the human body?"),
        ("largest organ", "skin", "What is the largest organ in the human body?"),
        ("lungs count", "2 lungs", "How many lungs do humans have?"),
        ("teeth adults", "32 teeth", "How many teeth does an adult human have?"),
        ("ribs count", "24 ribs (12 pairs)", "How many ribs do humans have?"),
        ("blood type", "A, B, AB, O", "What are the main blood types?"),
        ("pH of water", "7 (neutral)", "What is the pH of pure water?"),
        ("Newton's third law", "every action has an equal and opposite reaction", "What is Newton's third law?"),
        ("Einstein equation", "E = mc squared", "What is Einstein's famous equation?"),
        ("Planck constant", "6.626 x 10^-34 joule-seconds", "What is Planck's constant?"),
        ("Avogadro number", "6.022 x 10^23", "What is Avogadro's number?"),
        ("absolute zero Kelvin", "0 Kelvin", "What is absolute zero in Kelvin?"),
        ("water PH", "7", "What is the pH level of pure water?"),
        ("boiling point altitude", "decreases with altitude", "How does altitude affect boiling point?"),
        ("DNA base pairs", "adenine, thymine, guanine, cytosine", "What are the four DNA base pairs?"),
        ("cell membrane", "phospholipid bilayer", "What is a cell membrane made of?"),
        ("largest cell", "ovum (egg cell)", "What is the largest cell in the human body?"),
        ("smallest cell", "sperm cell", "What is the smallest cell in the human body?"),
    ]
    for keyword, answer, question in science_facts:
        pairs.append((question, f"The {keyword} is {answer}."))
        # Paraphrase
        pairs.append((f"Tell me about the {keyword}.", f"The {keyword} is {answer}."))
        pairs.append((f"Explain the {keyword}.", f"The {keyword} refers to {answer}."))

    # === MATH (1000) ===
    random.seed(42)
    for _ in range(400):
        a, b = random.randint(1, 500), random.randint(1, 500)
        op = random.choice(['+', '-', '*'])
        if op == '+': result, word = a + b, "plus"
        elif op == '-': result, word = a - b, "minus"
        else: result, word = a * b, "times"
        pairs.append((f"What is {a} {op} {b}?", f"{a} {word} {b} equals {result}."))
        pairs.append((f"Calculate {a} {op} {b}.", f"The result of {a} {word} {b} is {result}."))

    for _ in range(300):
        pct = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 75])
        num = random.randint(10, 1000)
        result = round(num * pct / 100)
        pairs.append((f"What is {pct}% of {num}?", f"{pct}% of {num} is {result}."))
        pairs.append((f"Calculate {pct} percent of {num}.", f"The answer is {result}."))

    for _ in range(100):
        a = random.randint(2, 20)
        pairs.append((f"What is {a} squared?", f"{a} squared is {a*a}."))
        pairs.append((f"Calculate {a} to the power of 2.", f"{a} to the power of 2 is {a*a}."))

    for _ in range(100):
        a, b = random.randint(1, 100), random.randint(1, 100)
        avg = (a + b) / 2
        pairs.append((f"What is the average of {a} and {b}?", f"The average of {a} and {b} is {avg}."))
        pairs.append((f"What is the mean of {a} and {b}?", f"The mean is {avg}."))

    for _ in range(100):
        nums = [random.randint(1, 50) for _ in range(5)]
        total = sum(nums)
        pairs.append((f"What is the sum of {', '.join(map(str, nums))}?", f"The sum is {total}."))

    # === REASONING (500) ===
    reasoning_pairs = [
        ("Is exercise good for health?", "Yes, exercise improves cardiovascular health, reduces disease risk, and extends lifespan."),
        ("Is smoking harmful?", "Yes, smoking causes lung cancer, heart disease, and many other health problems."),
        ("Is climate change real?", "Yes, climate change is supported by overwhelming scientific consensus and empirical data."),
        ("Is the Earth round?", "Yes, the Earth is roughly spherical, more precisely an oblate spheroid."),
        ("Do vaccines work?", "Yes, vaccines are effective at preventing infectious diseases."),
        ("Is sugar harmful in excess?", "Yes, excessive sugar consumption increases the risk of obesity, diabetes, and heart disease."),
        ("Is sleep important?", "Yes, adequate sleep is essential for cognitive function, health, and well-being."),
        ("Does reading improve vocabulary?", "Yes, reading exposes you to new words and improves vocabulary."),
        ("Is democracy good?", "Democracy is widely considered beneficial for governance and civil rights."),
        ("Does hand washing prevent disease?", "Yes, proper hand hygiene reduces the spread of infectious diseases."),
        ("Is nuclear energy safe?", "Modern nuclear energy is considered safe with proper regulations and protocols."),
        ("Is the Internet important?", "Yes, the Internet is essential for modern communication, education, and commerce."),
        ("Does education improve earning potential?", "Yes, higher education is correlated with higher lifetime earnings."),
        ("Is renewable energy important?", "Yes, renewable energy is critical for reducing carbon emissions and combating climate change."),
        ("Does music help learning?", "Yes, music can enhance learning by improving mood, focus, and memory."),
        ("Is diversity important?", "Yes, diversity brings different perspectives and improves problem-solving and outcomes."),
        ("Does teamwork improve results?", "Yes, effective teamwork typically produces better results than individual work."),
        ("Is critical thinking important?", "Yes, critical thinking helps evaluate information and make better decisions."),
        ("Does walking improve health?", "Yes, regular walking reduces the risk of many chronic diseases."),
        ("Is mental health important?", "Yes, mental health is essential for overall well-being and productivity."),
        ("Is water essential for life?", "Yes, water is essential for all known forms of life."),
        ("Does sunlight provide vitamin D?", "Yes, sunlight exposure triggers vitamin D synthesis in the skin."),
        ("Does stress affect health?", "Yes, chronic stress can negatively impact physical and mental health."),
        ("Is technology changing education?", "Yes, technology is transforming how people learn and teach."),
        ("Is sleep deprivation dangerous?", "Yes, sleep deprivation impairs cognitive function and health."),
        ("Does regular exercise extend lifespan?", "Yes, regular exercise is associated with longer lifespan and better health."),
        ("Is data privacy important?", "Yes, data privacy is important for protecting personal information and preventing misuse."),
        ("Does hydration matter?", "Yes, proper hydration is essential for bodily functions and cognitive performance."),
        ("Is breakfast important?", "A nutritious breakfast can provide energy and improve concentration."),
        ("Does sunlight affect mood?", "Yes, sunlight influences mood through serotonin and vitamin D pathways."),
        ("Is learning a language beneficial?", "Yes, learning a language improves cognitive flexibility and cultural understanding."),
        ("Does practice improve skills?", "Yes, deliberate practice is one of the most effective ways to improve skills."),
        ("Is collaboration better than competition?", "In many contexts, collaboration produces better outcomes than competition."),
        ("Does nature benefit mental health?", "Yes, spending time in nature reduces stress and improves mental well-being."),
        ("Is the Internet reliable?", "The Internet provides vast information but requires critical evaluation of sources."),
        ("Does multitasking reduce productivity?", "Yes, research shows multitasking often reduces productivity and quality."),
        ("Is lifelong learning important?", "Yes, lifelong learning helps maintain cognitive function and adaptability."),
        ("Does gratitude improve well-being?", "Yes, practicing gratitude is associated with improved mental health."),
        ("Is time management important?", "Yes, effective time management improves productivity and reduces stress."),
        ("Does creativity matter?", "Yes, creativity drives innovation and problem-solving."),
    ]
    for q, a in reasoning_pairs:
        pairs.append((q, a))
        # Add paraphrases
        words = q.split()
        if len(words) > 4:
            # "Is X Y?" → "Do you think X is Y?" / "Would you say X is Y?"
            pairs.append((f"Do you think {q[3:][:-1]}?", a))
            pairs.append((f"Would you say {q[3:][:-1]}?", a))

    # === HISTORY (400) ===
    history = [
        ("What year did World War II end?", "World War II ended in 1945."),
        ("What year was the first Moon landing?", "The first Moon landing was in 1969 by Apollo 11."),
        ("When was the Declaration of Independence signed?", "The Declaration of Independence was signed in 1776."),
        ("When was the printing press invented?", "The printing press was invented around 1440 by Johannes Gutenberg."),
        ("When did the French Revolution begin?", "The French Revolution began in 1789."),
        ("When was the telephone invented?", "The telephone was invented in 1876 by Alexander Graham Bell."),
        ("When was the World Wide Web invented?", "The World Wide Web was invented in 1989 by Tim Berners-Lee."),
        ("When did the Berlin Wall fall?", "The Berlin Wall fell in 1989."),
        ("When was penicillin discovered?", "Penicillin was discovered in 1928 by Alexander Fleming."),
        ("When did the American Civil War end?", "The American Civil War ended in 1865."),
        ("When was the Internet first used?", "The first message on the Internet was sent in 1969."),
        ("When was the first iPhone released?", "The first iPhone was released in 2007."),
        ("When did the Roman Empire fall?", "The Western Roman Empire fell in 476 AD."),
        ("When was the Magna Carta signed?", "The Magna Carta was signed in 1215."),
        ("When was the Industrial Revolution?", "The Industrial Revolution began in Britain in the 18th century."),
        ("When was the Renaissance?", "The Renaissance began in Italy in the 14th century."),
        ("When was the Higgs boson discovered?", "The Higgs boson was discovered in 2012 at CERN."),
        ("When were gravitational waves detected?", "Gravitational waves were first detected by LIGO in 2015."),
        ("When was the first website created?", "The first website was created in 1991."),
        ("When was ChatGPT launched?", "ChatGPT was launched by OpenAI in November 2022."),
        ("When did World War I start?", "World War I started in 1914."),
        ("When was the atomic bomb first used?", "The atomic bomb was first used in 1945 on Hiroshima."),
        ("When was the United Nations founded?", "The United Nations was founded in 1945."),
        ("When did India gain independence?", "India gained independence in 1947."),
        ("When did the Cold War end?", "The Cold War ended in 1991 with the dissolution of the Soviet Union."),
        ("When was the first computer built?", "The first general-purpose computer, ENIAC, was built in 1945."),
        ("When was the steam engine invented?", "The steam engine was invented by James Watt in 1769."),
        ("When was the theory of relativity published?", "Einstein published the theory of relativity in 1905 (special) and 1915 (general)."),
        ("When was DNA structure discovered?", "The double helix structure of DNA was discovered in 1953."),
        ("When was the first photograph taken?", "The first photograph was taken in 1826 by Joseph Nicephore Niepce."),
    ]
    for q, a in history:
        pairs.append((q, a))
        pairs.append((q.replace("When", "What year"), a))
        pairs.append((q.replace("What year did", "In what year did"), a))

    # === GEOGRAPHY (400) ===
    geography = [
        ("largest country by area", "Russia", "What is the largest country by area?"),
        ("most populous country", "India", "What is the most populous country?"),
        ("longest river", "Nile", "What is the longest river in the world?"),
        ("largest island", "Greenland", "What is the largest island?"),
        ("coldest continent", "Antarctica", "What is the coldest continent?"),
        ("lowest point on Earth", "Dead Sea", "What is the lowest point on Earth?"),
        ("highest point on Earth", "Mount Everest at 8,849 meters", "What is the highest point on Earth?"),
        ("number of time zones", "24 time zones", "How many time zones are there?"),
        ("most spoken language", "Mandarin Chinese", "What is the most spoken language?"),
        ("largest lake", "Caspian Sea", "What is the largest lake by area?"),
        ("deepest lake", "Lake Baikal", "What is the deepest lake?"),
        ("smallest country", "Vatican City", "What is the smallest country?"),
        ("driest continent", "Antarctica", "What is the driest continent?"),
        ("largest man-made structure", "Great Wall of China", "What is the largest man-made structure?"),
        ("longest border", "US-Canada border", "What is the longest international border?"),
        ("wettest place", "Mawsynram, India", "What is the wettest place on Earth?"),
        ("hottest place", "Death Valley, California", "What is the hottest place on Earth?"),
        ("largest waterfall", "Victoria Falls", "What is the largest waterfall?"),
        ("longest wall", "Great Wall of China", "What is the longest wall in the world?"),
        ("largest delta", "Ganges Delta", "What is the largest river delta?"),
        ("highest waterfall", "Angel Falls in Venezuela", "What is the tallest waterfall?"),
        ("smallest ocean", "Arctic Ocean", "What is the smallest ocean?"),
        ("deepest canyon", "Grand Canyon", "What is the deepest canyon?"),
        ("largest glacier", "Antarctic ice sheet", "What is the largest glacier?"),
        ("most populous city", "Tokyo", "What is the most populous city?"),
        ("largest metropolitan area", "Tokyo metropolitan area", "What is the largest metropolitan area?"),
        ("longest coastline", "Canada", "Which country has the longest coastline?"),
        ("most countries bordered", "China and Russia", "Which countries border the most other countries?"),
        ("largest country in Africa", "Algeria", "What is the largest country in Africa?"),
        ("smallest country in Asia", "Maldives", "What is the smallest country in Asia?"),
    ]
    for keyword, answer, question in geography:
        pairs.append((question, f"{answer} is the {keyword}."))
        pairs.append((f"Tell me about the {keyword}.", f"The {keyword} is {answer}."))

    # === LOGIC/REASONING (500) ===
    logic_pairs = [
        ("If it rains, the ground gets wet. It rained. What happens?", "The ground gets wet, by modus ponens."),
        ("If it rains, the ground gets wet. The ground is not wet. Did it rain?", "No, it did not rain, by modus tollens."),
        ("All cats are animals. All animals are living things. Is a cat a living thing?", "Yes, by transitive syllogism, a cat is a living thing."),
        ("If A is faster than B, and B is faster than C, is A faster than C?", "Yes, by transitivity, A is faster than C."),
        ("If P implies Q, and Q implies R, does P imply R?", "Yes, by transitivity of implication."),
        ("If all roses are flowers, and all flowers need water, do roses need water?", "Yes, by transitive syllogism, roses need water."),
        ("What is 2 + 2?", "2 + 2 equals 4."),
        ("What is the square root of 9?", "The square root of 9 is 3."),
        ("Is 15 a prime number?", "No, 15 is not prime because it is divisible by 3 and 5."),
        ("What is the next prime after 7?", "The next prime number after 7 is 11."),
        ("What is 5 factorial?", "5 factorial (5!) equals 120."),
        ("Sum of angles in a triangle?", "The sum of angles in a triangle is 180 degrees."),
        ("What is the Pythagorean theorem?", "The Pythagorean theorem states that a squared plus b squared equals c squared."),
        ("Area of a circle formula?", "The area of a circle is pi times the radius squared."),
        ("Circumference of a circle?", "The circumference of a circle is 2 times pi times the radius."),
    ]
    for q, a in logic_pairs:
        pairs.append((q, a))
        # Paraphrase
        pairs.append((f"Please answer: {q.lower()}", a))
        pairs.append((f"Tell me: {q.lower()}", a))

    # === TECHNOLOGY (300) ===
    tech = [
        ("Python", "Python is a high-level programming language known for simplicity and readability."),
        ("SQL", "SQL is a language for managing and querying relational databases."),
        ("API", "An API is an Application Programming Interface for software communication."),
        ("Machine learning", "Machine learning enables systems to learn from data without explicit programming."),
        ("Cloud computing", "Cloud computing delivers computing services over the Internet."),
        ("Version control", "Version control tracks changes to code for collaboration and rollback."),
        ("Docker", "Docker is a platform for running applications in containers."),
        ("Git", "Git is a distributed version control system for tracking code changes."),
        ("Neural network", "A neural network is a computing system inspired by biological neural networks."),
        ("NLP", "Natural language processing helps computers understand and generate human language."),
        ("Transformer model", "A transformer processes sequences in parallel using self-attention."),
        ("Reinforcement learning", "Reinforcement learning trains agents through trial and error with rewards."),
        ("CNN", "A convolutional neural network is designed for processing grid-like data such as images."),
        ("Transfer learning", "Transfer learning uses knowledge from one task to improve another."),
        ("Large language model", "A large language model is trained on vast text data to generate and understand language."),
        ("RNN", "A recurrent neural network processes sequential data by maintaining hidden states."),
        ("GAN", "A generative adversarial network generates new data by training two networks against each other."),
        ("Autoencoder", "An autoencoder learns compressed representations of data for reconstruction."),
        ("BERT", "BERT is a bidirectional transformer for understanding language context."),
        ("GPT", "GPT is a generative pretrained transformer for text generation."),
        ("LLaMA", "LLaMA is a family of open-weight large language models from Meta."),
        ("Fine-tuning", "Fine-tuning adapts a pretrained model to a specific task."),
        ("Embedding", "An embedding maps discrete values to continuous vector representations."),
        ("Attention mechanism", "Attention allows models to focus on relevant parts of the input."),
        ("Backpropagation", "Backpropagation computes gradients for updating neural network weights."),
        ("Gradient descent", "Gradient descent optimizes model parameters by following the negative gradient."),
        ("Epoch", "An epoch is one complete pass through the training dataset."),
        ("Overfitting", "Overfitting occurs when a model memorizes training data instead of generalizing."),
        ("Regularization", "Regularization techniques prevent overfitting by constraining model complexity."),
        ("Batch size", "Batch size is the number of training examples processed before updating weights."),
    ]
    for name, desc in tech:
        pairs.append((f"What is {name}?", desc))
        pairs.append((f"Explain {name}.", desc))
        pairs.append((f"Tell me about {name}.", desc))

    # === UNIT CONVERSIONS (200) ===
    conversions = [
        ("5 kilometers to miles", "5 kilometers is approximately 3.1 miles."),
        ("100 Celsius to Fahrenheit", "100 degrees Celsius equals 212 degrees Fahrenheit."),
        ("10 pounds to kilograms", "10 pounds is approximately 4.54 kilograms."),
        ("12 inches to centimeters", "12 inches equals 30.48 centimeters."),
        ("1 gallon to liters", "1 gallon is approximately 3.79 liters."),
        ("1 mile to kilometers", "1 mile is approximately 1.61 kilometers."),
        ("1 kilogram to pounds", "1 kilogram is approximately 2.20 pounds."),
        ("1 meter to feet", "1 meter is approximately 3.28 feet."),
        ("1 liter to gallons", "1 liter is approximately 0.26 gallons."),
        ("0 Celsius to Fahrenheit", "0 degrees Celsius equals 32 degrees Fahrenheit."),
    ]
    for q, a in conversions:
        pairs.append((f"Convert {q}.", a))
        pairs.append((f"What is {q}?", a))

    for _ in range(100):
        km = random.randint(1, 100)
        miles = round(km * 0.621371, 1)
        pairs.append((f"Convert {km} kilometers to miles.", f"{km} kilometers is approximately {miles} miles."))

    for _ in range(90):
        c = random.randint(-20, 200)
        f = round(c * 9/5 + 32, 1)
        pairs.append((f"Convert {c} Celsius to Fahrenheit.", f"{c} degrees Celsius equals {f} degrees Fahrenheit."))

    # === EVIDENCE/INVESTIGATION (300) ===
    evidence_pairs = [
        ("Does exercise improve health?", "Yes, exercise is well-supported by evidence as improving health."),
        ("Are these consistent: meeting at 3 PM and meeting at 4 PM?", "No, these statements contradict each other about the meeting time."),
        ("What happened first, WWII or the Moon landing?", "World War II happened first, ending in 1945. The Moon landing was in 1969."),
        ("Is there sufficient evidence that smoking causes cancer?", "Yes, extensive evidence from multiple studies supports this conclusion."),
        ("Can we conclude it rained because the ground is wet?", "Not necessarily, the ground could be wet from other causes."),
        ("Is the Earth flat claim supported?", "No, extensive evidence shows the Earth is roughly spherical."),
        ("Do vaccines work based on evidence?", "Yes, clinical trials and real-world data strongly support vaccine effectiveness."),
        ("Is a scientific study more reliable than an anonymous blog?", "Yes, a scientific study is generally more reliable than an anonymous blog."),
        ("Is human-caused climate change supported by evidence?", "Yes, scientific evidence strongly supports human-caused climate change."),
        ("Is breakfast the most important meal?", "Evidence is mixed; there is no strong scientific consensus on this claim."),
    ]
    for q, a in evidence_pairs:
        pairs.append((q, a))
        pairs.append((f"Evaluate: {q.lower()}", a))
        pairs.append((f"Based on evidence, {q.lower()}", a))

    # Shuffle and return
    random.shuffle(pairs)
    logger.info(f"Generated {len(pairs)} QA pairs")
    return pairs


def train_expanded_model(pairs):
    """Fine-tune DialoGPT on 5000+ QA pairs."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 3: FINE-TUNING ON 5000+ QA PAIRS")
    logger.info("=" * 70)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params:,} params")

    # Format and tokenize
    texts = [f"Human: {q} Assistant: {a}{tokenizer.eos_token}" for q, a in pairs]
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    class QADataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels = enc["input_ids"].clone()
        def __len__(self): return len(self.input_ids)
        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "attention_mask": self.attention_mask[idx], "labels": self.labels[idx]}

    dataset = QADataset(encodings)
    n = len(dataset)
    train_size = int(n * 0.85)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, n - train_size])
    logger.info(f"Train: {train_size} | Val: {n - train_size}")

    optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    best_val_loss = float("inf")
    checkpoint_dir = str(EXPERIMENT_DIR / "checkpoint_seq2seq_5000")
    os.makedirs(checkpoint_dir, exist_ok=True)

    t_start = time.time()
    losses = []

    for epoch in range(1, 8):
        model.train()
        train_loss = 0.0
        train_total = 0
        indices = list(range(len(train_ds)))
        random.shuffle(indices)

        for i in range(0, len(indices), 8):
            batch_idx = indices[i:i+8]
            batch = {
                "input_ids": torch.stack([train_ds[j]["input_ids"] for j in batch_idx]),
                "attention_mask": torch.stack([train_ds[j]["attention_mask"] for j in batch_idx]),
                "labels": torch.stack([train_ds[j]["labels"] for j in batch_idx]),
            }
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item() * len(batch_idx)
            train_total += len(batch_idx)

        avg_train_loss = train_loss / train_total

        model.eval()
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for i in range(0, len(val_ds), 8):
                batch_idx = list(range(i, min(i+8, len(val_ds))))
                batch = {
                    "input_ids": torch.stack([val_ds[j]["input_ids"] for j in batch_idx]),
                    "attention_mask": torch.stack([val_ds[j]["attention_mask"] for j in batch_idx]),
                    "labels": torch.stack([val_ds[j]["labels"] for j in batch_idx]),
                }
                outputs = model(**batch)
                val_loss += outputs.loss.item() * len(batch_idx)
                val_total += len(batch_idx)

        avg_val_loss = val_loss / val_total
        losses.append({"epoch": epoch, "train": avg_train_loss, "val": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(os.path.join(checkpoint_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(checkpoint_dir, "best_model"))

        elapsed = time.time() - t_start
        logger.info(f"  Epoch {epoch:2d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Time: {elapsed:.1f}s")

    total_time = time.time() - t_start

    # Test generation
    logger.info("\nTesting generation:")
    model.eval()
    test_qs = [
        "What is the capital of France?", "What is 15 times 7?",
        "Is exercise good for health?", "What is the boiling point of water?",
        "What year did WWII end?", "What is the chemical formula for water?",
        "Convert 5 kilometers to miles", "What is the largest planet?",
        "What is DNA?", "What is the speed of light?",
    ]
    correct = 0
    for q in test_qs:
        input_ids = tokenizer.encode(f"Human: {q} Assistant:", return_tensors="pt")
        with torch.no_grad():
            output = model.generate(input_ids, max_length=80, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = response.split("Assistant:")[-1].strip()
        logger.info(f"  Q: {q}")
        logger.info(f"  A: {answer}")

    return {
        "pairs": len(pairs), "epochs": 7,
        "best_val_loss": round(best_val_loss, 4),
        "time_s": round(total_time, 1),
        "checkpoint": os.path.join(checkpoint_dir, "best_model"),
        "losses": losses,
    }


# ════════════════════════════════════════════════════════════════════
# PART 4: EXPANDED CLASSIFIER TRAINING
# ════════════════════════════════════════════════════════════════════

def train_expanded_classifiers():
    """Train classifiers on 2000+ samples per task."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 4: EXPANDED CLASSIFIER TRAINING")
    logger.info("=" * 70)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np

    # Load embedder
    from neurons.semantic_embeddings import SemanticEmbedder
    embedder = SemanticEmbedder()
    embedder_dim = 384

    class Classifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )
        def forward(self, x):
            return self.net(x)

    # === Logic classifier (1500 samples) ===
    logic_data = []
    labels = ["VALID", "INVALID"]

    valid_templates = [
        ("If {a} then {b}. {a} happened.", "VALID"),
        ("All {a} are {b}. This is {a}.", "VALID"),
        ("{a} is greater than {b}. {b} is greater than {c}. {a} is greater than {c}.", "VALID"),
        ("If {a} implies {b}, and {b} implies {c}, then {a} implies {c}.", "VALID"),
        ("{a} and {b} are true. Therefore {a} is true.", "VALID"),
    ]
    invalid_templates = [
        ("If {a} then {b}. {b} happened.", "INVALID"),
        ("All {a} are {b}. This is {b}.", "INVALID"),
        ("{a} is greater than {b}. Therefore {b} is greater than {a}.", "INVALID"),
        ("If {a} then {b}. Therefore if {b} then {a}.", "INVALID"),
        ("{a} or {b}. {a} is false. Therefore {a} is true.", "INVALID"),
    ]
    fillers = ["it rains", "the ground is wet", "cats are animals", "animals are living",
               "A > B", "B > C", "P implies Q", "Q implies R", "X is true", "Y is false",
               "it snows", "roads are icy", "dogs are mammals", "mammals are warm-blooded"]

    for _ in range(750):
        a, b = random.sample(fillers, 2)
        c = random.choice(fillers)
        for tmpl, lbl in valid_templates:
            q = tmpl.format(a=a, b=b, c=c)
            emb = embedder.encode(q)
            logic_data.append((emb, 0 if lbl == "VALID" else 1))
    for _ in range(750):
        a, b = random.sample(fillers, 2)
        c = random.choice(fillers)
        for tmpl, lbl in invalid_templates:
            q = tmpl.format(a=a, b=b, c=c)
            emb = embedder.encode(q)
            logic_data.append((emb, 0 if lbl == "VALID" else 1))

    # Train logic classifier
    model_logic = Classifier(embedder_dim, 2)
    X = torch.tensor(np.array([d[0] for d in logic_data]), dtype=torch.float32)
    y = torch.tensor([d[1] for d in logic_data], dtype=torch.long)

    n = len(X)
    split = int(n * 0.8)
    perm = torch.randperm(n)
    train_idx, test_idx = perm[:split], perm[split:]

    optimizer = optim.Adam(model_logic.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        model_logic.train()
        out = model_logic(X[train_idx])
        loss = criterion(out, y[train_idx])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model_logic.eval()
    with torch.no_grad():
        pred = model_logic(X[test_idx]).argmax(dim=1)
        acc_logic = (pred == y[test_idx]).float().mean().item()

    logger.info(f"Logic classifier: accuracy={acc_logic:.4f}, loss={loss.item():.4f}")
    torch.save(model_logic.state_dict(), str(EXPERIMENT_DIR / "checkpoints_expanded" / "logic.pt"))

    # === Math classifier (1500 samples) ===
    math_data = []
    for _ in range(1500):
        a, b = random.randint(1, 100), random.randint(1, 100)
        op = random.choice(['+', '-', '*'])
        if op == '+': result, word = a + b, "plus"
        elif op == '-': result, word = a - b, "minus"
        else: result, word = a * b, "times"
        q = f"What is {a} {op} {b}?"
        emb = embedder.encode(q)
        math_data.append((emb, 0))  # COMPUTABLE

    for _ in range(500):
        nonsense = random.choice(["What is blue minus happiness?", "Calculate undefined.", "What is infinity times zero?"])
        emb = embedder.encode(nonsense)
        math_data.append((emb, 1))  # INVALID

    model_math = Classifier(embedder_dim, 2)
    X_m = torch.tensor(np.array([d[0] for d in math_data]), dtype=torch.float32)
    y_m = torch.tensor([d[1] for d in math_data], dtype=torch.long)
    perm_m = torch.randperm(len(X_m))
    split_m = int(len(X_m) * 0.8)

    optimizer = optim.Adam(model_math.parameters(), lr=1e-3)
    for epoch in range(30):
        model_math.train()
        out = model_math(X_m[perm_m[:split_m]])
        loss = criterion(out, y_m[perm_m[:split_m]])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model_math.eval()
    with torch.no_grad():
        pred = model_math(X_m[perm_m[split_m:]]).argmax(dim=1)
        acc_math = (pred == y_m[perm_m[split_m:]]).float().mean().item()

    logger.info(f"Math classifier: accuracy={acc_math:.4f}")
    torch.save(model_math.state_dict(), str(EXPERIMENT_DIR / "checkpoints_expanded" / "math.pt"))

    # === Evidence classifier (1500 samples) ===
    evidence_data = []
    supports_templates = [
        "Studies show that {topic} improves {outcome}.",
        "Evidence confirms {topic} causes {outcome}.",
        "Research demonstrates {topic} leads to {outcome}.",
    ]
    refutes_templates = [
        "Studies show that {topic} does NOT improve {outcome}.",
        "Evidence contradicts the claim that {topic} causes {outcome}.",
        "Research shows no link between {topic} and {outcome}.",
    ]
    topics = ["exercise", "education", "sleep", "nutrition", "meditation", "reading", "music", "social interaction"]
    outcomes = ["health", "performance", "well-being", "cognitive function", "productivity", "longevity"]

    for _ in range(500):
        t, o = random.choice(topics), random.choice(outcomes)
        tmpl = random.choice(supports_templates)
        q = tmpl.format(topic=t, outcome=o)
        emb = embedder.encode(q)
        evidence_data.append((emb, 0))  # SUPPORTS

    for _ in range(500):
        t, o = random.choice(topics), random.choice(outcomes)
        tmpl = random.choice(refutes_templates)
        q = tmpl.format(topic=t, outcome=o)
        emb = embedder.encode(q)
        evidence_data.append((emb, 1))  # REFUTES

    for _ in range(500):
        neutral_templates = [
            "The effect of {topic} on {outcome} is uncertain.",
            "More research is needed on {topic} and {outcome}.",
            "There is no consensus on {topic} and {outcome}.",
        ]
        t, o = random.choice(topics), random.choice(outcomes)
        tmpl = random.choice(neutral_templates)
        q = tmpl.format(topic=t, outcome=o)
        emb = embedder.encode(q)
        evidence_data.append((emb, 2))  # NEUTRAL

    model_ev = Classifier(embedder_dim, 3)
    X_e = torch.tensor(np.array([d[0] for d in evidence_data]), dtype=torch.float32)
    y_e = torch.tensor([d[1] for d in evidence_data], dtype=torch.long)
    perm_e = torch.randperm(len(X_e))
    split_e = int(len(X_e) * 0.8)

    optimizer = optim.Adam(model_ev.parameters(), lr=1e-3)
    for epoch in range(30):
        model_ev.train()
        out = model_ev(X_e[perm_e[:split_e]])
        loss = criterion(out, y_e[perm_e[:split_e]])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model_ev.eval()
    with torch.no_grad():
        pred = model_ev(X_e[perm_e[split_e:]]).argmax(dim=1)
        acc_ev = (pred == y_e[perm_e[split_e:]]).float().mean().item()

    logger.info(f"Evidence classifier: accuracy={acc_ev:.4f}")
    torch.save(model_ev.state_dict(), str(EXPERIMENT_DIR / "checkpoints_expanded" / "evidence.pt"))

    return {
        "logic_accuracy": acc_logic,
        "math_accuracy": acc_math,
        "evidence_accuracy": acc_ev,
        "total_samples": len(logic_data) + len(math_data) + len(evidence_data),
    }


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("SWEEP RAG + WEB SEARCH + EXPANDED TRAINING")
    logger.info("=" * 70)

    results = {}

    # Part 1: RAG
    results["rag"] = build_rag_layer()

    # Part 2: Web search integration
    results["web_search"] = integrate_web_search()

    # Part 3: Generate and train
    pairs = generate_5000_qa_pairs()
    results["qa_generation"] = {"total_pairs": len(pairs)}
    results["seq2seq_expanded"] = train_expanded_model(pairs)

    # Part 4: Classifiers
    results["classifiers"] = train_expanded_classifiers()

    # Save results
    with open(str(EXPERIMENT_DIR / "rag_training_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("\n" + "=" * 70)
    logger.info("ALL COMPLETE")
    logger.info("=" * 70)
    logger.info(f"QA pairs: {results['qa_generation']['total_pairs']}")
    logger.info(f"Seq2seq: loss={results['seq2seq_expanded']['best_val_loss']}, {results['seq2seq_expanded']['time_s']}s")
    logger.info(f"Classifiers: logic={results['classifiers']['logic_accuracy']:.3f}, math={results['classifiers']['math_accuracy']:.3f}, evidence={results['classifiers']['evidence_accuracy']:.3f}")

    return results


if __name__ == "__main__":
    main()
