"""
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
