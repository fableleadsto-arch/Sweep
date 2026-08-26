"""
Semantic Embeddings — deep semantic understanding via pre-trained models.

Falls back to SimHash when ML models are unavailable.

Architecture:
    Text Input
        ↓
    [Primary: sentence-transformers/all-MiniLM-L6-v2]
    [Fallback 1: minishlab/potion-base-32M via model2vec]
    [Fallback 2: SimHash from embeddings.py]
        ↓
    384-dim / 256-dim / 128-bit embedding
        ↓
    Cosine similarity (0.0–1.0)

All models lazy-loaded on first use.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_EMBEDDING_BACKEND = None  # 'minilm', 'potion', 'simhash', None (untried)


def _try_load_backend():
    global _EMBEDDING_BACKEND
    if _EMBEDDING_BACKEND is not None:
        return _EMBEDDING_BACKEND

    import os
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # Try minishlab/potion-base-32M (smallest, fastest)
    try:
        from model2vec import StaticModel
        model = StaticModel.from_pretrained("minishlab/potion-base-32M")
        _EMBEDDING_BACKEND = "potion"
        logger.info("Using minishlab/potion-base-32M for embeddings")
        return _EMBEDDING_BACKEND
    except Exception as e:
        logger.debug(f"potion unavailable: {e}")

    # Try sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        _EMBEDDING_BACKEND = "minilm"
        logger.info("Using all-MiniLM-L6-v2 for embeddings")
        return _EMBEDDING_BACKEND
    except Exception as e:
        logger.debug(f"minilm unavailable: {e}")

    _EMBEDDING_BACKEND = "simhash"
    logger.info("Falling back to SimHash embeddings")
    return _EMBEDDING_BACKEND


_potion_model = None
_minilm_model = None
_simhash_engine = None


def _get_potion():
    global _potion_model
    if _potion_model is None:
        from model2vec import StaticModel
        _potion_model = StaticModel.from_pretrained("minishlab/potion-base-32M")
    return _potion_model


def _get_minilm():
    global _minilm_model
    if _minilm_model is None:
        from sentence_transformers import SentenceTransformer
        _minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _minilm_model


def _get_simhash():
    global _simhash_engine
    if _simhash_engine is None:
        from .embeddings import EmbeddingEngine
        _simhash_engine = EmbeddingEngine()
    return _simhash_engine


@dataclass
class EmbeddingResult:
    text: str
    vector: list[float] | None
    dim: int
    backend: str
    latency_ms: float = 0.0


@dataclass
class SimilarityResult:
    text1: str
    text2: str
    score: float
    backend: str


class SemanticEmbedder:
    def __init__(self, backend: str | None = None):
        self._forced_backend = backend
        self._backend = None
        self._embedding_dim = 0

    def _ensure_backend(self):
        if self._backend is not None:
            return
        if self._forced_backend:
            self._backend = self._forced_backend
        else:
            self._backend = _try_load_backend()
        if self._backend == "minilm":
            self._embedding_dim = 384
        elif self._backend == "potion":
            self._embedding_dim = 256
        else:
            self._embedding_dim = 0

    @property
    def backend(self) -> str:
        self._ensure_backend()
        return self._backend

    @property
    def embedding_dim(self) -> int:
        self._ensure_backend()
        return self._embedding_dim

    def embed(self, text: str) -> EmbeddingResult:
        t0 = time.perf_counter()
        self._ensure_backend()
        latency = (time.perf_counter() - t0) * 1000

        if self._backend == "minilm":
            try:
                model = _get_minilm()
                vec = model.encode(text).tolist()
                return EmbeddingResult(text=text, vector=vec, dim=len(vec),
                                       backend="minilm",
                                       latency_ms=(time.perf_counter() - t0) * 1000)
            except Exception as e:
                logger.warning(f"minilm embed failed: {e}, falling back to simhash")
                self._backend = "simhash"

        if self._backend == "potion":
            try:
                model = _get_potion()
                vec = model.encode(text).tolist()
                return EmbeddingResult(text=text, vector=vec, dim=len(vec),
                                       backend="potion",
                                       latency_ms=(time.perf_counter() - t0) * 1000)
            except Exception as e:
                logger.warning(f"potion embed failed: {e}, falling back to simhash")
                self._backend = "simhash"

        engine = _get_simhash()
        fp = engine.fingerprint(text)
        bits = fp.bits
        vector = [(bits >> i) & 1 for i in range(128)]
        return EmbeddingResult(text=text, vector=vector, dim=128,
                               backend="simhash",
                               latency_ms=(time.perf_counter() - t0) * 1000)

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.embed(t) for t in texts]

    def similarity(self, text1: str, text2: str) -> SimilarityResult:
        self._ensure_backend()
        if self._backend in ("minilm", "potion"):
            r1 = self.embed(text1)
            r2 = self.embed(text2)
            if r1.vector and r2.vector:
                dot = sum(a * b for a, b in zip(r1.vector, r2.vector))
                n1 = math.sqrt(sum(a * a for a in r1.vector))
                n2 = math.sqrt(sum(b * b for b in r2.vector))
                score = dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0
                return SimilarityResult(text1=text1, text2=text2, score=score,
                                        backend=self._backend)

        engine = _get_simhash()
        fp1 = engine.fingerprint(text1)
        fp2 = engine.fingerprint(text2)
        score = engine.similarity(fp1, fp2)
        return SimilarityResult(text1=text1, text2=text2, score=score,
                                backend="simhash")

    def most_similar(self, query: str, candidates: list[str],
                     top_k: int = 5) -> list[tuple[int, float, str]]:
        results = []
        for idx, cand in enumerate(candidates):
            sim = self.similarity(query, cand)
            results.append((idx, sim.score, cand))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


_default_embedder: SemanticEmbedder | None = None


def get_embedder() -> SemanticEmbedder:
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = SemanticEmbedder()
    return _default_embedder
