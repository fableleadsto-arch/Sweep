"""
Lightweight Embeddings — SimHash + cosine similarity for fast semantic comparison.

No external ML libraries needed. Uses SimHash ( locality-sensitive hashing )
to produce fixed-size fingerprints from text, with cosine similarity for
distance measurement. This gives "good enough" semantic similarity
without requiring GPU or large model downloads.

Architecture:

    Text Input
        ↓
    Tokenize + Weight (TF-IDF-lite)
        ↓
    SimHash Fingerprint (128-bit)
        ↓
    Cosine Similarity (bit-level)
        ↓
    Similarity Score (0.0–1.0)

Properties:
    - O(n) per document, O(1) comparison
    - Memory: 16 bytes per fingerprint
    - Quality: ~85% agreement with cosine on real text (per SimHash papers)
    - Domain-agnostic: works on any text without training
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


# Common English stop words (top 50)
_STOP_WORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "about", "against",
    "up", "down", "it", "its", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose",
}


@dataclass
class SimHashFingerprint:
    """A compact text fingerprint for fast similarity comparison."""
    bits: int                 # 128-bit fingerprint stored as Python int
    text_length: int          # number of tokens used to generate this
    timestamp: float = field(default_factory=time.time)

    def hamming_distance(self, other: SimHashFingerprint) -> int:
        """Count differing bits between two fingerprints."""
        return bin(self.bits ^ other.bits).count("1")

    def cosine_similarity(self, other: SimHashFingerprint) -> float:
        """
        Approximate cosine similarity from bit-level comparison.

        Uses the relationship: cos_sim ≈ 1 - (hamming_dist / total_bits)
        This is an approximation that works well for 128-bit fingerprints.
        """
        distance = self.hamming_distance(other)
        return 1.0 - (distance / 128.0)


class EmbeddingEngine:
    """
    Lightweight embedding engine using SimHash.

    Produces fixed-size 128-bit fingerprints from text, enabling
    fast semantic similarity comparison without ML models.

    Usage:
        engine = EmbeddingEngine()
        fp1 = engine.fingerprint("Python is a great language")
        fp2 = engine.fingerprint("Python is an excellent language")
        sim = engine.similarity(fp1, fp2)  # ~0.85

    Properties:
        - Constant memory per document (16 bytes)
        - Constant-time comparison
        - No model loading, no GPU needed
        - Works on any text (code, natural language, mixed)
    """

    def __init__(self, hash_bits: int = 128) -> None:
        self._hash_bits = hash_bits
        self._shingle_size = 3  # character shingle size for SimHash
        self._cache: dict[str, SimHashFingerprint] = {}
        self._cache_max = 10000

    def fingerprint(self, text: str) -> SimHashFingerprint:
        """
        Generate a SimHash fingerprint from text.

        Algorithm:
        1. Tokenize and weight tokens (TF-IDF-lite)
        2. Generate shingles (character n-grams)
        3. Hash each shingle
        4. Weight and accumulate into bit vector
        5. Threshold into binary fingerprint
        """
        if not text:
            return SimHashFingerprint(bits=0, text_length=0)

        # Check cache
        cache_key = text[:500]
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Tokenize
        tokens = self._tokenize(text)
        if not tokens:
            return SimHashFingerprint(bits=0, text_length=0)

        # Compute token weights (TF-IDF-lite)
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        weights: dict[str, float] = {}
        for token, count in token_counts.items():
            tf = count / total_tokens
            # Rare tokens get higher weight (inverse document frequency approximation)
            idf_boost = 1.0 + math.log(1.0 + 1.0 / max(1, count))
            weights[token] = tf * idf_boost

        # Generate shingles from the full text
        shingles = self._generate_shingles(text.lower())

        # Initialize bit vector
        bit_vector = [0.0] * self._hash_bits

        # For each shingle, hash and accumulate weighted votes
        for shingle in shingles:
            shingle_hash = self._hash_shingle(shingle)
            # Get weight from tokens in this shingle
            shingle_tokens = shingle.split()
            weight = sum(weights.get(t, 0.5) for t in shingle_tokens) / max(1, len(shingle_tokens))

            # Accumulate into bit vector
            for i in range(self._hash_bits):
                if shingle_hash & (1 << i):
                    bit_vector[i] += weight
                else:
                    bit_vector[i] -= weight

        # Threshold into binary fingerprint
        bits = 0
        for i in range(self._hash_bits):
            if bit_vector[i] > 0:
                bits |= (1 << i)

        fingerprint = SimHashFingerprint(bits=bits, text_length=total_tokens)

        # Cache
        if len(self._cache) < self._cache_max:
            self._cache[cache_key] = fingerprint

        return fingerprint

    def similarity(self, fp1: SimHashFingerprint, fp2: SimHashFingerprint) -> float:
        """Compute similarity between two fingerprints (0.0–1.0)."""
        return fp1.cosine_similarity(fp2)

    def text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two text strings directly."""
        fp1 = self.fingerprint(text1)
        fp2 = self.fingerprint(text2)
        return self.similarity(fp1, fp2)

    def batch_fingerprint(self, texts: list[str]) -> list[SimHashFingerprint]:
        """Generate fingerprints for multiple texts."""
        return [self.fingerprint(t) for t in texts]

    def most_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float, str]]:
        """
        Find the most similar texts to a query.

        Returns list of (index, similarity, text) sorted by similarity.
        """
        query_fp = self.fingerprint(query)
        results: list[tuple[int, float, str]] = []

        for idx, text in enumerate(candidates):
            cand_fp = self.fingerprint(text)
            sim = self.similarity(query_fp, cand_fp)
            results.append((idx, sim, text))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase words, removing stop words."""
        words = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return [w for w in words if w not in _STOP_WORDS and len(w) > 1]

    def _generate_shingles(self, text: str) -> list[str]:
        """Generate character-level shingles from text."""
        # Use word-level trigrams instead of character shingles for better semantics
        words = re.findall(r'\b[a-z0-9]+\b', text)
        shingles: list[str] = []
        for i in range(len(words) - self._shingle_size + 1):
            shingle = " ".join(words[i:i + self._shingle_size])
            shingles.append(shingle)
        # Also add single words as unigram shingles
        shingles.extend(words)
        return shingles if shingles else words

    def _hash_shingle(self, shingle: str) -> int:
        """Hash a shingle to a 128-bit integer."""
        h = hashlib.md5(shingle.encode("utf-8")).digest()
        return int.from_bytes(h, byteorder="big")


# Global singleton for convenience
_default_engine: EmbeddingEngine | None = None


def get_embedding_engine() -> EmbeddingEngine:
    """Get or create the global embedding engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = EmbeddingEngine()
    return _default_engine


def text_similarity(text1: str, text2: str) -> float:
    """Quick similarity check between two texts."""
    return get_embedding_engine().text_similarity(text1, text2)
