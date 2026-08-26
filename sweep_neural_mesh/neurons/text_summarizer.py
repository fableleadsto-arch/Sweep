"""
Text Summarizer — extractive summarization with TF-IDF scoring.

Architecture:
    Long Text Input
        ↓
    Split into sentences
        ↓
    TF-IDF scoring (sentence importance)
        ↓
    Top-K sentence selection (order preserved)
        ↓
    Summary text

Pure sklearn, no large model downloads needed.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    original_text: str
    summary: str
    sentence_count: int
    original_length: int
    summary_length: int
    compression_ratio: float
    key_sentences: list[str]
    latency_ms: float = 0.0


class TextSummarizer:
    def __init__(self, max_sentences: int = 3):
        self._max_sentences = max_sentences

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _score_sentences(self, sentences: list[str]) -> list[float]:
        if len(sentences) <= 2:
            return [1.0] * len(sentences)

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np

            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(sentences)

            scores = []
            for i in range(tfidf_matrix.shape[0]):
                row = tfidf_matrix[i].toarray().flatten()
                score = float(np.sum(row))
                # Position bonus: first and last sentences slightly boosted
                if i == 0:
                    score *= 1.2
                elif i == len(sentences) - 1:
                    score *= 1.1
                scores.append(score)
            return scores
        except Exception as e:
            logger.debug(f"TF-IDF scoring failed: {e}, using position-based fallback")
            return [1.0 / (i + 1) for i in range(len(sentences))]

    def summarize(self, text: str, max_sentences: int | None = None) -> SummaryResult:
        t0 = time.perf_counter()
        max_sent = max_sentences or self._max_sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return SummaryResult(
                original_text=text, summary=text,
                sentence_count=0, original_length=len(text),
                summary_length=len(text), compression_ratio=1.0,
                key_sentences=[], latency_ms=(time.perf_counter() - t0) * 1000,
            )

        if len(sentences) <= max_sent:
            return SummaryResult(
                original_text=text, summary=text,
                sentence_count=len(sentences), original_length=len(text),
                summary_length=len(text), compression_ratio=1.0,
                key_sentences=sentences, latency_ms=(time.perf_counter() - t0) * 1000,
            )

        scores = self._score_sentences(sentences)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        top_indices = sorted([idx for idx, _ in indexed_scores[:max_sent]])
        selected = [sentences[i] for i in top_indices]
        summary = " ".join(selected)

        return SummaryResult(
            original_text=text,
            summary=summary,
            sentence_count=len(selected),
            original_length=len(text),
            summary_length=len(summary),
            compression_ratio=len(summary) / max(1, len(text)),
            key_sentences=selected,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    def summarize_by_compression(self, text: str, target_ratio: float = 0.3) -> SummaryResult:
        sentences = self._split_sentences(text)
        max_sent = max(1, int(len(sentences) * target_ratio))
        return self.summarize(text, max_sentences=max_sent)

    def extract_keywords(self, text: str, top_k: int = 10) -> list[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return text.split()[:top_k]

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500, ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            mean_tfidf = tfidf_matrix.mean(axis=0).A1
            top_indices = mean_tfidf.argsort()[-top_k:][::-1]
            return [feature_names[i] for i in top_indices if mean_tfidf[i] > 0]
        except Exception:
            return []


_default_summarizer: TextSummarizer | None = None


def get_summarizer() -> TextSummarizer:
    global _default_summarizer
    if _default_summarizer is None:
        _default_summarizer = TextSummarizer()
    return _default_summarizer
