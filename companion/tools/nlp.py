"""NLP capability — spaCy / NLTK (with builtin fallbacks).

Tokenization, sentence splitting, word frequency, entities (when a spaCy
model is installed) and POS tagging (spaCy model, else NLTK, else builtin
heuristics). Lazy imports and graceful degradation when model data is absent.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .common import as_text


def run_nlp(payload: dict[str, Any]) -> dict[str, Any]:
    """Analyze a text sample."""
    params = payload.get("params") or {}
    text = as_text(payload.get("data"), params).strip()
    if not text:
        raise ValueError(
            "No text found. Send `data` as a string (or params.text)."
        )

    libraries_used: list[str] = []
    result: dict[str, Any] = {"text_length": len(text), "characters": len(text)}

    # ── spaCy path (best quality when a model is installed) ─────────────
    spacy_nlp = _spacy_pipeline()
    if spacy_nlp is not None:
        libraries_used.append("spacy")
        doc = spacy_nlp(text[:200_000])
        tokens = [t.text for t in doc if not t.is_space]
        sentences = [str(s) for s in doc.sents]
        result["tokens"] = len(tokens)
        result["unique_tokens"] = len(set(tokens))
        result["sentences"] = len(sentences)
        result["entities"] = [
            {"text": ent.text, "label": ent.label_} for ent in doc.ents[:50]
        ]
        result["pos_tags"] = [
            {"token": t.text, "pos": t.pos_} for t in doc[:200] if not t.is_space
        ]
        freq = Counter(t.lower() for t in tokens if t.strip() and not _is_stop(t))
    else:
        # ── builtin tokenizer (works with zero model data) ───────────────
        tokens = _tokenize(text)
        sentences = _split_sentences(text)
        result["tokens"] = len(tokens)
        result["unique_tokens"] = len(set(tokens))
        result["sentences"] = len(sentences)
        freq = Counter(t for t in tokens if t and not _is_stop(t))
        # NLTK POS when installed (best-effort; needs punkt+averaged_perceptron_tagger).
        nltk_pos = _nltk_pos_tags(tokens)
        if nltk_pos is not None:
            libraries_used.append("nltk")
            result["pos_tags"] = [{"token": t, "pos": p} for t, p in nltk_pos[:200]]

    result["top_keywords"] = [
        {"word": word, "count": int(count)}
        for word, count in freq.most_common(15)
    ]
    if libraries_used:
        engine = libraries_used[0]
    elif spacy_nlp is not None:
        engine = "spacy"
    else:
        engine = "builtin"
    result["engine"] = engine

    summary = (
        f"Text: {result['tokens']} tokens, {result['sentences']} sentences, "
        f"{result['unique_tokens']} unique words (engine: {engine}). "
    )
    if result["top_keywords"]:
        top = ", ".join(k["word"] for k in result["top_keywords"][:5])
        summary += f"Top keywords: {top}."
    if result.get("entities"):
        summary += f" Entities: {', '.join(e['text'] for e in result['entities'][:5])}."

    return {"result": result, "summary": summary, "libraries_used": libraries_used}


def _spacy_pipeline():
    try:
        from .common import module_available

        if not module_available("spacy"):
            return None
        import spacy

        for name in ("en_core_web_sm", "en_core_web_md", "xx_ent_wiki_sm"):
            try:
                return spacy.load(name)
            except Exception:  # noqa: BLE001 - model not downloaded
                continue
    except Exception:  # noqa: BLE001 - spaCy unavailable
        pass
    return None


def _nltk_pos_tags(tokens: list[str]):
    try:
        from .common import module_available

        if not module_available("nltk"):
            return None
        import nltk

        for resource in ("punkt", "averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
            try:
                nltk.download(resource, quiet=True)
            except Exception:  # noqa: BLE001 - offline
                pass
        from nltk import pos_tag

        return pos_tag(tokens[:2000])
    except Exception:  # noqa: BLE001 - tagging is best-effort
        return None


_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on",
    "for", "with", "at", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "it", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "me", "my", "your", "his", "her", "our", "their", "do", "does",
    "did", "have", "has", "had", "will", "would", "can", "could", "should",
}


def _is_stop(token: str) -> bool:
    return token.lower() in _STOP_WORDS


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
