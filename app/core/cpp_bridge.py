"""Bridge to the C++ performance engine.

Falls back to pure Python implementations when the C++ module
is not compiled. Exposes ML, data, NLP, and search engines.
"""

from __future__ import annotations

from typing import Optional, Any

# Try to import the C++ engine
try:
    import sweep_engine as _cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False


# ── HTML / Text ───────────────────────────────────────────────────────

def html_to_text(html: str) -> str:
    if HAS_CPP:
        return _cpp.html_to_text(html)
    import re
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def html_to_markdown(html: str, url: str, max_chars: int = 12000):
    if HAS_CPP:
        return _cpp.html_to_markdown(html, url, max_chars)
    from ..scraping.markdown import html_to_markdown as py_html_to_markdown
    return py_html_to_markdown(html, url, max_chars)


# ── Search / Ranking ──────────────────────────────────────────────────

def score_relevance(text: str, keywords: list[str]) -> float:
    if HAS_CPP:
        return _cpp.score_relevance(text, keywords)
    return sum(1 for kw in keywords if kw in text.lower()) / max(len(keywords) * 0.3, 1.0)


def extract_top_sentences(text: str, keywords: list[str], max_sentences: int = 8) -> list[str]:
    if HAS_CPP:
        return _cpp.extract_top_sentences(text, keywords, max_sentences)
    import re
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    scored = []
    for i, s in enumerate(sentences):
        if len(s.strip()) < 40:
            continue
        score = sum(1 for kw in keywords if kw in s.lower())
        if re.search(r"\d", s):
            score += 0.5
        if len(s) >= 120:
            score += 0.25
        if i < 4:
            score += 0.25
        scored.append((s.strip(), score))
    scored.sort(key=lambda x: -x[1])
    return [s for s, _ in scored[:max_sentences]]


def detect_injection(text: str) -> tuple[bool, list[str]]:
    if HAS_CPP:
        signals: list[str] = []
        suspect = _cpp.detect_injection(text, signals)
        return suspect, signals
    from ..core.guard import assess_injection
    result = assess_injection(text)
    return result.suspect, result.signals


def rank_hits(query: str, hits: list[dict], limit: int = 10) -> list[dict]:
    if HAS_CPP:
        cpp_hits = []
        for h in hits:
            rh = _cpp.RankedHit()
            rh.url = h.get("url", "")
            rh.title = h.get("title", "")
            rh.snippet = h.get("snippet", "")
            rh.engine = h.get("engine", "")
            cpp_hits.append(rh)
        ranked = _cpp.rank_hits(query, cpp_hits, limit)
        return [
            {"url": r.url, "title": r.title, "snippet": r.snippet, "engine": r.engine, "score": r.score}
            for r in ranked
        ]
    import re
    result = []
    for h in hits:
        text = f"{h.get('title', '')} {h.get('snippet', '')}".lower()
        score = sum(1 for word in query.lower().split() if word in text)
        result.append({**h, "score": score})
    result.sort(key=lambda x: -x.get("score", 0))
    return result[:limit]


def extract_heuristic(text: str) -> dict:
    if HAS_CPP:
        fields = _cpp.extract_heuristic(text)
        return {"emails": list(fields.emails), "phones": list(fields.phones),
                "social_urls": list(fields.social_urls), "fields_found": fields.fields_found}
    import re
    emails = list(set(re.findall(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", text.lower())))
    phones = list(set(re.findall(r"\+?\d[\d\s().-]{7,17}\d", text)))
    return {"emails": emails[:20], "phones": phones[:10], "social_urls": [], "fields_found": len(emails) + len(phones)}


def looks_blocked(status: int, text: str) -> bool:
    if HAS_CPP:
        return _cpp.PatternSet().looks_blocked(status, text)
    if status in (403, 429, 503):
        return True
    head = text[:4000].lower()
    return any(m in head for m in ["captcha", "are you a robot", "access denied"])


# ── ML Engine (C++) ──────────────────────────────────────────────────

def compute_stats(data: list[float]) -> dict:
    if HAS_CPP:
        r = _cpp.compute_stats(data)
        return {"mean": r.mean, "median": r.median, "std_dev": r.std_dev,
                "variance": r.variance, "min": r.min_val, "max": r.max_val, "count": r.count}
    import statistics
    return {"mean": statistics.mean(data), "median": statistics.median(data),
            "std_dev": statistics.stdev(data) if len(data) > 1 else 0, "variance": statistics.variance(data) if len(data) > 1 else 0,
            "min": min(data), "max": max(data), "count": len(data)}


def matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if HAS_CPP:
        return _cpp.matrix_multiply(a, b)
    rows_a, cols_a = len(a), len(a[0])
    cols_b = len(b[0])
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result


def dot_product(a: list[float], b: list[float]) -> float:
    if HAS_CPP:
        return _cpp.dot_product(a, b)
    return sum(x * y for x, y in zip(a, b))


def correlation(x: list[float], y: list[float]) -> float:
    if HAS_CPP:
        return _cpp.correlation(x, y)
    import statistics
    return statistics.correlation(x, y) if len(x) == len(y) and len(x) > 1 else 0.0


# ── Data Engine (C++) ────────────────────────────────────────────────

def parse_csv(csv_text: str, has_header: bool = True):
    if HAS_CPP:
        return _cpp.parse_csv(csv_text, has_header)
    import csv, io
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        return {"columns": [], "data": []}
    columns = rows[0] if has_header else [f"col_{i}" for i in range(len(rows[0]))]
    data = rows[1:] if has_header else rows
    return {"columns": columns, "data": data}


def df_describe(columns: list[str], data: list[list]) -> list[dict]:
    if HAS_CPP:
        df = _cpp.DataFrame()
        df.columns = columns
        df.rows = data
        stats = _cpp.df_describe(df)
        return [{"name": s.name, "count": s.count, "mean": s.mean, "std": s.std_dev,
                 "min": s.min_val, "max": s.max_val, "median": s.median} for s in stats]
    return []


# ── NLP Engine (C++) ─────────────────────────────────────────────────

def tokenize_words(text: str) -> list[str]:
    if HAS_CPP:
        return _cpp.tokenize_words(text)
    return text.split()


def tokenize_sentences(text: str) -> list[str]:
    if HAS_CPP:
        return _cpp.tokenize_sentences(text)
    import re
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


def extract_entities(text: str) -> list[dict]:
    if HAS_CPP:
        entities = _cpp.cpp_extract_entities(text)
        return [{"text": e.text, "label": e.label, "confidence": e.confidence} for e in entities]
    return []


def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    if HAS_CPP:
        return _cpp.extract_keywords(text, top_n)
    from collections import Counter
    words = [w.lower() for w in text.split() if len(w) > 3]
    return [w for w, _ in Counter(words).most_common(top_n)]


def text_similarity(a: str, b: str) -> float:
    if HAS_CPP:
        return _cpp.text_similarity(a, b)
    set_a, set_b = set(a.lower().split()), set(b.lower().split())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def sentiment(text: str) -> str:
    if HAS_CPP:
        return _cpp.sentiment(text)
    positive = {"good", "great", "excellent", "amazing", "wonderful", "love", "happy", "best"}
    negative = {"bad", "terrible", "awful", "horrible", "worst", "hate", "poor", "fail"}
    words = set(text.lower().split())
    pos = len(words & positive)
    neg = len(words & negative)
    if pos > neg + 2:
        return "positive"
    if neg > pos + 2:
        return "negative"
    return "neutral"


def compute_tfidf(documents: list[str]) -> dict:
    if HAS_CPP:
        result = _cpp.compute_tfidf(documents)
        return {"vocabulary": result.vocabulary, "matrix": result.matrix}
    return {"vocabulary": [], "matrix": []}


# ── Status ────────────────────────────────────────────────────────────

def cpp_available() -> bool:
    return HAS_CPP


def cpp_modules() -> list[str]:
    if not HAS_CPP:
        return []
    return ["html_parser", "text_extractor", "search_ranker", "regex_engine",
            "ml_engine", "data_engine", "nlp_engine"]
