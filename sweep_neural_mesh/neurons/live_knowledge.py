"""
Live Knowledge Retrieval Module — fetches real-time knowledge from APIs.

Sources:
- Wikipedia API (free, no key required) — encyclopedic knowledge
- Wikidata API (free, no key required) — structured entity data
- Open-Meteo API (free, no key required) — weather data
- REST Countries API (free, no key required) — country data
- Numbers API (free, no key required) — mathematical facts

Features:
- Automatic fallback: static knowledge → live retrieval → abstention
- LRU cache to avoid redundant API calls
- Rate limiting to respect API limits
- Graceful degradation when APIs are unavailable
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any
from functools import lru_cache

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class RetrievalResult:
    """Result from a live knowledge retrieval."""
    source: str
    query: str
    answer: str
    confidence: float
    raw_data: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""


class LiveKnowledgeRetriever:
    """
    Fetches real-time knowledge from multiple APIs.

    Usage:
        retriever = LiveKnowledgeRetriever()
        result = retriever.retrieve("What is the capital of France?")
        if result.success:
            print(result.answer)  # "Paris"
    """

    def __init__(self, cache_size: int = 2000, timeout: float = 3.0) -> None:
        self._cache_size = cache_size
        self._timeout = timeout
        self._cache: dict[str, RetrievalResult] = {}
        self._cache_order: list[str] = []
        self._rate_limits: dict[str, float] = {}
        self._min_interval = 0.05  # 50ms between calls to same API
        self._stats = {"hits": 0, "misses": 0, "api_calls": 0}

        # Initialize HTTP client with proper User-Agent
        # Wikipedia requires a URL in User-Agent for automated requests
        self._headers = {
            "User-Agent": "SweepNeuralEngine/1.0 (https://github.com/sweep-ai/sweep; research@openresearch.org)",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }
        if HAS_HTTPX:
            # Use connection pooling with keep-alive for faster repeated requests
            self._client = httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                headers=self._headers,
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                    keepalive_expiry=30,
                ),
            )
        elif HAS_REQUESTS:
            self._client = None  # Will use requests directly
        else:
            self._client = None

    def _extract_topic(self, query: str) -> str:
        """Extract the main topic from a question for better search results.
        
        Smart extraction that identifies the entity/subject to search for.
        """
        import re
        q = query.strip().rstrip('?.!')
        
        # Pattern: "What is the X of Y?" -> "Y" (e.g., "capital of France" -> "France")
        # Pattern: "Who was X?" -> "X"
        # Pattern: "Where is X?" -> "X"
        # Pattern: "When was X built?" -> "X"
        # Pattern: "How does X work?" -> "X"
        # Pattern: "What is X?" -> "X"
        
        # Special case: "What is photosynthesis?" -> "photosynthesis"
        # Look for pattern: What is <lowercase_word>?
        what_is_match = re.search(r'(?:what|who|where|when|why|how|which)\s+(?:is|are|was|were|does|do|did|has|have|had)\s+([a-z][a-z\s]+)', q, re.IGNORECASE)
        if what_is_match:
            result = what_is_match.group(1).strip()
            # Don't return if it's still just a question word
            if result.lower() not in ('what', 'who', 'where', 'when', 'why', 'how', 'which'):
                return result
        
        # Special case: "Who invented X?" -> "X"
        who_match = re.search(r'(?:who|what)\s+(?:invented|created|discovered|found|made|built|wrote|composed)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', q, re.IGNORECASE)
        if who_match:
            return who_match.group(1)
        
        # Extract proper nouns / capitalized words as likely entities
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', q)
        
        # Also look for "X of Y" patterns - Y is likely the entity
        of_match = re.search(r'\bof\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', q)
        if of_match:
            return of_match.group(1)
        
        # Look for "X in Y" patterns
        in_match = re.search(r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', q)
        if in_match:
            return in_match.group(1)
        
        # Look for "X for Y" patterns  
        for_match = re.search(r'\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', q)
        if for_match:
            return for_match.group(1)
        
        # Use the last proper noun phrase (likely the main subject)
        if proper_nouns:
            # Filter out question words that might be capitalized
            filtered = [p for p in proper_nouns if p.lower() not in ('what', 'who', 'where', 'when', 'why', 'how', 'which', 'is', 'the')]
            if filtered:
                return max(filtered, key=len)
            return max(proper_nouns, key=len)
        
        # Fallback: remove question words and use remaining content
        q = re.sub(r'^(what|who|where|when|why|how|which|is|are|was|were|do|does|did|can|could|would|should|has|have|had|the|a|an)\s+', '', q, flags=re.IGNORECASE)
        q = re.sub(r'\s+', ' ', q).strip()
        
        if len(q) < 3:
            return query.strip()
        return q

    def retrieve(self, query: str) -> RetrievalResult | None:
        """
        Retrieve knowledge for a query from live APIs.

        Tries multiple sources in order of reliability:
        1. Wikipedia (encyclopedic knowledge)
        2. Wikidata (structured facts)
        3. Numbers API (mathematical facts)
        4. Open-Meteo (weather)
        5. REST Countries (country data)
        """
        # Check cache first
        cache_key = self._cache_key(query)
        if cache_key in self._cache:
            self._stats["hits"] += 1
            return self._cache[cache_key]
        self._stats["misses"] += 1

        # Extract topic from question for better search
        topic = self._extract_topic(query)

        # Try each source with both original query and extracted topic
        sources = [
            self._query_wikipedia,
            self._query_wikidata,
        ]

        for source_fn in sources:
            try:
                # Try extracted topic first (more specific)
                result = source_fn(topic)
                if result and result.success and result.answer:
                    self._cache_result(cache_key, result)
                    return result
                # Fall back to original query
                result = source_fn(query)
                if result and result.success and result.answer:
                    self._cache_result(cache_key, result)
                    return result
            except Exception:
                continue

        # Try math-specific sources
        if any(w in query.lower() for w in ['calculate', 'factorial', 'prime', 'sum', 'multiply']):
            try:
                result = self._query_numbers_api(query)
                if result and result.success and result.answer:
                    self._cache_result(cache_key, result)
                    return result
            except Exception:
                pass

        return None

    def retrieve_entity(self, entity: str) -> RetrievalResult | None:
        """Retrieve structured data about a specific entity."""
        cache_key = f"entity:{entity.lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try Wikipedia for entity info
        result = self._query_wikipedia(entity)
        if result and result.success:
            self._cache_result(cache_key, result)
            return result

        # Try Wikidata
        result = self._query_wikidata(entity)
        if result and result.success:
            self._cache_result(cache_key, result)
            return result

        return None

    def retrieve_fact(self, topic: str) -> RetrievalResult | None:
        """Retrieve a specific fact about a topic."""
        cache_key = f"fact:{topic.lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try Numbers API for math facts
        if any(w in topic.lower() for w in ["number", "math", "calculate", "factorial", "prime"]):
            result = self._query_numbers_api(topic)
            if result and result.success:
                self._cache_result(cache_key, result)
                return result

        # Try Wikipedia
        result = self._query_wikipedia(topic)
        if result and result.success:
            self._cache_result(cache_key, result)
            return result

        return None

    # ══════════════════════════════════════════════════════════════
    # API IMPLEMENTATIONS
    # ══════════════════════════════════════════════════════════════

    def _query_wikipedia(self, query: str) -> RetrievalResult | None:
        """Query Wikipedia API for encyclopedic knowledge.
        
        Optimized for speed: uses shorter extracts and skips fallback when possible.
        """
        if not self._rate_limit("wikipedia"):
            return None

        t0 = time.perf_counter()
        try:
            api_url = "https://en.wikipedia.org/w/api.php"
            
            # Strategy 1: Direct title lookup with SHORT extract (500 chars max)
            params = {
                "action": "query",
                "titles": query,
                "prop": "extracts",
                "exintro": "true",
                "explaintext": "true",
                "exchars": "500",  # Limit extract size for speed
                "format": "json",
                "exlimit": 1,
            }
            response = self._http_get(api_url, params=params)
            if response and response.status_code == 200:
                data = response.json()
                pages = data.get("query", {}).get("pages", {})
                for pid, page in pages.items():
                    if pid == "-1":
                        continue
                    extract = page.get("extract", "")
                    title = page.get("title", "")
                    if extract:
                        # Take first sentence as answer (faster than splitting)
                        dot_pos = extract.find('. ')
                        answer = extract[:dot_pos + 1] if dot_pos > 0 else extract[:200]
                        return RetrievalResult(
                            source="wikipedia",
                            query=query,
                            answer=answer,
                            confidence=0.90,
                            raw_data={"title": title, "extract": extract},
                            latency_ms=(time.perf_counter() - t0) * 1000,
                        )

            # Strategy 2: Search + extract in ONE request (combined query)
            # This avoids the second API call
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srinfo": "suggestion",
                "prop": "extracts",
                "exintro": "true",
                "explaintext": "true",
                "exchars": "500",
                "format": "json",
                "srlimit": 1,
                "exlimit": 1,
            }
            response = self._http_get(api_url, params=params)
            if response and response.status_code == 200:
                data = response.json()
                # Try to get extract directly from search results
                pages = data.get("query", {}).get("pages", {})
                for pid, page in pages.items():
                    extract = page.get("extract", "")
                    title = page.get("title", "")
                    if extract:
                        dot_pos = extract.find('. ')
                        answer = extract[:dot_pos + 1] if dot_pos > 0 else extract[:200]
                        return RetrievalResult(
                            source="wikipedia",
                            query=query,
                            answer=answer,
                            confidence=0.80,
                            raw_data={"title": title, "extract": extract},
                            latency_ms=(time.perf_counter() - t0) * 1000,
                        )
                # Fallback: search for title, then get extract
                results = data.get("query", {}).get("search", [])
                if results:
                    title = results[0].get("title", "")
                    params2 = {
                        "action": "query",
                        "titles": title,
                        "prop": "extracts",
                        "exintro": "true",
                        "explaintext": "true",
                        "exchars": "500",
                        "format": "json",
                        "exlimit": 1,
                    }
                    response2 = self._http_get(api_url, params=params2)
                    if response2 and response2.status_code == 200:
                        data2 = response2.json()
                        pages2 = data2.get("query", {}).get("pages", {})
                        for pid, page in pages2.items():
                            extract = page.get("extract", "")
                            if extract:
                                dot_pos = extract.find('. ')
                                answer = extract[:dot_pos + 1] if dot_pos > 0 else extract[:200]
                                return RetrievalResult(
                                    source="wikipedia",
                                    query=query,
                                    answer=answer,
                                    confidence=0.75,
                                    raw_data={"title": title, "extract": extract},
                                    latency_ms=(time.perf_counter() - t0) * 1000,
                                )

        except Exception as e:
            return RetrievalResult(
                source="wikipedia", query=query, answer="",
                confidence=0.0, success=False, error=str(e),
            )
        return None

    def _query_wikidata(self, query: str) -> RetrievalResult | None:
        """Query Wikidata API for structured entity data."""
        if not self._rate_limit("wikidata"):
            return None

        t0 = time.perf_counter()
        try:
            # Search for entity
            search_url = "https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "format": "json",
                "limit": 1,
            }
            response = self._http_get(search_url, params=params)
            if response and response.status_code == 200:
                data = response.json()
                results = data.get("search", [])
                if results:
                    entity_id = results[0].get("id", "")
                    label = results[0].get("label", "")
                    description = results[0].get("description", "")

                    # Get entity claims
                    entity_url = f"https://www.wikidata.org/w/api.php"
                    entity_params = {
                        "action": "wbgetentities",
                        "ids": entity_id,
                        "format": "json",
                        "props": "claims",
                    }
                    entity_response = self._http_get(entity_url, params=entity_params)

                    answer = f"{label}: {description}" if description else label
                    return RetrievalResult(
                        source="wikidata",
                        query=query,
                        answer=answer,
                        confidence=0.80,
                        raw_data={
                            "entity_id": entity_id,
                            "label": label,
                            "description": description,
                        },
                        latency_ms=(time.perf_counter() - t0) * 1000,
                    )

        except Exception as e:
            return RetrievalResult(
                source="wikidata", query=query, answer="",
                confidence=0.0, success=False, error=str(e),
            )
        return None

    def _query_numbers_api(self, query: str) -> RetrievalResult | None:
        """Query Numbers API for mathematical facts."""
        if not self._rate_limit("numbers"):
            return None

        t0 = time.perf_counter()
        try:
            # Extract number from query
            numbers = re.findall(r'\b(\d+)\b', query)
            if numbers:
                num = numbers[0]
                # Try different fact types
                for fact_type in ["trivia", "math", "date", "year"]:
                    url = f"http://numbersapi.com/{num}/{fact_type}?json"
                    response = self._http_get(url)
                    if response and response.status_code == 200:
                        data = response.json()
                        if data.get("found"):
                            return RetrievalResult(
                                source="numbers_api",
                                query=query,
                                answer=data.get("text", ""),
                                confidence=0.85,
                                raw_data=data,
                                latency_ms=(time.perf_counter() - t0) * 1000,
                            )

        except Exception as e:
            return RetrievalResult(
                source="numbers_api", query=query, answer="",
                confidence=0.0, success=False, error=str(e),
            )
        return None

    def _query_wolfram_alpha(self, query: str) -> RetrievalResult | None:
        """
        Query Wolfram Alpha for computational knowledge.
        Note: Requires WOLFRAM_APPID environment variable.
        """
        import os
        app_id = os.environ.get("WOLFRAM_APPID", "")
        if not app_id:
            return None

        if not self._rate_limit("wolfram"):
            return None

        t0 = time.perf_counter()
        try:
            url = "https://api.wolframalpha.com/v2/query"
            params = {
                "input": query,
                "appid": app_id,
                "output": "json",
                "format": "plaintext",
            }
            response = self._http_get(url, params=params)
            if response and response.status_code == 200:
                data = response.json()
                pods = data.get("queryresult", {}).get("pods", [])
                if pods:
                    # Get the first plaintext result
                    for pod in pods:
                        subpods = pod.get("subpods", [])
                        for subpod in subpods:
                            plaintext = subpod.get("plaintext", "")
                            if plaintext:
                                return RetrievalResult(
                                    source="wolfram_alpha",
                                    query=query,
                                    answer=plaintext,
                                    confidence=0.90,
                                    raw_data=data,
                                    latency_ms=(time.perf_counter() - t0) * 1000,
                                )

        except Exception as e:
            return RetrievalResult(
                source="wolfram_alpha", query=query, answer="",
                confidence=0.0, success=False, error=str(e),
            )
        return None

    # ══════════════════════════════════════════════════════════════
    # HTTP HELPERS
    # ══════════════════════════════════════════════════════════════

    def _http_get(self, url: str, params: dict | None = None) -> Any:
        """Make an HTTP GET request using the best available client."""
        self._stats["api_calls"] += 1
        if HAS_HTTPX and self._client:
            return self._client.get(url, params=params)
        elif HAS_REQUESTS:
            return requests.get(url, params=params, timeout=self._timeout, headers=self._headers)
        return None

    def _rate_limit(self, api_name: str) -> bool:
        """Check if we can make a request to this API (rate limiting)."""
        now = time.time()
        last_call = self._rate_limits.get(api_name, 0)
        if now - last_call < self._min_interval:
            return False
        self._rate_limits[api_name] = now
        return True

    def _cache_key(self, query: str) -> str:
        """Generate a normalized cache key for a query."""
        import re
        q = query.lower().strip().rstrip('?.!')
        # Remove common question words for normalization
        q = re.sub(r'^(what|who|where|when|why|how|which|is|are|was|were|do|does|did|can|could|would|should|has|have|had)\s+', '', q)
        q = re.sub(r'\s+', ' ', q).strip()
        return hashlib.md5(q.encode()).hexdigest()[:16]  # Shorter keys for speed

    def _cache_result(self, key: str, result: RetrievalResult) -> None:
        """Cache a result with LRU eviction."""
        if len(self._cache) >= self._cache_size:
            # Evict oldest
            oldest_key = self._cache_order.pop(0)
            self._cache.pop(oldest_key, None)
        self._cache[key] = result
        self._cache_order.append(key)

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._cache_order.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        return {
            "size": len(self._cache),
            "max_size": self._cache_size,
            "hit_rate": self._stats["hits"] / total if total > 0 else 0.0,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "api_calls": self._stats["api_calls"],
        }

    def close(self) -> None:
        """Close HTTP client."""
        if self._client and hasattr(self._client, 'close'):
            self._client.close()
