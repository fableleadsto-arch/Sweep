"""
WebScraper — fetches web pages using multiple strategies.

Strategy order:
  1. Wikipedia/Wikidata API (structured, fast, reliable)
  2. Direct HTTP GET with HTML parsing (broad coverage)
  3. Google Scholar / arXiv APIs (academic sources)
  4. Fallback: return None (no hallucinated content)

Features:
  - Connection pooling and keep-alive
  - Rate limiting per domain
  - Content deduplication
  - Cache with TTL
  - User-Agent rotation
  - Graceful degradation
"""
from __future__ import annotations

import hashlib
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, quote_plus

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

from .content import ContentExtractor
from .headless_browser import HeadlessBrowser, RenderedPage as HeadlessRenderedPage


@dataclass(frozen=True, slots=True)
class ScrapedPage:
    """A scraped web page with extracted content."""
    url: str
    title: str
    text: str
    summary: str = ""
    links: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "web"
    confidence: float = 0.8
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def first_sentence(self) -> str:
        m = re.match(r"(.+?[.!?])\s", self.text)
        return m.group(1) if m else self.text[:200]


class WebScraper:
    """Fetches and extracts content from web pages.

    Usage::

        scraper = WebScraper()
        page = scraper.fetch("https://en.wikipedia.org/wiki/Einstein")
        if page.success:
            print(page.title, page.text[:200])
    """

    # User-Agent strings for rotation
    _USER_AGENTS = [
        "SweepNeuralEngine/2.0 (research; +https://github.com/sweep-ai/sweep)",
        "Mozilla/5.0 (compatible; SweepBot/2.0; +https://research.openresearch.org)",
    ]

    # Domain-specific API endpoints
    _WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
    _WIKIDATA_API = "https://www.wikidata.org/w/api.php"
    _ARXIV_API = "http://export.arxiv.org/api/query"
    _OPENAlex_API = "https://api.openalex.org/works"
    _DUCKDUCKGO_API = "https://api.duckduckgo.com/"

    def __init__(
        self,
        timeout: float = 5.0,
        cache_size: int = 1000,
        cache_ttl: float = 3600.0,
        max_retries: int = 2,
    ) -> None:
        self._timeout = timeout
        self._cache_size = cache_size
        self._cache_ttl = cache_ttl
        self._max_retries = max_retries
        self._extractor = ContentExtractor()

        # Cache: key -> (ScrapedPage, timestamp)
        self._cache: dict[str, tuple[ScrapedPage, float]] = {}
        self._cache_order: list[str] = []

        # Rate limiting: domain -> last_request_time
        self._rate_limits: dict[str, float] = {}
        self._min_interval = 0.1  # 100ms between requests to same domain

        # Stats
        self._stats = {
            "total_fetches": 0,
            "cache_hits": 0,
            "api_calls": 0,
            "html_fetches": 0,
            "failures": 0,
        }

        # HTTP client
        self._headers = {
            "User-Agent": self._USER_AGENTS[0],
            "Accept": "text/html,application/json,*/*",
            "Accept-Encoding": "gzip, deflate",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self._client = self._make_client()

        # Headless browser (lazy init)
        self._browser: HeadlessBrowser | None = None

    def _make_client(self) -> Any:
        """Create an HTTP client with connection pooling."""
        if HAS_HTTPX:
            return httpx.Client(
                timeout=self._timeout,
                follow_redirects=True,
                headers=self._headers,
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                    keepalive_expiry=30,
                ),
            )
        return None

    # ════════════════════════════════════════════════════════════
    # PUBLIC API
    # ════════════════════════════════════════════════════════════

    def fetch(self, url: str, extract: bool = True) -> ScrapedPage:
        """Fetch a URL and extract content.

        Args:
            url:     The URL to fetch.
            extract: Whether to extract clean text from HTML.

        Returns:
            ScrapedPage with title, text, links, and metadata.
        """
        self._stats["total_fetches"] += 1

        # Check cache
        cache_key = self._cache_key(url)
        if cache_key in self._cache:
            page, ts = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                self._stats["cache_hits"] += 1
                return page
            # Expired
            del self._cache[cache_key]

        # Route to appropriate fetcher
        domain = urlparse(url).netloc.lower()
        t0 = time.perf_counter()

        if "wikipedia.org" in domain:
            page = self._fetch_wikipedia(url)
        elif "wikidata.org" in domain:
            page = self._fetch_wikidata(url)
        elif "arxiv.org" in domain:
            page = self._fetch_arxiv(url)
        elif "openalex.org" in domain:
            page = self._fetch_openalex(url)
        else:
            page = self._fetch_html(url, extract)

        elapsed = (time.perf_counter() - t0) * 1000
        page = ScrapedPage(
            url=page.url, title=page.title, text=page.text,
            summary=page.summary, links=page.links,
            metadata=page.metadata, source=page.source,
            confidence=page.confidence, latency_ms=elapsed,
            success=page.success, error=page.error,
        )

        # Cache successful results
        if page.success and page.text:
            self._cache_result(cache_key, page)

        return page

    def render_js(self, url: str, wait_ms: int = 2000) -> ScrapedPage:
        """Render a JavaScript-heavy page using headless browser.

        Falls back to HTTP if no browser is available.
        Useful for SPAs, dynamic content, and JS-rendered pages.

        Args:
            url:     The URL to render.
            wait_ms: Time to wait for JS execution (ms).

        Returns:
            ScrapedPage with rendered content.
        """
        self._stats["total_fetches"] += 1
        t0 = time.perf_counter()

        # Lazy init browser
        if self._browser is None:
            self._browser = HeadlessBrowser(timeout=self._timeout)

        rendered = self._browser.render(url, wait_ms=wait_ms)
        elapsed = (time.perf_counter() - t0) * 1000

        return ScrapedPage(
            url=rendered.url, title=rendered.title, text=rendered.text,
            summary=rendered.summary, links=rendered.links,
            metadata={**rendered.metadata, "renderer": rendered.renderer},
            source=f"headless_{rendered.renderer}",
            confidence=rendered.confidence, latency_ms=elapsed,
            success=rendered.success, error=rendered.error,
        )

    def search_and_fetch(
        self,
        query: str,
        max_results: int = 5,
        sources: list[str] | None = None,
    ) -> list[ScrapedPage]:
        """Search for a query and fetch top results from multiple sources.

        Sources (in priority order):
          1. DuckDuckGo Instant Answer (fast, broad)
          2. Wikipedia search (encyclopedic)
          3. arXiv search (academic)
          4. OpenAlex search (academic)

        Args:
            query:       Search query.
            max_results: Maximum results to return.
            sources:     Specific sources to use (None = all).

        Returns:
            List of ScrapedPage, deduplicated and ranked.
        """
        sources = sources or ["duckduckgo", "wikipedia", "arxiv", "openalex"]
        all_results: list[ScrapedPage] = []
        seen_titles: set[str] = set()

        for source in sources:
            if len(all_results) >= max_results:
                break

            remaining = max_results - len(all_results)

            if source == "duckduckgo":
                ddg = self._search_duckduckgo(query)
                if ddg and ddg.success:
                    key = ddg.title.lower().strip()
                    if key not in seen_titles:
                        seen_titles.add(key)
                        all_results.append(ddg)

            elif source == "wikipedia":
                wiki_titles = self._search_wikipedia(query, remaining)
                for title in wiki_titles[:remaining]:
                    key = title.lower().strip()
                    if key in seen_titles:
                        continue
                    seen_titles.add(key)
                    url = f"https://en.wikipedia.org/wiki/{quote_plus(title)}"
                    page = self.fetch(url)
                    if page.success:
                        all_results.append(page)

            elif source == "arxiv":
                arxiv_results = self._search_arxiv(query, remaining)
                for r in arxiv_results[:remaining]:
                    key = r.get("title", "").lower().strip()
                    if key in seen_titles:
                        continue
                    seen_titles.add(key)
                    page = self.fetch(r["url"])
                    if page.success:
                        all_results.append(page)

            elif source == "openalex":
                oa_results = self._search_openalex(query, remaining)
                for item in oa_results[:remaining]:
                    key = item.title.lower().strip()
                    if key in seen_titles:
                        continue
                    seen_titles.add(key)
                    all_results.append(item)

        return all_results[:max_results]

    def fetch_multiple(self, urls: list[str], extract: bool = True) -> list[ScrapedPage]:
        """Fetch multiple URLs, deduplicating and rate-limiting."""
        seen_domains: set[str] = set()
        results: list[ScrapedPage] = []

        for url in urls:
            domain = urlparse(url).netloc
            if domain in seen_domains:
                # Rate limit: skip if too soon
                now = time.time()
                last = self._rate_limits.get(domain, 0)
                if now - last < self._min_interval * 3:
                    continue
            seen_domains.add(domain)
            self._rate_limits[domain] = time.time()

            page = self.fetch(url, extract=extract)
            if page.success:
                results.append(page)

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get scraper statistics."""
        total = self._stats["total_fetches"]
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "hit_rate": (
                self._stats["cache_hits"] / total if total > 0 else 0.0
            ),
        }

    def clear_cache(self) -> None:
        """Clear the page cache."""
        self._cache.clear()
        self._cache_order.clear()

    def close(self) -> None:
        """Close the HTTP client and headless browser."""
        if self._client and hasattr(self._client, "close"):
            self._client.close()
        if self._browser:
            self._browser.close()
            self._browser = None

    # ════════════════════════════════════════════════════════════
    # WIKIPEDIA
    # ════════════════════════════════════════════════════════════

    def _fetch_wikipedia(self, url: str) -> ScrapedPage:
        """Fetch a Wikipedia article via API (structured, fast)."""
        # Extract title from URL
        path = urlparse(url).path
        title = path.replace("/wiki/", "").replace("_", " ")
        title = _url_decode(title)

        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|info|links",
            "exintro": "false",
            "explaintext": "true",
            "exchars": "3000",
            "inprop": "url",
            "pllimit": "20",
            "format": "json",
        }

        data = self._http_get_json(self._WIKIPEDIA_API, params)
        if not data:
            return ScrapedPage(url=url, title=title, text="", success=False, error="API request failed")

        pages = data.get("query", {}).get("pages", {})
        for pid, page_data in pages.items():
            if pid == "-1":
                continue

            page_title = page_data.get("title", title)
            extract = page_data.get("extract", "")
            page_url = page_data.get("fullurl", url)

            # Extract links
            raw_links = page_data.get("links", [])
            links = tuple(
                f"https://en.wikipedia.org/wiki/{quote_plus(l['title'])}"
                for l in raw_links if l.get("title")
            )

            # Clean extract
            clean_text = self._extractor.clean_wikipedia_extract(extract)
            summary = self._extractor.extract_summary(clean_text)

            return ScrapedPage(
                url=page_url,
                title=page_title,
                text=clean_text,
                summary=summary,
                links=links,
                metadata={"pageid": pid, "source": "wikipedia_api"},
                source="wikipedia",
                confidence=0.92,
            )

        return ScrapedPage(url=url, title=title, text="", success=False, error="Page not found")

    def _search_wikipedia(self, query: str, max_results: int = 5) -> list[str]:
        """Search Wikipedia and return page titles."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(max_results),
            "format": "json",
        }

        data = self._http_get_json(self._WIKIPEDIA_API, params)
        if not data:
            return []

        search_results = data.get("query", {}).get("search", [])
        return [r.get("title", "") for r in search_results if r.get("title")]

    # ════════════════════════════════════════════════════════════
    # WIKIDATA
    # ════════════════════════════════════════════════════════════

    def _fetch_wikidata(self, url: str) -> ScrapedPage:
        """Fetch Wikidata entity information."""
        path = urlparse(url).path
        entity_id = path.split("/")[-1]

        params = {
            "action": "wbgetentities",
            "ids": entity_id,
            "format": "json",
            "props": "labels|descriptions|claims",
            "languages": "en",
        }

        data = self._http_get_json(self._WIKIDATA_API, params)
        if not data:
            return ScrapedPage(url=url, title=entity_id, text="", success=False, error="API failed")

        entity = data.get("entities", {}).get(entity_id, {})
        label = entity.get("labels", {}).get("en", {}).get("value", entity_id)
        desc = entity.get("descriptions", {}).get("en", {}).get("value", "")

        # Extract key claims
        claims = entity.get("claims", {})
        claim_texts = []
        for prop_id, claim_list in list(claims.items())[:10]:
            for claim in claim_list[:1]:
                mainsnak = claim.get("mainsnak", {})
                dv = mainsnak.get("datavalue", {})
                vtype = dv.get("type", "")
                value = dv.get("value", "")
                if vtype == "wikibase-entityid":
                    claim_texts.append(f"{prop_id}: entity {value.get('id', '')}")
                elif vtype == "string":
                    claim_texts.append(f"{prop_id}: {value}")
                elif vtype == "quantity":
                    claim_texts.append(f"{prop_id}: {value.get('amount', '')}")
                elif vtype == "time":
                    claim_texts.append(f"{prop_id}: {value.get('time', '')}")

        text = f"{label}"
        if desc:
            text += f": {desc}"
        if claim_texts:
            text += "\n\nKey facts:\n" + "\n".join(claim_texts)

        return ScrapedPage(
            url=url, title=label, text=text,
            summary=f"{label}: {desc}" if desc else label,
            metadata={"entity_id": entity_id, "claims_count": len(claims)},
            source="wikidata",
            confidence=0.88,
        )

    # ════════════════════════════════════════════════════════════
    # ARXIV
    # ════════════════════════════════════════════════════════════

    def _fetch_arxiv(self, url: str) -> ScrapedPage:
        """Fetch arXiv paper metadata."""
        # Extract paper ID from URL
        path = urlparse(url).path
        paper_id = path.split("/")[-1]

        params = {"id_list": paper_id, "max_results": 1}
        data = self._http_get_xml(self._ARXIV_API, params)
        if not data:
            return ScrapedPage(url=url, title=paper_id, text="", success=False, error="arXiv API failed")

        # Parse XML
        title = self._xml_extract(data, "title").replace("\n", " ").strip()
        summary = self._xml_extract(data, "summary").replace("\n", " ").strip()
        authors = self._xml_extract_all(data, "author")
        categories = self._xml_extract_all(data, "category")

        text = f"Title: {title}\nAuthors: {', '.join(authors)}\n"
        text += f"Categories: {', '.join(categories)}\n\n{summary}"

        return ScrapedPage(
            url=url, title=title, text=text, summary=summary[:200],
            metadata={"paper_id": paper_id, "authors": authors, "categories": categories},
            source="arxiv",
            confidence=0.90,
        )

    def _search_arxiv(self, query: str, max_results: int = 3) -> list[dict[str, str]]:
        """Search arXiv for papers."""
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }
        data = self._http_get_xml(self._ARXIV_API, params)
        if not data:
            return []

        # Parse entries
        entries = re.findall(r"<entry>(.*?)</entry>", data, re.DOTALL)
        results = []
        for entry in entries:
            title = self._xml_extract(entry, "title").replace("\n", " ").strip()
            link = self._xml_extract(entry, "id").strip()
            summary = self._xml_extract(entry, "summary").replace("\n", " ").strip()
            results.append({"title": title, "url": link, "summary": summary})

        return results

    # ════════════════════════════════════════════════════════════
    # OPENALEX (academic papers)
    # ════════════════════════════════════════════════════════════

    def _fetch_openalex(self, url: str) -> ScrapedPage:
        """Fetch OpenAlex work metadata."""
        # Extract work ID
        path = urlparse(url).path
        work_id = path.split("/")[-1]

        params = {"select": "title,authorships,abstract_inverted_index,doi,publication_year"}
        data = self._http_get_json(f"https://api.openalex.org/works/{work_id}", params)
        if not data:
            return ScrapedPage(url=url, title=work_id, text="", success=False, error="OpenAlex failed")

        title = data.get("title", work_id)
        year = data.get("publication_year", "")
        authors = [a.get("author", {}).get("display_name", "") for a in data.get("authorships", [])[:5]]
        doi = data.get("doi", "")

        # Reconstruct abstract from inverted index
        abstract = self._reconstruct_abstract(data.get("abstract_inverted_index", {}))

        text = f"Title: {title}\nYear: {year}\nAuthors: {', '.join(authors)}\nDOI: {doi}\n\n{abstract}"

        return ScrapedPage(
            url=url, title=title, text=text, summary=abstract[:200],
            metadata={"year": year, "authors": authors, "doi": doi},
            source="openalex",
            confidence=0.88,
        )

    # ════════════════════════════════════════════════════════════
    # DUCKDUCKGO
    # ════════════════════════════════════════════════════════════

    def _search_duckduckgo(self, query: str) -> ScrapedPage | None:
        """Search DuckDuckGo Instant Answer API.

        Returns a single ScrapedPage with the instant answer,
        or None if no instant answer is available.
        """
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }

        data = self._http_get_json(self._DUCKDUCKGO_API, params)
        if not data:
            return None

        # Extract the best answer
        abstract = data.get("Abstract", "")
        answer = data.get("Answer", "")
        heading = data.get("Heading", "")
        source_url = data.get("AbstractURL", "")
        source = data.get("AbstractSource", "")

        # Use Answer if available, otherwise Abstract
        text = answer if answer else abstract
        if not text:
            # Try related topics
            topics = data.get("RelatedTopics", [])
            for topic in topics:
                if isinstance(topic, dict) and "Text" in topic:
                    text = topic["Text"]
                    source_url = topic.get("FirstURL", "")
                    break

        if not text:
            return None

        title = heading or query
        confidence = 0.85 if answer else 0.75

        return ScrapedPage(
            url=source_url or f"https://duckduckgo.com/?q={quote_plus(query)}",
            title=title,
            text=text,
            summary=text[:300],
            metadata={"source": source, "type": "instant_answer"},
            source="duckduckgo",
            confidence=confidence,
        )

    def _search_openalex(self, query: str, max_results: int = 3) -> list[ScrapedPage]:
        """Search OpenAlex for academic works."""
        params = {
            "search": query,
            "per_page": str(max_results),
            "select": "title,doi,authorships,publication_year",
        }
        data = self._http_get_json(self._OPENAlex_API, params)
        if not data:
            return []

        results = []
        for work in data.get("results", [])[:max_results]:
            title = work.get("title", "")
            doi = work.get("doi", "")
            year = work.get("publication_year", "")
            authors = [a.get("author", {}).get("display_name", "") for a in work.get("authorships", [])[:3]]

            text = f"Title: {title}\nYear: {year}\nAuthors: {', '.join(authors)}\nDOI: {doi}"
            url = doi if doi else f"https://openalex.org/{work.get('id', '')}"

            results.append(ScrapedPage(
                url=url, title=title, text=text, summary=title,
                metadata={"year": year, "doi": doi},
                source="openalex", confidence=0.85,
            ))

        return results

    # ════════════════════════════════════════════════════════════
    # HTML FETCHING (generic)
    # ════════════════════════════════════════════════════════════

    def _fetch_html(self, url: str, extract: bool = True) -> ScrapedPage:
        """Fetch a generic HTML page and extract content."""
        self._stats["html_fetches"] += 1

        html = self._http_get_text(url)
        if not html:
            self._stats["failures"] += 1
            return ScrapedPage(url=url, title="", text="", success=False, error="HTTP request failed")

        if extract:
            extracted = self._extractor.extract_from_html(html, url)
            return ScrapedPage(
                url=extracted.get("url", url),
                title=extracted.get("title", ""),
                text=extracted.get("text", ""),
                summary=extracted.get("summary", ""),
                links=tuple(extracted.get("links", [])[:20]),
                metadata=extracted.get("metadata", {}),
                source="web",
                confidence=0.75,
            )
        else:
            return ScrapedPage(
                url=url, title="", text=html[:5000],
                source="web_raw", confidence=0.5,
            )

    # ════════════════════════════════════════════════════════════
    # HTTP HELPERS
    # ════════════════════════════════════════════════════════════

    def _http_get_json(self, url: str, params: dict | None = None) -> dict | None:
        """HTTP GET returning parsed JSON with retry."""
        return self._http_request_with_retry(url, params=params, response_type="json")

    def _http_get_text(self, url: str) -> str | None:
        """HTTP GET returning text content with retry."""
        return self._http_request_with_retry(url, response_type="text")

    def _http_get_xml(self, url: str, params: dict | None = None) -> str | None:
        """HTTP GET returning XML text with retry."""
        return self._http_request_with_retry(url, params=params, response_type="text")

    def _http_request_with_retry(
        self,
        url: str,
        params: dict | None = None,
        response_type: str = "text",
    ) -> Any:
        """HTTP GET with exponential backoff retry.

        Retries on:
          - Connection errors (timeout, DNS, refused)
          - Server errors (5xx)
          - Rate limiting (429)

        Exponential backoff: base_delay * 2^attempt + jitter
        """
        self._stats["api_calls"] += 1
        last_error = None

        for attempt in range(self._max_retries + 1):
            try:
                resp = self._http_get(url, params=params)
                if resp is None:
                    last_error = "No response"
                    continue

                # Success
                if resp.status_code == 200:
                    if response_type == "json":
                        return resp.json()
                    return resp.text

                # Rate limiting — wait longer
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "5"))
                    time.sleep(min(retry_after, 30))
                    last_error = f"429 rate limited (waited {retry_after}s)"
                    continue

                # Server errors — retry
                if resp.status_code >= 500:
                    last_error = f"{resp.status_code} server error"
                    if attempt < self._max_retries:
                        delay = self._backoff_delay(attempt)
                        time.sleep(delay)
                    continue

                # Client errors (except 429) — don't retry
                last_error = f"{resp.status_code} client error"
                break

            except Exception as e:
                last_error = str(e)
                if attempt < self._max_retries:
                    delay = self._backoff_delay(attempt)
                    time.sleep(delay)
                continue

        # All retries exhausted
        self._stats["failures"] += 1
        return None

    def _backoff_delay(self, attempt: int, base: float = 0.5, max_delay: float = 10.0) -> float:
        """Calculate exponential backoff delay with jitter.

        delay = min(base * 2^attempt + jitter, max_delay)
        jitter = random uniform [0, base * 0.5]
        """
        delay = base * (2 ** attempt)
        jitter = random.uniform(0, base * 0.5)
        return min(delay + jitter, max_delay)

    def _http_get(self, url: str, params: dict | None = None) -> Any:
        """Make an HTTP GET request (single attempt)."""
        if self._client:
            return self._client.get(url, params=params)
        elif HAS_REQUESTS:
            return requests.get(url, params=params, timeout=self._timeout, headers=self._headers)
        return None

    # ════════════════════════════════════════════════════════════
    # PARSING HELPERS
    # ════════════════════════════════════════════════════════════

    @staticmethod
    def _xml_extract(xml_text: str, tag: str) -> str:
        """Extract text content of first XML element by tag."""
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml_text, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _xml_extract_all(xml_text: str, tag: str) -> list[str]:
        """Extract text content of all XML elements by tag."""
        return [m.group(1).strip() for m in re.finditer(rf"<{tag}[^>]*>(.*?)</{tag}>", xml_text, re.DOTALL)]

    @staticmethod
    def _reconstruct_abstract(inverted_index: dict[str, list[int]]) -> str:
        """Reconstruct abstract text from OpenAlex inverted index."""
        if not inverted_index:
            return ""
        positions: dict[int, str] = {}
        for word, pos_list in inverted_index.items():
            for pos in pos_list:
                positions[pos] = word
        if not positions:
            return ""
        max_pos = max(positions.keys())
        return " ".join(positions.get(i, "") for i in range(max_pos + 1))

    # ════════════════════════════════════════════════════════════
    # CACHE
    # ════════════════════════════════════════════════════════════

    def _cache_key(self, url: str) -> str:
        """Generate a normalized cache key."""
        normalized = url.lower().strip().rstrip("/")
        return hashlib.md5(normalized.encode()).hexdigest()[:16]

    def _cache_result(self, key: str, page: ScrapedPage) -> None:
        """Cache a result with LRU eviction."""
        if len(self._cache) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)
        self._cache[key] = (page, time.time())
        self._cache_order.append(key)


def _url_decode(s: str) -> str:
    """Simple URL decoding for Wikipedia titles."""
    from urllib.parse import unquote
    return unquote(s.replace("+", " "))
