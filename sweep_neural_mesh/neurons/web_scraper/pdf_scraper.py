"""
PDFScraper — extracts text, metadata, and structure from PDF documents.

Strategy:
  1. Try pypdf (fast, reliable) if available
  2. Fallback to pure-Python regex extraction from raw PDF bytes
  3. Fallback to HTTP fetch + stream parsing for remote PDFs

Features:
  - Text extraction with page boundaries
  - Metadata extraction (title, author, subject, keywords)
  - Table detection (basic)
  - Link extraction from PDF annotations
  - Remote PDF fetching with streaming
  - No external dependencies required (pure-Python fallback)

Usage::

    from neurons.web_scraper.pdf_scraper import PDFScraper

    scraper = PDFScraper()

    # From file
    result = scraper.extract_from_file("paper.pdf")
    print(result.title, result.text[:200])

    # From URL
    result = scraper.extract_from_url("https://arxiv.org/pdf/2301.00001.pdf")
    print(result.text[:500])
"""
from __future__ import annotations

import io
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

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

# Try pypdf for reliable PDF parsing
try:
    from pypdf import PdfReader as PyPDFReader
    HAS_PYPDF = True
except ImportError:
    try:
        from PyPDF2 import PdfReader as PyPDFReader
        HAS_PYPDF = True
    except ImportError:
        HAS_PYPDF = False
        PyPDFReader = None  # type: ignore


@dataclass(frozen=True, slots=True)
class PDFResult:
    """Result from PDF extraction."""
    text: str
    title: str = ""
    author: str = ""
    subject: str = ""
    keywords: tuple[str, ...] = ()
    page_count: int = 0
    pages: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    links: tuple[str, ...] = ()
    tables_detected: int = 0
    source: str = "pdf"
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

    @property
    def summary(self) -> str:
        """First paragraph or first 500 chars."""
        paragraphs = self.text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:
                return para[:500]
        return self.text[:500]


class PDFScraper:
    """Extracts content from PDF documents.

    Usage::

        scraper = PDFScraper()
        result = scraper.extract_from_file("paper.pdf")
        print(result.text[:200])
    """

    def __init__(self, timeout: float = 10.0, max_retries: int = 3) -> None:
        self._timeout = timeout
        self._max_retries = max_retries
        self._headers = {
            "User-Agent": "SweepNeuralEngine/2.0 (research; +https://github.com/sweep-ai/sweep)",
            "Accept": "application/pdf,*/*",
        }
        self._client = self._make_client()
        self._stats = {"total_extractions": 0, "pypdf": 0, "pure_python": 0, "failures": 0, "retries": 0}

    def _make_client(self) -> Any:
        """Create HTTP client for fetching remote PDFs."""
        if HAS_HTTPX:
            return httpx.Client(
                timeout=self._timeout,
                follow_redirects=True,
                headers=self._headers,
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=3),
            )
        return None

    # ════════════════════════════════════════════════════════════
    # PUBLIC API
    # ════════════════════════════════════════════════════════════

    def extract_from_file(self, path: str) -> PDFResult:
        """Extract text and metadata from a local PDF file."""
        self._stats["total_extractions"] += 1
        t0 = time.perf_counter()

        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception as e:
            self._stats["failures"] += 1
            return PDFResult(text="", success=False, error=f"Cannot read file: {e}")

        result = self._extract_from_bytes(data)
        elapsed = (time.perf_counter() - t0) * 1000
        return PDFResult(
            text=result.text, title=result.title, author=result.author,
            subject=result.subject, keywords=result.keywords,
            page_count=result.page_count, pages=result.pages,
            metadata=result.metadata, links=result.links,
            tables_detected=result.tables_detected, source="pdf_file",
            confidence=result.confidence, latency_ms=elapsed,
            success=result.success, error=result.error,
        )

    def extract_from_url(self, url: str) -> PDFResult:
        """Fetch a PDF from a URL and extract content."""
        self._stats["total_extractions"] += 1
        t0 = time.perf_counter()

        # Fetch PDF bytes
        data = self._fetch_pdf(url)
        if data is None:
            self._stats["failures"] += 1
            return PDFResult(text="", success=False, error=f"Failed to fetch PDF from {url}")

        result = self._extract_from_bytes(data)
        elapsed = (time.perf_counter() - t0) * 1000
        return PDFResult(
            text=result.text, title=result.title, author=result.author,
            subject=result.subject, keywords=result.keywords,
            page_count=result.page_count, pages=result.pages,
            metadata={**result.metadata, "url": url}, links=result.links,
            tables_detected=result.tables_detected, source="pdf_url",
            confidence=result.confidence, latency_ms=elapsed,
            success=result.success, error=result.error,
        )

    def extract_from_bytes(self, data: bytes) -> PDFResult:
        """Extract content from raw PDF bytes."""
        self._stats["total_extractions"] += 1
        t0 = time.perf_counter()
        result = self._extract_from_bytes(data)
        elapsed = (time.perf_counter() - t0) * 1000
        return PDFResult(
            text=result.text, title=result.title, author=result.author,
            subject=result.subject, keywords=result.keywords,
            page_count=result.page_count, pages=result.pages,
            metadata=result.metadata, links=result.links,
            tables_detected=result.tables_detected, source="pdf_bytes",
            confidence=result.confidence, latency_ms=elapsed,
            success=result.success, error=result.error,
        )

    def get_stats(self) -> dict[str, int]:
        """Get extraction statistics."""
        return dict(self._stats)

    def close(self) -> None:
        """Close HTTP client."""
        if self._client and hasattr(self._client, "close"):
            self._client.close()

    # ════════════════════════════════════════════════════════════
    # CORE EXTRACTION
    # ════════════════════════════════════════════════════════════

    def _extract_from_bytes(self, data: bytes) -> PDFResult:
        """Extract content from PDF bytes using best available method."""
        # Validate PDF header
        if not data or not data[:5].startswith(b"%PDF"):
            return PDFResult(text="", success=False, error="Not a valid PDF file")

        # Try pypdf first (most reliable)
        if HAS_PYPDF:
            try:
                return self._extract_with_pypdf(data)
            except Exception:
                pass  # Fall through to pure Python

        # Pure Python fallback
        self._stats["pure_python"] += 1
        return self._extract_pure_python(data)

    def _extract_with_pypdf(self, data: bytes) -> PDFResult:
        """Extract using pypdf library."""
        self._stats["pypdf"] += 1

        reader = PyPDFReader(io.BytesIO(data))  # type: ignore

        # Metadata
        meta = reader.metadata or {}
        title = getattr(meta, "title", "") or ""
        author = getattr(meta, "author", "") or ""
        subject = getattr(meta, "subject", "") or ""
        keywords_raw = getattr(meta, "keywords", "") or ""
        keywords = tuple(k.strip() for k in keywords_raw.split(",") if k.strip()) if keywords_raw else ()

        # Extract text per page
        pages = []
        all_text = []
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                pages.append(page_text)
                all_text.append(page_text)
            except Exception:
                pages.append("")

        full_text = "\n\n".join(all_text)

        # Extract links from annotations
        links = self._extract_links_from_pypdf(reader)

        # Detect tables (basic heuristic)
        tables = self._detect_tables(full_text)

        # Extract metadata dict
        metadata = {
            "page_count": len(reader.pages),
            "title": title,
            "author": author,
            "subject": subject,
        }

        return PDFResult(
            text=full_text,
            title=title,
            author=author,
            subject=subject,
            keywords=keywords,
            page_count=len(reader.pages),
            pages=tuple(pages),
            metadata=metadata,
            links=tuple(links),
            tables_detected=tables,
            confidence=0.90 if full_text.strip() else 0.3,
        )

    def _extract_pure_python(self, data: bytes) -> PDFResult:
        """Extract text using pure Python regex (no dependencies)."""
        # Extract text between BT/ET markers (PDF text objects)
        text_objects = re.findall(rb"\((.*?)\)", data, re.DOTALL)

        # Also try to find text in stream objects
        stream_texts = re.findall(rb"BT\s*(.*?)\s*ET", data, re.DOTALL)

        # Decode and clean
        raw_parts = []
        for obj in text_objects:
            try:
                decoded = obj.decode("latin-1", errors="replace")
                # Clean PDF escape sequences
                decoded = decoded.replace("\\n", "\n").replace("\\r", "\n")
                decoded = decoded.replace("\\t", " ")
                decoded = re.sub(r"\\(\d{3})", lambda m: chr(int(m.group(1))), decoded)
                if len(decoded.strip()) > 2:
                    raw_parts.append(decoded.strip())
            except Exception:
                continue

        full_text = "\n".join(raw_parts)

        # Clean up
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        full_text = re.sub(r"[ ]{2,}", " ", full_text)

        # Try to extract title from PDF metadata strings
        title = ""
        title_m = re.search(rb"/Title\s*\((.*?)\)", data)
        if title_m:
            try:
                title = title_m.group(1).decode("latin-1", errors="replace")
            except Exception:
                pass

        author = ""
        author_m = re.search(rb"/Author\s*\((.*?)\)", data)
        if author_m:
            try:
                author = author_m.group(1).decode("latin-1", errors="replace")
            except Exception:
                pass

        # Count pages
        page_count = len(re.findall(rb"/Type\s*/Page[^s]", data))

        return PDFResult(
            text=full_text,
            title=title,
            author=author,
            page_count=max(1, page_count),
            metadata={"extraction_method": "pure_python"},
            confidence=0.60 if full_text.strip() else 0.2,
        )

    # ════════════════════════════════════════════════════════════
    # LINK EXTRACTION
    # ════════════════════════════════════════════════════════════

    def _extract_links_from_pypdf(self, reader: Any) -> list[str]:
        """Extract URLs from PDF annotations."""
        links = []
        seen = set()

        for page in reader.pages:
            try:
                annotations = page.get("/Annots")
                if not annotations:
                    continue
                for annot in annotations:
                    annot_obj = annot.get_object() if hasattr(annot, "get_object") else annot
                    if isinstance(annot_obj, dict):
                        uri = annot_obj.get("/A", {})
                        if isinstance(uri, dict):
                            url = uri.get("/URI", "")
                            if url and url not in seen:
                                seen.add(url)
                                links.append(url)
            except Exception:
                continue

        return links[:50]

    # ════════════════════════════════════════════════════════════
    # TABLE DETECTION
    # ════════════════════════════════════════════════════════════

    def _detect_tables(self, text: str) -> int:
        """Basic table detection using structural heuristics."""
        tables = 0
        lines = text.split("\n")

        # Look for consecutive lines with consistent column separators
        in_table = False
        col_pattern = re.compile(r"\s{3,}|\t")

        for i, line in enumerate(lines):
            cols = col_pattern.split(line.strip())
            if len(cols) >= 3:
                if not in_table:
                    in_table = True
                    tables += 1
            else:
                in_table = False

        return tables

    # ════════════════════════════════════════════════════════════
    # HTTP FETCHING
    # ════════════════════════════════════════════════════════════

    def _fetch_pdf(self, url: str) -> bytes | None:
        """Fetch PDF bytes from a URL with exponential backoff retry."""
        last_error = None

        for attempt in range(self._max_retries + 1):
            try:
                resp = self._http_get(url)
                if resp is None:
                    last_error = "No response"
                    continue

                if resp.status_code == 200:
                    return resp.content

                # Rate limiting
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "5"))
                    time.sleep(min(retry_after, 30))
                    last_error = f"429 rate limited"
                    self._stats["retries"] += 1
                    continue

                # Server errors — retry
                if resp.status_code >= 500:
                    last_error = f"{resp.status_code}"
                    if attempt < self._max_retries:
                        delay = self._backoff_delay(attempt)
                        time.sleep(delay)
                        self._stats["retries"] += 1
                    continue

                # Client error — don't retry
                last_error = f"{resp.status_code}"
                break

            except Exception as e:
                last_error = str(e)
                if attempt < self._max_retries:
                    delay = self._backoff_delay(attempt)
                    time.sleep(delay)
                    self._stats["retries"] += 1
                continue

        self._stats["failures"] += 1
        return None

    def _http_get(self, url: str) -> Any:
        """Make an HTTP GET request (single attempt)."""
        if self._client:
            return self._client.get(url)
        elif HAS_REQUESTS:
            return requests.get(url, timeout=self._timeout, headers=self._headers)
        return None

    def _backoff_delay(self, attempt: int, base: float = 1.0, max_delay: float = 15.0) -> float:
        """Exponential backoff with jitter."""
        delay = base * (2 ** attempt)
        jitter = random.uniform(0, base * 0.5)
        return min(delay + jitter, max_delay)


# ══════════════════════════════════════════════════════════════
# BATCH EXTRACTION
# ══════════════════════════════════════════════════════════════

def extract_pdfs_batch(
    sources: list[str],
    timeout: float = 10.0,
) -> list[PDFResult]:
    """Extract text from multiple PDFs (files or URLs).

    Args:
        sources: List of file paths or URLs pointing to PDFs.
        timeout: HTTP timeout for remote PDFs.

    Returns:
        List of PDFResult, one per source.
    """
    scraper = PDFScraper(timeout=timeout)
    results = []

    for source in sources:
        if source.startswith("http://") or source.startswith("https://"):
            result = scraper.extract_from_url(source)
        else:
            result = scraper.extract_from_file(source)
        results.append(result)

    scraper.close()
    return results
