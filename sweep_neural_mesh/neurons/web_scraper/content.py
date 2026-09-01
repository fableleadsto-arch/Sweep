"""
ContentExtractor — extracts clean text, metadata, and structure from HTML.

No external HTML parser dependencies — uses regex-based extraction
that works with any Python installation.

Features:
  - Title extraction
  - Main content extraction (strips nav, footer, ads)
  - Link extraction
  - Metadata extraction (description, keywords, og: tags)
  - Wikipedia-specific extraction
  - Sentence segmentation
  - Text cleaning and normalization
"""
from __future__ import annotations

import re
from typing import Any
from html import unescape as html_unescape


class ContentExtractor:
    """Extracts clean content from HTML pages.

    Usage::

        extractor = ContentExtractor()
        result = extractor.extract_from_html(html_text, url)
        print(result["title"], result["text"][:200])
    """

    # Tags that typically contain boilerplate (not main content)
    _BOILERPLATE_TAGS = (
        "nav", "footer", "header", "aside", "script", "style",
        "noscript", "iframe", "form", "button", "svg", "canvas",
    )

    # CSS classes/ids that indicate boilerplate
    _BOILERPLATE_PATTERNS = re.compile(
        r"(sidebar|menu|nav|footer|header|ad[-_]|banner|cookie|popup|modal|"
        r"breadcrumb|pagination|social|share|comment|widget|signup|login|"
        r"related[-_]articles|recommended)",
        re.IGNORECASE,
    )

    def extract_from_html(self, html: str, url: str = "") -> dict[str, Any]:
        """Extract clean content from HTML.

        Returns dict with:
            title:    Page title
            text:     Clean main text content
            summary:  First paragraph / meta description
            links:    List of extracted URLs
            metadata: Dict of extracted metadata
        """
        if not html:
            return {"title": "", "text": "", "summary": "", "links": [], "metadata": {}}

        # 1. Extract metadata first (before stripping tags)
        metadata = self._extract_metadata(html)

        # 2. Extract title
        title = self._extract_title(html, metadata)

        # 3. Remove boilerplate
        cleaned = self._remove_boilerplate(html)

        # 4. Extract main content text
        text = self._extract_text(cleaned)

        # 5. Extract links
        links = self._extract_links(html, url)

        # 6. Generate summary
        summary = metadata.get("description", "")
        if not summary:
            summary = self._extract_summary(text)

        return {
            "title": title,
            "text": text,
            "summary": summary,
            "links": links,
            "metadata": metadata,
            "url": url,
        }

    def extract_text_only(self, html: str) -> str:
        """Extract just the clean text content, no metadata."""
        if not html:
            return ""
        cleaned = self._remove_boilerplate(html)
        return self._extract_text(cleaned)

    def clean_wikipedia_extract(self, extract: str) -> str:
        """Clean a Wikipedia API extract (plain text, no HTML)."""
        if not extract:
            return ""

        # Remove section headers (lines that are all caps or end with colon)
        lines = extract.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                cleaned.append("")
                continue
            # Skip lines that look like section headers
            if re.match(r"^[A-Z][A-Z\s]{3,}$", line):
                continue
            if line.endswith(":") and len(line) < 80 and not line[0].isdigit():
                continue
            cleaned.append(line)

        text = "\n".join(cleaned)

        # Clean up reference markers [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", text)
        # Clean up multiple spaces
        text = re.sub(r"  +", " ", text)
        # Clean up multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def extract_summary(self, text: str, max_sentences: int = 3) -> str:
        """Extract a summary from text (first N sentences)."""
        if not text:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        summary_parts = []
        char_count = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if char_count + len(s) > 500:
                break
            summary_parts.append(s)
            char_count += len(s)
            if len(summary_parts) >= max_sentences:
                break
        return " ".join(summary_parts)

    # ════════════════════════════════════════════════════════════
    # METADATA
    # ════════════════════════════════════════════════════════════

    def _extract_metadata(self, html: str) -> dict[str, str]:
        """Extract metadata from HTML head."""
        metadata: dict[str, str] = {}

        # Meta tags
        for m in re.finditer(
            r'<meta\s+[^>]*?(?:name|property)=["\']([^"\']+)["\']\s+[^>]*?content=["\']([^"\']*)["\']',
            html, re.IGNORECASE,
        ):
            key = m.group(1).lower()
            val = html_unescape(m.group(2))
            if key.startswith("og:"):
                metadata[key] = val
            elif key in ("description", "keywords", "author", "robots"):
                metadata[key] = val

        # Also check reversed attribute order: content before name/property
        for m in re.finditer(
            r'<meta\s+[^>]*?content=["\']([^"\']*)["\']\s+[^>]*?(?:name|property)=["\']([^"\']+)["\']',
            html, re.IGNORECASE,
        ):
            val = html_unescape(m.group(1))
            key = m.group(2).lower()
            if key.startswith("og:"):
                metadata.setdefault(key, val)
            elif key in ("description", "keywords", "author", "robots"):
                metadata.setdefault(key, val)

        # Title tag
        title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if title_m:
            metadata["html_title"] = html_unescape(title_m.group(1)).strip()

        return metadata

    # ════════════════════════════════════════════════════════════
    # TITLE
    # ════════════════════════════════════════════════════════════

    def _extract_title(self, html: str, metadata: dict) -> str:
        """Extract the page title."""
        # og:title takes priority
        if "og:title" in metadata:
            return metadata["og:title"]

        # HTML title tag
        if "html_title" in metadata:
            title = metadata["html_title"]
            # Wikipedia: remove " - Wikipedia" suffix
            title = re.sub(r"\s*[-–—]\s*(Wikipedia|Wikidata).*$", "", title)
            return title

        # <h1> tag
        h1_m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
        if h1_m:
            return self._strip_tags(h1_m.group(1)).strip()

        return ""

    # ════════════════════════════════════════════════════════════
    # BOILERPLATE REMOVAL
    # ════════════════════════════════════════════════════════════

    def _remove_boilerplate(self, html: str) -> str:
        """Remove navigation, footer, scripts, and other boilerplate."""
        text = html

        # Remove script and style blocks
        text = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML comments
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

        # Remove boilerplate tags and their content
        for tag in self._BOILERPLATE_TAGS:
            text = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Remove elements with boilerplate classes/ids
        text = re.sub(
            r'<(?:div|section|aside|article)\s+[^>]*(?:class|id)=["\'][^"\']*'
            + self._BOILERPLATE_PATTERNS.pattern
            + r'[^"\']*["\'][^>]*>.*?</(?:div|section|aside|article)>',
            "", text, flags=re.DOTALL | re.IGNORECASE,
        )

        return text

    # ════════════════════════════════════════════════════════════
    # TEXT EXTRACTION
    # ════════════════════════════════════════════════════════════

    def _extract_text(self, html: str) -> str:
        """Extract clean text from HTML."""
        # Convert block elements to newlines
        text = re.sub(r"<(?:br|/p|/div|/h[1-6]|/li|/tr|/blockquote)[^>]*>", "\n", html, flags=re.IGNORECASE)

        # Remove remaining tags
        text = self._strip_tags(text)

        # Decode HTML entities
        text = html_unescape(text)

        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Clean up
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if line and len(line) > 3:  # Skip very short fragments
                lines.append(line)

        return "\n".join(lines)

    def _extract_links(self, html: str, base_url: str = "") -> list[str]:
        """Extract URLs from HTML."""
        links = []
        seen = set()

        for m in re.finditer(r'<a\s+[^>]*?href=["\']([^"\']+)["\']', html, re.IGNORECASE):
            href = m.group(1).strip()
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue
            if href in seen:
                continue
            seen.add(href)

            # Make absolute if needed
            if href.startswith("/") and base_url:
                from urllib.parse import urljoin
                href = urljoin(base_url, href)

            links.append(href)

        return links[:50]  # Limit

    # ════════════════════════════════════════════════════════════
    # HELPERS
    # ════════════════════════════════════════════════════════════

    def _extract_summary(self, text: str) -> str:
        """Extract the first meaningful paragraph as summary."""
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            # Skip very short paragraphs (likely navigation fragments)
            if len(para) > 50:
                # Return first sentence or first 300 chars
                first_sentence = re.match(r"(.+?[.!?])\s", para)
                if first_sentence:
                    return first_sentence.group(1)
                return para[:300]
        return text[:300] if text else ""

    @staticmethod
    def _strip_tags(html: str) -> str:
        """Remove all HTML tags."""
        return re.sub(r"<[^>]+>", "", html)
