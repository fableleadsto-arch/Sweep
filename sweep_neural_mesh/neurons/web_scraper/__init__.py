"""
Web Scraper — fetches and extracts clean content from web pages.

Modules:
  - Scraper:          Core web fetching with retry logic
  - ContentExtractor: HTML → clean text, metadata, links
  - WebResearcher:    Multi-query, multi-source web research
  - PDFScraper:       PDF text and metadata extraction
  - HeadlessBrowser:  JavaScript-rendered page support

Usage::

    from neurons.web_scraper import WebScraper, WebResearcher, PDFScraper, HeadlessBrowser

    # Web scraping (with retry)
    scraper = WebScraper()
    page = scraper.fetch("https://en.wikipedia.org/wiki/Quantum_computing")
    print(page.title, page.text[:200])

    # JavaScript-rendered pages
    js_page = scraper.render_js("https://example.com/spa")

    # Multi-source research
    researcher = WebResearcher()
    report = researcher.research("quantum computing applications")
    for item in report.findings:
        print(item.text[:100])

    # PDF extraction
    pdf = PDFScraper()
    result = pdf.extract_from_url("https://arxiv.org/pdf/2301.00001.pdf")
    print(result.text[:500])
"""
from .scraper import WebScraper, ScrapedPage
from .content import ContentExtractor
from .researcher import WebResearcher, ResearchFinding, ResearchReport
from .pdf_scraper import PDFScraper, PDFResult
from .headless_browser import HeadlessBrowser, RenderedPage, render_page, render_pages

__all__ = [
    "WebScraper",
    "ScrapedPage",
    "ContentExtractor",
    "WebResearcher",
    "ResearchFinding",
    "ResearchReport",
    "PDFScraper",
    "PDFResult",
    "HeadlessBrowser",
    "RenderedPage",
    "render_page",
    "render_pages",
]
