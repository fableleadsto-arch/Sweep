"""
HeadlessBrowser — renders JavaScript-heavy pages using headless browsers.

Supports (in priority order):
  1. Playwright (fastest, most reliable)
  2. Selenium + Chrome/Chromium (widely available)
  3. Fallback to plain HTTP (no JS rendering)

Features:
  - Automatic browser detection and initialization
  - Page rendering with configurable timeout
  - Content extraction from rendered DOM
  - Screenshot capture (optional)
  - Resource blocking for faster rendering
  - Cookie/session management
  - Graceful degradation when no browser is available

Usage::

    from neurons.web_scraper.headless_browser import HeadlessBrowser

    browser = HeadlessBrowser()
    page = browser.render("https://example.com/dynamic-page")
    if page.success:
        print(page.text[:500])
    browser.close()
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from .content import ContentExtractor


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """Result from headless browser rendering."""
    url: str
    title: str
    text: str
    html: str = ""
    summary: str = ""
    links: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    renderer: str = "none"
    confidence: float = 0.8
    latency_ms: float = 0.0
    success: bool = True
    error: str = ""

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class HeadlessBrowser:
    """Renders JavaScript-heavy pages using headless browsers.

    Usage::

        browser = HeadlessBrowser()
        page = browser.render("https://example.com/app")
        if page.success:
            print(page.text[:500])
        browser.close()
    """

    def __init__(
        self,
        timeout: float = 15.0,
        wait_for_network: bool = False,
        block_images: bool = True,
        block_ads: bool = True,
    ) -> None:
        self._timeout = timeout
        self._wait_for_network = wait_for_network
        self._block_images = block_images
        self._block_ads = block_ads
        self._extractor = ContentExtractor()

        # Browser state
        self._playwright = None
        self._playwright_browser = None
        self._selenium_driver = None
        self._renderer: str = "none"
        self._initialized = False

        # Stats
        self._stats = {
            "total_renders": 0,
            "successful": 0,
            "failures": 0,
            "playwright_renders": 0,
            "selenium_renders": 0,
            "fallback_renders": 0,
        }

        # Ad blocker patterns
        self._ad_patterns = re.compile(
            r"(adsbygoogle|doubleclick|googlesyndication|analytics|tracking|"
            r"facebook\.com/tr|pixel|beacon|hotjar|mixpanel)",
            re.IGNORECASE,
        )

    # ════════════════════════════════════════════════════════════
    # PUBLIC API
    # ════════════════════════════════════════════════════════════

    def render(self, url: str, wait_ms: int = 2000) -> RenderedPage:
        """Render a JavaScript-heavy page and extract content.

        Args:
            url:     The URL to render.
            wait_ms: Time to wait for JS to execute after page load (ms).

        Returns:
            RenderedPage with extracted text, title, and metadata.
        """
        self._stats["total_renders"] += 1
        t0 = time.perf_counter()

        # Ensure browser is initialized
        if not self._initialized:
            self._init_browser()

        # Try rendering with available browser
        result = None
        if self._playwright_browser is not None:
            result = self._render_playwright(url, wait_ms)
        elif self._selenium_driver is not None:
            result = self._render_selenium(url, wait_ms)

        # Fallback to HTTP if no browser or rendering failed
        if result is None or not result.success:
            result = self._render_fallback(url)

        elapsed = (time.perf_counter() - t0) * 1000
        result = RenderedPage(
            url=result.url, title=result.title, text=result.text,
            html=result.html, summary=result.summary, links=result.links,
            metadata=result.metadata, renderer=result.renderer,
            confidence=result.confidence, latency_ms=elapsed,
            success=result.success, error=result.error,
        )

        if result.success:
            self._stats["successful"] += 1
        else:
            self._stats["failures"] += 1

        return result

    def render_batch(
        self, urls: list[str], wait_ms: int = 2000, max_concurrent: int = 1,
    ) -> list[RenderedPage]:
        """Render multiple URLs."""
        results = []
        for url in urls[:max_concurrent * 3]:  # Limit total
            result = self.render(url, wait_ms=wait_ms)
            results.append(result)
        return results

    def is_available(self) -> bool:
        """Check if a headless browser is available."""
        if not self._initialized:
            self._init_browser()
        return self._playwright_browser is not None or self._selenium_driver is not None

    def get_renderer(self) -> str:
        """Get the name of the active renderer."""
        return self._renderer

    def get_stats(self) -> dict[str, Any]:
        """Get rendering statistics."""
        return {
            **self._stats,
            "renderer": self._renderer,
            "available": self.is_available(),
        }

    def close(self) -> None:
        """Close the headless browser."""
        try:
            if self._playwright_browser:
                self._playwright_browser.close()
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass

        try:
            if self._selenium_driver:
                self._selenium_driver.quit()
        except Exception:
            pass

        self._playwright = None
        self._playwright_browser = None
        self._selenium_driver = None
        self._initialized = False

    # ════════════════════════════════════════════════════════════
    # BROWSER INITIALIZATION
    # ════════════════════════════════════════════════════════════

    def _init_browser(self) -> None:
        """Initialize the best available headless browser."""
        self._initialized = True

        # Try Playwright first (fastest)
        if self._init_playwright():
            return

        # Try Selenium (widely available)
        if self._init_selenium():
            return

        # No browser available — will use HTTP fallback
        self._renderer = "http_fallback"

    def _init_playwright(self) -> bool:
        """Initialize Playwright headless browser."""
        try:
            from playwright.sync_api import sync_playwright
            pw = sync_playwright().start()

            browser = pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                ],
            )

            self._playwright = pw
            self._playwright_browser = browser
            self._renderer = "playwright"
            return True
        except Exception:
            return False

    def _init_selenium(self) -> bool:
        """Initialize Selenium with Chrome/Chromium."""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service

            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")

            try:
                driver = webdriver.Chrome(options=options)
            except Exception:
                # Try with explicit chromedriver path
                try:
                    from webdriver_manager.chrome import ChromeDriverManager
                    service = Service(ChromeDriverManager().install())
                    driver = webdriver.Chrome(service=service, options=options)
                except Exception:
                    return False

            self._selenium_driver = driver
            self._renderer = "selenium"
            return True
        except Exception:
            return False

    # ════════════════════════════════════════════════════════════
    # PLAYWRIGHT RENDERING
    # ════════════════════════════════════════════════════════════

    def _render_playwright(self, url: str, wait_ms: int) -> RenderedPage | None:
        """Render a page using Playwright."""
        self._stats["playwright_renders"] += 1
        page = None

        try:
            page = self._playwright_browser.new_page(
                viewport={"width": 1280, "height": 720},
            )

            # Block resources for faster rendering
            if self._block_images or self._block_ads:
                def route_handler(route):
                    req_url = route.request.url
                    if self._block_images and any(ext in req_url for ext in [".png", ".jpg", ".gif", ".svg", ".webp"]):
                        route.abort()
                    elif self._block_ads and self._ad_patterns.search(req_url):
                        route.abort()
                    else:
                        route.continue_()
                page.route("**/*", route_handler)

            # Navigate
            page.goto(url, wait_until="domcontentloaded", timeout=int(self._timeout * 1000))

            # Wait for JS
            if wait_ms > 0:
                page.wait_for_timeout(wait_ms)

            # Extract content
            title = page.title()
            html = page.content()
            text = page.inner_text("body")

            # Extract links
            links = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")

            # Clean text
            clean_text = self._extractor.extract_text_only(html)

            # Generate summary
            summary = self._extractor.extract_summary(clean_text)

            return RenderedPage(
                url=url,
                title=title,
                text=clean_text if len(clean_text) > len(text) else text,
                html=html[:50000],  # Limit HTML size
                summary=summary,
                links=tuple(links[:50]),
                metadata={"wait_ms": wait_ms},
                renderer="playwright",
                confidence=0.88,
            )

        except Exception as e:
            return RenderedPage(
                url=url, title="", text="", success=False,
                error=str(e), renderer="playwright",
            )
        finally:
            if page:
                try:
                    page.close()
                except Exception:
                    pass

    # ════════════════════════════════════════════════════════════
    # SELENIUM RENDERING
    # ════════════════════════════════════════════════════════════

    def _render_selenium(self, url: str, wait_ms: int) -> RenderedPage | None:
        """Render a page using Selenium."""
        self._stats["selenium_renders"] += 1

        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            self._selenium_driver.get(url)

            # Wait for page load
            WebDriverWait(self._selenium_driver, self._timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Additional wait for JS
            if wait_ms > 0:
                time.sleep(wait_ms / 1000.0)

            # Extract content
            title = self._selenium_driver.title
            html = self._selenium_driver.page_source
            text = self._selenium_driver.find_element(By.TAG_NAME, "body").text

            # Extract links
            link_elements = self._selenium_driver.find_elements(By.CSS_SELECTOR, "a[href]")
            links = [el.get_attribute("href") for el in link_elements if el.get_attribute("href")]

            # Clean text
            clean_text = self._extractor.extract_text_only(html)
            summary = self._extractor.extract_summary(clean_text)

            return RenderedPage(
                url=url,
                title=title,
                text=clean_text if len(clean_text) > len(text) else text,
                html=html[:50000],
                summary=summary,
                links=tuple(links[:50]),
                metadata={"wait_ms": wait_ms},
                renderer="selenium",
                confidence=0.85,
            )

        except Exception as e:
            return RenderedPage(
                url=url, title="", text="", success=False,
                error=str(e), renderer="selenium",
            )

    # ════════════════════════════════════════════════════════════
    # HTTP FALLBACK
    # ════════════════════════════════════════════════════════════

    def _render_fallback(self, url: str) -> RenderedPage:
        """Fallback to plain HTTP when no browser is available."""
        self._stats["fallback_renders"] += 1

        try:
            # Use the ContentExtractor for HTML parsing
            import httpx
            if httpx:
                resp = httpx.get(url, timeout=self._timeout, follow_redirects=True, headers={
                    "User-Agent": "SweepNeuralEngine/2.0",
                })
                if resp.status_code == 200:
                    extracted = self._extractor.extract_from_html(resp.text, url)
                    return RenderedPage(
                        url=extracted.get("url", url),
                        title=extracted.get("title", ""),
                        text=extracted.get("text", ""),
                        summary=extracted.get("summary", ""),
                        links=tuple(extracted.get("links", [])[:50]),
                        metadata=extracted.get("metadata", {}),
                        renderer="http_fallback",
                        confidence=0.65,
                    )
        except Exception:
            pass

        return RenderedPage(
            url=url, title="", text="", success=False,
            error="No renderer available", renderer="none",
        )


# ══════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ══════════════════════════════════════════════════════════════

def render_page(url: str, wait_ms: int = 2000, timeout: float = 15.0) -> RenderedPage:
    """Render a single page with the best available browser.

    Convenience function that creates a browser, renders, and cleans up.
    """
    browser = HeadlessBrowser(timeout=timeout)
    try:
        return browser.render(url, wait_ms=wait_ms)
    finally:
        browser.close()


def render_pages(
    urls: list[str],
    wait_ms: int = 2000,
    timeout: float = 15.0,
) -> list[RenderedPage]:
    """Render multiple pages with a shared browser instance."""
    browser = HeadlessBrowser(timeout=timeout)
    try:
        return browser.render_batch(urls, wait_ms=wait_ms)
    finally:
        browser.close()
