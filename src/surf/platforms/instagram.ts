/**
 * Relay Surf — Instagram adapter (server-only).
 *
 * Instagram requires authentication for native search and blocks most
 * automated access. This adapter supports only publicly indexed information:
 *   - search: public web indexing of instagram.com, honestly labeled.
 *   - extract: profiles/posts reachable through public indexing when the fetch
 *     layer is not blocked (usually it is). Never bypasses login/CAPTCHA.
 */
import { assertSafeUrl } from "../guard";
import type { PlatformAdapter, SearchResult } from "../types";

function igHost(url: string): boolean {
  try {
    return new URL(url).hostname.replace(/^www\./, "") === "instagram.com";
  } catch {
    return false;
  }
}

export const instagramAdapter: PlatformAdapter = {
  platform: "instagram",
  canHandle: igHost,
  async search(query, options = {}) {
    const { relaiSearch } = await import("@/RelAI/web/search.server");
    const limit = options.limit ?? 8;
    const res = await relaiSearch(query, { site: "instagram.com", limit: limit * 2 });
    const results: SearchResult[] = res.hits
      .filter((h) => igHost(h.url))
      .slice(0, limit)
      .map((h) => ({
        url: h.url,
        title: h.title,
        snippet: h.snippet,
        provider: "web-index",
        accessMode: "public",
        platform: "instagram",
      }));
    return {
      results,
      accessMode: "public",
      note:
        "Instagram native search requires authentication. Results come from public web indexing — this is not native Instagram search.",
    };
  },
  async extractPage(url) {
    const safeUrl = assertSafeUrl(url);
    if (!igHost(safeUrl)) return null;
    const { relaiReadPage } = await import("@/RelAI/web/search.server");
    try {
      const page = await relaiReadPage(safeUrl, { maxChars: 3000 });
      if (page.status >= 400 || !page.text.trim()) return null;
      return {
        url: page.url,
        title: page.title,
        text: page.text.slice(0, 3000),
        links: [],
        headings: [],
        metadata: {},
        truncated: page.truncated,
        fetchedAt: new Date().toISOString(),
        status: page.status,
        contentType: "text/html",
        accessMode: "public",
      };
    } catch {
      return null;
    }
  },
};
