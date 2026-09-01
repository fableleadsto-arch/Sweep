/**
 * Relay Surf — X / Twitter adapter (server-only).
 *
 * X has no public unauthenticated search API and blocks automated access
 * behind login/CAPTCHA. This adapter therefore NEVER claims native X search:
 *   - search: public web indexing of x.com/twitter.com only, clearly labeled.
 *   - extract: graceful — public indexed pages only; blocked pages reported.
 *
 * If an authenticated X connector is configured in the future it plugs in
 * here behind an explicit accessMode: "authenticated" — never via bypasses.
 */
import { assertSafeUrl } from "../guard";
import type { PlatformAdapter, PlatformSearchOptions, SearchResult } from "../types";

function xHost(url: string): boolean {
  try {
    const host = new URL(url).hostname.replace(/^www\./, "");
    return host === "x.com" || host === "twitter.com";
  } catch {
    return false;
  }
}

export const xAdapter: PlatformAdapter = {
  platform: "x",
  canHandle: xHost,
  async search(query, options: PlatformSearchOptions = {}) {
    const { relaiSearch } = await import("@/RelAI/web/search.server");
    const limit = options.limit ?? 8;
    const res = await relaiSearch(query, { site: "x.com OR twitter.com", limit: limit * 2 });
    const results: SearchResult[] = res.hits
      .filter((h) => xHost(h.url))
      .slice(0, limit)
      .map((h) => ({
        url: h.url,
        title: h.title,
        snippet: h.snippet,
        provider: "web-index",
        accessMode: "public",
        platform: "x",
      }));
    return {
      results,
      accessMode: "public",
      note:
        "X search isn't accessible through a configured connector. Results come from public web indexing — this is not native X search.",
    };
  },
  async extractPage(url) {
    const safeUrl = assertSafeUrl(url);
    if (!xHost(safeUrl)) return null;
    // Public indexed pages may still be captcha-walled; do not pretend we read
    // the post. Extract from the web index's cached content only when the
    // plain fetch layer succeeds (it usually won't for x.com).
    const { relaiReadPage } = await import("@/RelAI/web/search.server");
    try {
      const page = await relaiReadPage(safeUrl, { maxChars: 4000 });
      if (page.status >= 400 || !page.text.trim()) return null;
      return {
        url: page.url,
        title: page.title,
        text: page.text.slice(0, 4000),
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
