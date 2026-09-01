/**
 * Relay Surf — generic website adapter (server-only).
 *
 * The fallback adapter for any website without a dedicated integration:
 * standard web search plus full page extraction (semantic PageData with links,
 * headings, metadata, structured data). Also implements native site-search
 * detection — when a page exposes a search form/query parameter, Relay can
 * search within the site using its own mechanism.
 */
import { relaiFetch } from "@/RelAI/web/http.server";
import { assertSafeUrl } from "../guard";
import { extractPageData } from "../browse/extract";
import type { PlatformAdapter, SearchResult } from "../types";

export function detectSiteSearchBase(url: string): { base: string; param: string } | null {
  try {
    const u = new URL(url);
    // Docs sites commonly use these engines.
    const param =
      u.searchParams.get("q") !== null
        ? "q"
        : u.searchParams.get("query") !== null
          ? "query"
          : u.searchParams.get("search") !== null
            ? "search"
            : "q";
    return { base: `${u.origin}${u.pathname}`, param };
  } catch {
    return null;
  }
}

export const genericAdapter: PlatformAdapter = {
  platform: "generic",
  canHandle() {
    return true;
  },
  async search(query, options = {}) {
    const { relaiSearch } = await import("@/RelAI/web/search.server");
    const res = await relaiSearch(query, {
      limit: Math.min(options.limit ?? 8, 20),
      site: options.subreddit,
    });
    return {
      results: res.hits.map((h) => ({
        url: h.url,
        title: h.title,
        snippet: h.snippet,
        provider: h.engine,
        accessMode: "public",
      })),
      accessMode: "public",
      note: res.hits.length === 0 ? "No results from any configured web engine." : undefined,
    };
  },
  async extractPage(url) {
    const safeUrl = assertSafeUrl(url);
    const res = await relaiFetch(safeUrl, { timeoutMs: 15_000, retries: 2, cache: true });
    if (!res.ok) return null;
    const isJson = res.contentType.includes("json") || /^\s*[[{]/.test(res.text.slice(0, 200));
    const page = extractPageData({
      html: isJson ? undefined : res.text,
      json: isJson ? res.text : undefined,
      url: res.url,
      status: res.status,
      contentType: res.contentType,
      maxChars: 16_000,
      fetchedAt: res.fetchedAt,
    });
    return page;
  },
};

/** Native site search within a website when its own search is reachable. */
export async function searchWithinSite(
  siteUrl: string,
  query: string,
  limit = 8,
): Promise<{ results: SearchResult[]; native: boolean }> {
  const safeUrl = assertSafeUrl(siteUrl);
  const candidate = detectSiteSearchBase(safeUrl);
  if (candidate) {
    const searchUrl = `${candidate.base}?${candidate.param}=${encodeURIComponent(query)}`;
    try {
      const res = await relaiFetch(searchUrl, { timeoutMs: 12_000, retries: 1, cache: true });
      if (res.ok && res.text.length > 400) {
        const page = extractPageData({
          html: res.text,
          url: res.url,
          status: res.status,
          contentType: res.contentType,
          maxChars: 16_000,
          fetchedAt: res.fetchedAt,
        });
        const results: SearchResult[] = page.links
          .filter((l) => l.url.startsWith(new URL(safeUrl).origin))
          .slice(0, limit)
          .map((l) => ({ url: l.url, title: l.text || l.url, snippet: "", provider: "site-native", accessMode: "public" }));
        if (results.length > 0) return { results, native: true };
      }
    } catch {
      /* fall through to web-scoped search */
    }
  }
  // Fall back to site-scoped web search, honestly reported.
  const { relaiSearch } = await import("@/RelAI/web/search.server");
  const res = await relaiSearch(query, { site: new URL(safeUrl).hostname, limit });
  return {
    results: res.hits.map((h) => ({
      url: h.url,
      title: h.title,
      snippet: h.snippet,
      provider: "web-index",
      accessMode: "public",
    })),
    native: false,
  };
}
