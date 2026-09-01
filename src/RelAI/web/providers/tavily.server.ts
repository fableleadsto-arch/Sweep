/**
 * RelAI — Tavily Search Provider (server-only).
 *
 * AI-native search API that provides high-quality web search results
 * with optional content extraction and crawling.
 *
 * Environment:
 *   TAVILY_API_KEY  - Tavily API key
 *   TAVILY_SEARCH_DEPTH - "basic" or "deep" (default: "basic")
 *
 * Package: @tavily/core
 */
import type { RelAIWebHit } from "../search.server";

export interface TavilySearchOptions {
  limit?: number;
  // Matches @tavily/core v0.7: "deep" was renamed in the type union.
  depth?: "basic" | "advanced" | "fast" | "ultra-fast";
  includeAnswer?: boolean;
  includeImages?: boolean;
  includeRawContent?: boolean;
  maxTokens?: number;
}

export interface TavilySearchResult {
  hits: RelAIWebHit[];
  answer?: string;
  engine: string;
  /** Set when the provider was unreachable or failed — NOT "no results". */
  error?: string;
}

/** Check if Tavily is configured. */
export function tavilyConfigured(): boolean {
  return Boolean(process.env.TAVILY_API_KEY);
}

/**
 * Search the web using Tavily.
 */
export async function tavilySearch(
  query: string,
  opts: TavilySearchOptions = {},
): Promise<TavilySearchResult> {
  if (!tavilyConfigured()) {
    return { hits: [], engine: "tavily", error: "tavily: TAVILY_API_KEY not set" };
  }

  try {
    const { tavily } = await import("@tavily/core");
    const tvly = tavily({ apiKey: process.env.TAVILY_API_KEY });

    const response = await tvly.search(query, {
      searchDepth: opts.depth ?? "basic",
      maxResults: opts.limit ?? 10,
      includeAnswer: opts.includeAnswer ?? true,
      includeImages: opts.includeImages ?? false,
      // v0.7 types raw-content as a format selector (false | "text" | "markdown").
      includeRawContent: opts.includeRawContent ? "text" : false,
    });

    const hits: RelAIWebHit[] = (response.results ?? []).map((r: any) => ({
      url: r.url ?? "",
      title: r.title ?? "",
      snippet: r.content ?? "",
      engine: "tavily",
      relevance: r.score,
    }));

    return {
      hits,
      answer: response.answer,
      engine: "tavily",
    };
  } catch (err) {
    console.warn("[Tavily] Search failed:", err);
    return {
      hits: [],
      engine: "tavily",
      error: `tavily: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

/**
 * Extract content from specific URLs using Tavily.
 */
export async function tavilyExtract(
  urls: string[],
): Promise<Array<{ url: string; content: string }>> {
  if (!tavilyConfigured() || urls.length === 0) return [];

  try {
    const { tavily } = await import("@tavily/core");
    const tvly = tavily({ apiKey: process.env.TAVILY_API_KEY });
    const result = await tvly.extract(urls);
    return (result.results ?? []).map((r: any) => ({
      url: r.url ?? "",
      content: r.rawContent ?? r.content ?? "",
    }));
  } catch {
    return [];
  }
}
