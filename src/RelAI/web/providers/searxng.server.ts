/**
 * RelAI — SearXNG Search Provider (server-only).
 *
 * Self-hosted meta search engine that aggregates results from multiple
 * sources. Provides clean JSON output.
 *
 * Environment:
 *   SEARXNG_BASE_URL  - SearXNG instance URL (e.g., "http://localhost:8888")
 *
 * Architecture:
 *   SearXNG aggregates: Google, Bing, DuckDuckGo, Wikipedia, and many
 *   more engines. Results are returned as clean JSON.
 */
import type { RelAIWebHit } from "../search.server";

export interface SearXNGOptions {
  limit?: number;
  engines?: string[];
  categories?: string[];
  language?: string;
  format?: "json";
}

/** Check if SearXNG is configured. */
export function searxngConfigured(): boolean {
  return Boolean(process.env.SEARXNG_BASE_URL);
}

function searxngBaseUrl(): string {
  return (process.env.SEARXNG_BASE_URL ?? "http://localhost:8888").replace(/\/$/, "");
}

/**
 * Search via a self-hosted SearXNG instance.
 */
export async function searxngSearch(
  query: string,
  opts: SearXNGOptions = {},
): Promise<{
  hits: RelAIWebHit[];
  engine: string;
  /** Set when the provider was unreachable or failed — NOT "no results". */
  error?: string;
}> {
  if (!searxngConfigured()) {
    return { hits: [], engine: "searxng", error: "searxng: SEARXNG_BASE_URL not set" };
  }

  const baseUrl = searxngBaseUrl();
  const params = new URLSearchParams({
    q: query,
    format: opts.format ?? "json",
    language: opts.language ?? "en",
    ...(opts.limit ? { number_of_results: String(opts.limit) } : {}),
  });

  // Add specific engines if requested
  if (opts.engines) {
    params.set("engines", opts.engines.join(","));
  }
  if (opts.categories) {
    params.set("categories", opts.categories.join(","));
  }

  try {
    const res = await fetch(`${baseUrl}/search?${params.toString()}`, {
      signal: AbortSignal.timeout(15_000),
      headers: {
        Accept: "application/json",
      },
    });

    if (!res.ok) {
      return { hits: [], engine: "searxng", error: `searxng: HTTP ${res.status}` };
    }

    const json = (await res.json()) as {
      results?: Array<{
        url: string;
        title: string;
        content?: string;
        engine?: string;
        score?: number;
        publishedDate?: string;
      }>;
      answers?: string[];
      infoboxes?: Array<{ infobox: string }>;
    };

    const hits: RelAIWebHit[] = (json.results ?? []).map((r) => ({
      url: r.url ?? "",
      title: r.title ?? "",
      snippet: r.content ?? "",
      engine: `searxng/${r.engine ?? "web"}`,
    }));

    return { hits, engine: "searxng" };
  } catch (err) {
    console.warn("[SearXNG] Search failed:", err);
    return {
      hits: [],
      engine: "searxng",
      error: `searxng: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}
