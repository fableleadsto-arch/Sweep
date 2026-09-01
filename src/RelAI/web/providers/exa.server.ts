/**
 * RelAI — Exa Search Provider (server-only).
 *
 * Embeddings-based search engine that understands meaning, not just keywords.
 * Provides search, content retrieval, and answer generation.
 *
 * Environment:
 *   EXA_API_KEY  - Exa API key
 *
 * Package: exa-js
 */
import type { RelAIWebHit } from "../search.server";

export interface ExaSearchOptions {
  limit?: number;
  // Matches the exa-js v2 type: the legacy "magic" value is no longer valid.
  type?: "auto" | "keyword" | "instant" | "fast" | "neural" | "hybrid";
  includeHighlights?: boolean;
  includeAnswer?: boolean;
  startPublishedDate?: string;
  endPublishedDate?: string;
}

export interface ExaSearchResult {
  hits: RelAIWebHit[];
  answer?: string;
  engine: string;
  /** Set when the provider was unreachable or failed — NOT "no results". */
  error?: string;
}

/** Check if Exa is configured. */
export function exaConfigured(): boolean {
  return Boolean(process.env.EXA_API_KEY);
}

/**
 * Search the web using Exa.
 */
export async function exaSearch(
  query: string,
  opts: ExaSearchOptions = {},
): Promise<ExaSearchResult> {
  if (!exaConfigured()) {
    return { hits: [], engine: "exa", error: "exa: EXA_API_KEY not set" };
  }

  try {
    const Exa = (await import("exa-js")).default;
    const exa = new Exa(process.env.EXA_API_KEY);

    // Shared options — branched below so each search call matches a single
    // overload (a union `contents` value would match none of them).
    const baseOpts = {
      type: opts.type ?? "auto",
      numResults: opts.limit ?? 10,
      startPublishedDate: opts.startPublishedDate,
      endPublishedDate: opts.endPublishedDate,
    };

    // exa-js v2 moved highlights under `contents`.
    const result =
      opts.includeHighlights === false
        ? await exa.search(query, { ...baseOpts, contents: false })
        : await exa.search(query, {
            ...baseOpts,
            contents: { highlights: { maxCharacters: 300 } },
          });

    const hits: RelAIWebHit[] = (result.results ?? []).map((r: any) => ({
      url: r.url ?? "",
      title: r.title ?? "",
      snippet: Array.isArray(r.highlights)
        ? r.highlights
            .map((h: any) => (typeof h === "string" ? h : h?.text ?? ""))
            .filter(Boolean)
            .join(" ")
        : r.text?.slice(0, 300) ?? "",
      engine: "exa",
      relevance: r.score,
    }));

    return {
      hits,
      engine: "exa",
    };
  } catch (err) {
    console.warn("[Exa] Search failed:", err);
    return {
      hits: [],
      engine: "exa",
      error: `exa: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

/**
 * Get an AI-generated answer with citations from Exa.
 */
export async function exaAnswer(query: string): Promise<{
  answer: string;
  citations: Array<{ url: string; title: string }>;
}> {
  if (!exaConfigured()) {
    return { answer: "", citations: [] };
  }

  try {
    const Exa = (await import("exa-js")).default;
    const exa = new Exa(process.env.EXA_API_KEY);

    const result = await exa.answer(query);
    return {
      // answer can be a plain string or a structured object in v2 — only
      // surface the string form.
      answer: typeof result.answer === "string" ? result.answer : "",
      citations: (result.citations ?? []).map((c: any) => ({
        url: c.url ?? "",
        title: c.title ?? "",
      })),
    };
  } catch {
    return { answer: "", citations: [] };
  }
}

/**
 * Get full contents of specific URLs from Exa.
 */
export async function exaGetContents(
  urls: string[],
): Promise<Array<{ url: string; content: string; title: string }>> {
  if (!exaConfigured() || urls.length === 0) return [];

  try {
    const Exa = (await import("exa-js")).default;
    const exa = new Exa(process.env.EXA_API_KEY);

    const result = await exa.getContents(urls, {
      text: { maxCharacters: 5000 },
    });

    return (result.results ?? []).map((r: any) => ({
      url: r.url ?? "",
      content: r.text ?? "",
      title: r.title ?? "",
    }));
  } catch {
    return [];
  }
}
