/**
 * RelAI — Jina Reader & Search Provider (server-only).
 *
 * Jina provides two powerful services via simple HTTP:
 *   - r.jina.ai: Convert any URL to LLM-ready Markdown
 *   - s.jina.ai: AI-powered web search with grounded results
 *
 * Environment:
 *   JINA_API_KEY  - Optional Jina API key (for higher rate limits)
 */
import type { RelAIWebHit } from "../search.server";

/** Check if Jina is configured (returns true even without API key — rate limited). */
export function jinaConfigured(): boolean {
  return true; // Jina has a free tier without API key
}

/**
 * Read a URL and return clean markdown content via Jina Reader.
 */
export async function jinaReadUrl(
  url: string,
): Promise<{ title?: string; content: string; url: string } | null> {
  try {
    const headers: Record<string, string> = {
      "X-Respond-With": "markdown",
      "Accept": "text/markdown",
    };
    if (process.env.JINA_API_KEY) {
      headers["Authorization"] = `Bearer ${process.env.JINA_API_KEY}`;
    }

    const targetUrl = `https://r.jina.ai/${encodeURIComponent(url)}`;
    const res = await fetch(targetUrl, {
      headers,
      signal: AbortSignal.timeout(30_000),
    });

    if (!res.ok) return null;

    const text = await res.text();
    return {
      content: text,
      url: url,
    };
  } catch {
    return null;
  }
}

export interface JinaSearchOptions {
  limit?: number;
}

export interface JinaSearchResult {
  hits: RelAIWebHit[];
  answer?: string;
  engine: string;
  /** Set when the provider was unreachable or rate-limited — NOT "no results". */
  error?: string;
}

/**
 * Search the web using Jina AI search (s.jina.ai).
 * Returns grounded results with citations.
 */
export async function jinaSearch(
  query: string,
  opts: JinaSearchOptions = {},
): Promise<JinaSearchResult> {
  try {
    const headers: Record<string, string> = {};
    if (process.env.JINA_API_KEY) {
      headers["Authorization"] = `Bearer ${process.env.JINA_API_KEY}`;
    }

    const searchUrl = `https://s.jina.ai/${encodeURIComponent(query)}`;
    const res = await fetch(searchUrl, {
      headers,
      signal: AbortSignal.timeout(30_000),
    });

    if (!res.ok) {
      return {
        hits: [],
        engine: "jina",
        error: `jina: HTTP ${res.status}${res.status === 429 ? " (rate limited)" : ""}`,
      };
    }

    const text = await res.text();
    const hits: RelAIWebHit[] = [];
    const limit = opts.limit ?? 10;

    // Parse Jina's markdown response. It typically looks like:
    // ## [Title](url)
    // snippet text
    // ...
    const lines = text.split("\n");
    let currentTitle = "";
    let currentUrl = "";
    let currentSnippet = "";

    for (const line of lines) {
      const matches = [...line.matchAll(/\[([^\]]+)\]\(([^)]+)\)/g)];
      if (matches.length > 0) {
        // Save previous hit
        if (currentUrl && hits.length < limit) {
          hits.push({
            url: currentUrl,
            title: currentTitle,
            snippet: currentSnippet.slice(0, 300),
            engine: "jina",
          });
        }
        const match = matches[0];
        currentTitle = match[1];
        currentUrl = match[2];
        currentSnippet = line.replace(/\[([^\]]+)\]\(([^)]+)\)/g, "").trim();
      } else if (currentUrl && line.trim()) {
        currentSnippet += " " + line.trim();
      }
    }

    // Don't forget the last hit
    if (currentUrl && hits.length < limit) {
      hits.push({
        url: currentUrl,
        title: currentTitle,
        snippet: currentSnippet.slice(0, 300),
        engine: "jina",
      });
    }

    // First line often contains the AI answer
    const answer = lines[0]?.trim().startsWith("[") ? undefined : lines[0]?.trim();

    return {
      hits,
      answer,
      engine: "jina",
    };
  } catch (err) {
    return {
      hits: [],
      engine: "jina",
      error: `jina: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}
