/**
 * Relay Surf — search provider routing (server-only).
 *
 * Decides which provider(s) satisfy a search request based on intent, rather
 * than hard-coding one provider into Relay Brain. Routing is deterministic and
 * transparent: every run records which provider was used and why, so the UI can
 * honestly say "searched via Exa" instead of "searched the web".
 *
 * Routing rules (checked in order):
 *   - technical research      → Exa (embeddings-based) when configured, else keyless
 *   - news                    → news-capable provider (tavily/exa), else keyless
 *   - platform-scoped search  → platform adapters (see ../platforms), never claimed
 *   - default                 → keyless (multi-engine HTML), always available
 *   - deep                    → multiple providers, merged + deduped
 */
import type { SearchOptions, SearchRunResult, SearchResult } from "../types";
import { getProvider } from "./providers";

export type SearchIntent = "general" | "technical" | "news" | "deep" | "platform";

export interface RouteSearchInput {
  query: string;
  intent?: SearchIntent;
  options?: SearchOptions;
  /** e.g. "reddit" | "github" — platform searches are routed to adapters. */
  platform?: string;
}

const KEYLESS = "keyless";

/** Deterministic provider priority per intent (exported for tests + UI hints). */
export function routePriority(intent: SearchIntent): string[] {
  switch (intent) {
    case "technical":
      return ["exa", "tavily", KEYLESS];
    case "news":
      return ["tavily", "exa", KEYLESS];
    case "deep":
      // Multiple providers — resolved by the caller via `runDeep`.
      return ["exa", "tavily", "searxng", "jina", KEYLESS];
    case "platform":
      return [KEYLESS, "exa", "tavily"];
    case "general":
    default:
      // Keyed API providers first: HTML engines (keyless) are the fallback.
      // Datacenter egress IPs (Railway) get blocked by DDG/Brave/Bing, so the
      // HTML path is slow and flaky there — Tavily/Exa are reliable and return
      // far better results when a key is configured.
      return ["tavily", "exa", KEYLESS];
  }
}

/** Single-provider search with routing and fallback. */
export async function routeSearch(input: RouteSearchInput): Promise<SearchRunResult> {
  const priority = routePriority(input.intent ?? "general");
  const errors: string[] = [];

  for (const name of priority) {
    const provider = getProvider(name);
    if (!provider) continue;
    try {
      const run = await provider.search(input.query, input.options ?? {});
      // If the provider returned results (or was blocked), use this run.
      if (run.results.length > 0 || run.blocked) {
        return {
          provider: provider.name,
          query: input.query,
          results: run.results,
          blocked: run.blocked,
          note: run.note,
          errors: run.errors,
        };
      }
      // Zero results but no error — record and try next provider.
      errors.push(`${provider.name}: 0 results`);
    } catch (err) {
      errors.push(`${provider.name}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  return {
    provider: priority.find((n) => getProvider(n)) ?? KEYLESS,
    query: input.query,
    results: [],
    blocked: true,
    note: "No provider returned results",
    errors: errors.length > 0 ? errors : undefined,
  };
}

/** Merge providers for deep searches; dedupes by canonical URL. */
export async function runDeepSearch(
  query: string,
  options: SearchOptions = {},
): Promise<SearchRunResult & { providersUsed: string[]; deduped: number }> {
  const used: string[] = [];
  const seen = new Set<string>();
  const merged: SearchResult[] = [];
  const errors: string[] = [];
  let dropped = 0;
  const limit = options.limit ?? 10;

  for (const name of routePriority("deep")) {
    const provider = getProvider(name);
    if (!provider) continue;
    try {
      const run = await provider.search(query, options);
      used.push(provider.name);
      for (const error of run.errors ?? []) {
        if (!errors.includes(error)) errors.push(error);
      }
      for (const hit of run.results) {
        const key = hit.url.split("#")[0].replace(/\/+$/, "");
        if (seen.has(key)) {
          dropped++;
          continue;
        }
        seen.add(key);
        merged.push(hit);
        if (merged.length >= limit) break;
      }
    } catch (err) {
      // A failing provider in a deep merge is not fatal — surface why.
      errors.push(`${name}: ${err instanceof Error ? err.message : String(err)}`);
      continue;
    }
    if (merged.length >= limit) break;
  }

  return {
    provider: used.join("+") || KEYLESS,
    query,
    results: merged,
    blocked: merged.length === 0,
    providersUsed: used,
    deduped: dropped,
    errors: errors.length > 0 ? errors : undefined,
    note: used.length === 0 ? "No configured provider returned results" : undefined,
  };
}
