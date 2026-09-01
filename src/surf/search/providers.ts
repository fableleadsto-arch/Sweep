/**
 * Relay Surf — search provider abstraction (server-only).
 *
 * Implements the `SearchProvider` contract from ../types over every search
 * surface Relay actually has configured. Providers are discovered at call time
 * and NEVER fabricated: an unconfigured key simply does not produce a provider.
 *
 * Every provider caches successful runs through the shared research cache
 * (`../cache`), so repeat searches for the same public query skip the network.
 * Blocked/failed runs are never cached — a transient 403 must not poison the
 * cache. The lead finder shares the same cache via its own adapter.
 *
 * Provider list (in default routing order):
 *   keyless   — RelAI's multi-engine HTML search (DuckDuckGo/Brave/Bing/Mojeek)
 *               with API fallbacks; always available, no key needed.
 *   tavily    — when TAVILY_API_KEY is set.
 *   exa       — when EXA_API_KEY is set.
 *   searxng   — when SEARXNG_BASE_URL is set (self-hosted meta search).
 *   jina      — free-tier reader/search.
 */
import type { SearchProvider, SearchResult, SearchRun } from "../types";
import { cacheGet, cacheKey, cacheSet } from "../cache";

export type { SearchRun } from "../types";

/** Short probe query used by every provider's on-demand health check. */
const PROBE_QUERY = "technology startups funding";
const PROBE_LIMIT = 3;

/** Run a health probe with a hard timeout so a hung provider can't block a run. */
async function probeHealth(search: () => Promise<{ hits: ArrayLike<unknown>; error?: string }>): Promise<boolean> {
  const res = await withProbeTimeout(search(), 10_000);
  if (!res.error && res.hits.length > 0) return true;
  throw new Error(res.error || "provider returned no results");
}

function withProbeTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("health probe timed out")), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      },
    );
  });
}

/* ------------------------------------------------------------------ */
/*  Adapters over the existing RelAI provider modules                  */
/* ------------------------------------------------------------------ */

function mapRelAIHit(hit: { url: string; title: string; snippet?: string; engine: string; relevance?: number }): SearchResult {
  return {
    url: hit.url,
    title: hit.title,
    snippet: hit.snippet ?? "",
    provider: hit.engine,
    accessMode: "public",
  };
}

/** Cache a successful provider run; blocked runs are never cached. */
function withSuccessCache(key: string, produce: () => Promise<SearchRun>): Promise<SearchRun> {
  const cached = cacheGet<SearchRun>(key);
  if (cached) return Promise.resolve(cached);
  return produce().then((run) => {
    if (!run.blocked) cacheSet(key, run);
    return run;
  });
}

const keyless: SearchProvider = {
  name: "keyless",
  configured: () => true,
  async health() {
    const { relaiSearch } = await import("@/RelAI/web/search.server");
    return probeHealth(() => relaiSearch(PROBE_QUERY, { limit: PROBE_LIMIT, rank: false }));
  },
  async search(query, options = {}) {
    const { relaiSearch } = await import("@/RelAI/web/search.server");
    const limit = options.limit ?? 10;
    const key = cacheKey(
      "surf-search",
      "keyless",
      query,
      limit,
      options.site,
      options.timeRange,
      options.page,
      options.aggregate,
    );
    return withSuccessCache(key, async () => {
      const res = await relaiSearch(query, {
        limit,
        site: options.site,
        page: options.page,
        aggregate: options.aggregate,
        mode: options.timeRange === "day" || options.timeRange === "week" ? "news" : undefined,
      });
      return {
        results: res.hits.map(mapRelAIHit),
        provider: res.engine,
        blocked: res.hits.length === 0,
        errors: res.errors.length > 0 ? res.errors : undefined,
        note: res.hits.length === 0 ? (res.errors[0] ?? "No results from any keyless engine") : undefined,
      };
    });
  },
  capabilities: { siteFilter: true },
};

const tavily: SearchProvider = {
  name: "tavily",
  configured: () => Boolean(process.env.TAVILY_API_KEY),
  async health() {
    const { tavilySearch } = await import("@/RelAI/web/providers/tavily.server");
    return probeHealth(() => tavilySearch(PROBE_QUERY, { limit: PROBE_LIMIT }));
  },
  async search(query, options = {}) {
    const { tavilySearch } = await import("@/RelAI/web/providers/tavily.server");
    const key = cacheKey("surf-search", "tavily", query, options.limit ?? 10, options.site, options.after, options.before);
    return withSuccessCache(key, async () => {
      const res = await tavilySearch(query, {
        limit: options.limit ?? 10,
        includeAnswer: false,
      });
      return {
        results: res.hits.map(mapRelAIHit),
        provider: "tavily",
        blocked: res.hits.length === 0,
        errors: res.error ? [res.error] : undefined,
      };
    });
  },
  capabilities: { siteFilter: true, news: true },
};

const exa: SearchProvider = {
  name: "exa",
  configured: () => Boolean(process.env.EXA_API_KEY),
  async health() {
    const { exaSearch } = await import("@/RelAI/web/providers/exa.server");
    return probeHealth(() => exaSearch(PROBE_QUERY, { limit: PROBE_LIMIT }));
  },
  async search(query, options = {}) {
    const { exaSearch } = await import("@/RelAI/web/providers/exa.server");
    const key = cacheKey("surf-search", "exa", query, options.limit ?? 10, options.after, options.before);
    return withSuccessCache(key, async () => {
      const res = await exaSearch(query, {
        limit: options.limit ?? 10,
        includeHighlights: true,
        startPublishedDate: options.after,
        endPublishedDate: options.before,
      });
      return {
        results: res.hits.map(mapRelAIHit),
        provider: "exa",
        blocked: res.hits.length === 0,
        errors: res.error ? [res.error] : undefined,
      };
    });
  },
  capabilities: { siteFilter: true, technical: true, news: true },
};

const searxng: SearchProvider = {
  name: "searxng",
  configured: () => Boolean(process.env.SEARXNG_BASE_URL),
  async health() {
    const { searxngSearch } = await import("@/RelAI/web/providers/searxng.server");
    return probeHealth(() => searxngSearch(PROBE_QUERY, { limit: PROBE_LIMIT }));
  },
  async search(query, options = {}) {
    const { searxngSearch } = await import("@/RelAI/web/providers/searxng.server");
    const key = cacheKey("surf-search", "searxng", query, options.limit ?? 10);
    return withSuccessCache(key, async () => {
      const res = await searxngSearch(query, { limit: options.limit ?? 10 });
      return {
        results: res.hits.map(mapRelAIHit),
        provider: "searxng",
        blocked: res.hits.length === 0,
        errors: res.error ? [res.error] : undefined,
      };
    });
  },
};

const jina: SearchProvider = {
  name: "jina",
  configured: () => true,
  async health() {
    const { jinaSearch } = await import("@/RelAI/web/providers/jina.server");
    return probeHealth(() => jinaSearch(PROBE_QUERY, { limit: PROBE_LIMIT }));
  },
  async search(query, options = {}) {
    const { jinaSearch } = await import("@/RelAI/web/providers/jina.server");
    const key = cacheKey("surf-search", "jina", query, options.limit ?? 10);
    return withSuccessCache(key, async () => {
      const res = await jinaSearch(query, { limit: options.limit ?? 10 });
      return {
        results: res.hits.map(mapRelAIHit),
        provider: "jina",
        blocked: res.hits.length === 0,
        errors: res.error ? [res.error] : undefined,
      };
    });
  },
};

const ALL_PROVIDERS: SearchProvider[] = [keyless, tavily, exa, searxng, jina];

/* ------------------------------------------------------------------ */
/*  Discovery                                                          */
/* ------------------------------------------------------------------ */

/** Every provider that is actually configured on this deployment. */
export function configuredProviders(): SearchProvider[] {
  return ALL_PROVIDERS.filter((p) => p.configured());
}

/** Look a provider up by name; undefined when it isn't configured. */
export function getProvider(name: string): SearchProvider | undefined {
  const provider = ALL_PROVIDERS.find((p) => p.name === name);
  return provider?.configured() ? provider : undefined;
}

/**
 * Provider capabilities for observability/UI — never claims a provider that
 * isn't configured.
 */
export function providerSummary(): Array<{ name: string; available: boolean; capabilities: string[] }> {
  return ALL_PROVIDERS.map((p) => ({
    name: p.name,
    available: p.configured(),
    capabilities: Object.keys(p.capabilities ?? {}) ?? [],
  }));
}

/** Names of providers that are configured (used by the router and UI). */
export function availableProviderNames(): string[] {
  return configuredProviders().map((p) => p.name);
}

export interface ProviderHealthResult {
  name: string;
  ok: boolean;
  ms: number;
  error?: string;
}

/**
 * Probe every configured provider on demand. Each probe has its own 10s
 * timeout; a provider that can't reach its backend reports `ok: false` with
 * the reason — the UI shows live status instead of assuming config == reachable.
 */
export async function providerHealth(): Promise<ProviderHealthResult[]> {
  const providers = configuredProviders();
  return Promise.all(
    providers.map(async (provider) => {
      const started = Date.now();
      if (!provider.health) {
        return { name: provider.name, ok: true, ms: 0 };
      }
      try {
        const ok = await provider.health();
        return { name: provider.name, ok, ms: Date.now() - started };
      } catch (err) {
        return {
          name: provider.name,
          ok: false,
          ms: Date.now() - started,
          error: err instanceof Error ? err.message : String(err),
        };
      }
    }),
  );
}

export { ALL_PROVIDERS };
