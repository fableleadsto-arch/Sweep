/**
 * Surf search router tests.
 *
 * The router must never hit the network in these tests: every provider's
 * `search()` dynamically imports a RelAI search module, and those modules are
 * mocked below so provider selection + merge/dedupe can be asserted
 * deterministically.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

const mockRelaiSearch = vi.fn(async () => ({
  hits: [
    { url: "https://keyless.example.com/a", title: "Keyless A", snippet: "ka", engine: "keyless" },
    { url: "https://keyless.example.com/b", title: "Keyless B", snippet: "kb", engine: "keyless" },
  ],
  engine: "keyless",
  query: "",
  tried: 2,
  errors: [] as string[],
}));

const mockTavilySearch = vi.fn(async () => ({
  hits: [
    { url: "https://tavily.example.com/1", title: "Tavily 1", snippet: "t1", engine: "tavily" },
    { url: "https://tavily.example.com/2", title: "Tavily 2", snippet: "t2", engine: "tavily" },
  ],
  provider: "tavily",
  blocked: false,
}));

const mockExaSearch = vi.fn(async () => ({
  hits: [
    { url: "https://exa.example.com/1", title: "Exa 1", snippet: "e1", engine: "exa" },
    { url: "https://tavily.example.com/1", title: "Exa dup of tavily", snippet: "t1", engine: "exa" },
  ],
  provider: "exa",
  blocked: false,
}));

const mockSearxngSearch = vi.fn(async () => ({ hits: [], provider: "searxng", blocked: true }));
const mockJinaSearch = vi.fn(async () => ({ hits: [], provider: "jina", blocked: true, error: "jina: HTTP 429 (rate limited)" }));

vi.mock("@/RelAI/web/search.server", () => ({ relaiSearch: mockRelaiSearch }));
vi.mock("@/RelAI/web/providers/tavily.server", () => ({ tavilySearch: mockTavilySearch }));
vi.mock("@/RelAI/web/providers/exa.server", () => ({ exaSearch: mockExaSearch }));
vi.mock("@/RelAI/web/providers/searxng.server", () => ({ searxngSearch: mockSearxngSearch }));
vi.mock("@/RelAI/web/providers/jina.server", () => ({ jinaSearch: mockJinaSearch }));

import { routePriority, routeSearch, runDeepSearch } from "@/surf/search/router";

const ALL_KEYS = {
  TAVILY_API_KEY: "test-tavily",
  EXA_API_KEY: "test-exa",
  SEARXNG_BASE_URL: "https://searxng.invalid",
};

function clearKeys() {
  delete process.env.TAVILY_API_KEY;
  delete process.env.EXA_API_KEY;
  delete process.env.SEARXNG_BASE_URL;
}

describe("routePriority", () => {
  it("orders keyed APIs before the keyless HTML fallback for general intent", () => {
    expect(routePriority("general")).toEqual(["tavily", "exa", "keyless"]);
  });

  it("puts Exa first for technical research", () => {
    expect(routePriority("technical")).toEqual(["exa", "tavily", "keyless"]);
  });

  it("puts Tavily first for news", () => {
    expect(routePriority("news")).toEqual(["tavily", "exa", "keyless"]);
  });

  it("fans out across every provider for deep searches", () => {
    expect(routePriority("deep")).toEqual(["exa", "tavily", "searxng", "jina", "keyless"]);
  });

  it("keeps keyless first for platform-scoped searches (adapters win)", () => {
    expect(routePriority("platform")).toEqual(["keyless", "exa", "tavily"]);
  });

  it("defaults unknown intents to general", () => {
    expect(routePriority("mystery" as never)).toEqual(["tavily", "exa", "keyless"]);
  });
});

describe("routeSearch", () => {
  beforeEach(() => {
    clearKeys();
    vi.clearAllMocks();
  });

  it("uses Tavily for general intent when keyed providers are configured", async () => {
    process.env.TAVILY_API_KEY = ALL_KEYS.TAVILY_API_KEY;
    process.env.EXA_API_KEY = ALL_KEYS.EXA_API_KEY;
    const res = await routeSearch({ query: "crm for agencies", intent: "general" });
    expect(res.provider).toBe("tavily");
    expect(res.results).toHaveLength(2);
  });

  it("uses Exa for technical intent when both keyed providers are configured", async () => {
    process.env.TAVILY_API_KEY = ALL_KEYS.TAVILY_API_KEY;
    process.env.EXA_API_KEY = ALL_KEYS.EXA_API_KEY;
    const res = await routeSearch({ query: "postgres index tuning", intent: "technical" });
    expect(res.provider).toBe("exa");
  });

  it("falls back to the keyless HTML chain when no API keys are configured", async () => {
    const res = await routeSearch({ query: "anything", intent: "general" });
    expect(res.provider).toBe("keyless");
    expect(res.results.length).toBeGreaterThan(0);
    expect(mockRelaiSearch).toHaveBeenCalledOnce();
    expect(mockTavilySearch).not.toHaveBeenCalled();
    expect(mockExaSearch).not.toHaveBeenCalled();
  });

  it("never picks an unconfigured provider over a configured fallback", async () => {
    // Only Exa configured: general intent prefers Tavily, but Tavily is absent,
    // so Exa must win — NOT the keyless chain.
    process.env.EXA_API_KEY = ALL_KEYS.EXA_API_KEY;
    const res = await routeSearch({ query: "x", intent: "general" });
    expect(res.provider).toBe("exa");
    expect(mockExaSearch).toHaveBeenCalledOnce();
  });

  it("keeps keyless first for platform-scoped searches even when API keys are set", async () => {
    process.env.TAVILY_API_KEY = ALL_KEYS.TAVILY_API_KEY;
    process.env.EXA_API_KEY = ALL_KEYS.EXA_API_KEY;
    const res = await routeSearch({ query: "reddit automation pain", intent: "platform" });
    expect(res.provider).toBe("keyless");
    expect(mockRelaiSearch).toHaveBeenCalledOnce();
    expect(mockTavilySearch).not.toHaveBeenCalled();
    expect(mockExaSearch).not.toHaveBeenCalled();
  });
});

describe("runDeepSearch", () => {
  beforeEach(() => {
    clearKeys();
    vi.clearAllMocks();
  });

  it("merges results across configured providers and dedupes by URL", async () => {
    process.env.TAVILY_API_KEY = ALL_KEYS.TAVILY_API_KEY;
    process.env.EXA_API_KEY = ALL_KEYS.EXA_API_KEY;
    const res = await runDeepSearch("deep query", { limit: 10 });

    // Exa first (deep priority): Exa 1 + dup of Tavily 1. Then Tavily: 1 is
    // already seen, 2 is new. Jina + keyless still run (always configured) and
    // add more unique URLs.
    expect(res.providersUsed).toContain("exa");
    expect(res.providersUsed).toContain("tavily");
    expect(res.deduped).toBeGreaterThanOrEqual(1);
    const urls = res.results.map((r) => r.url);
    expect(new Set(urls).size).toBe(urls.length);
    expect(urls).toContain("https://exa.example.com/1");
    expect(urls).toContain("https://tavily.example.com/2");
  });

  it("honors the limit and stops merging once it is reached", async () => {
    process.env.TAVILY_API_KEY = ALL_KEYS.TAVILY_API_KEY;
    process.env.EXA_API_KEY = ALL_KEYS.EXA_API_KEY;
    const res = await runDeepSearch("limited", { limit: 2 });
    expect(res.results).toHaveLength(2);
    // A single provider can fill the quota — the merge must not overshoot.
    expect(res.providersUsed.length).toBeGreaterThanOrEqual(1);
    expect(res.provider.split("+").length).toBe(res.providersUsed.length);
  });

  it("reports blocked when no provider returns anything", async () => {
    mockRelaiSearch.mockResolvedValueOnce({
      hits: [],
      engine: "keyless",
      query: "",
      tried: 0,
      errors: ["No results from any keyless engine"],
    });
    // No API keys: only jina + keyless are configured, and both return empty.
    // Jina is always configured — that is what makes this path reachable.
    const res = await runDeepSearch("empty world", { limit: 10 });
    expect(res.blocked).toBe(true);
    expect(res.results).toHaveLength(0);
    expect(res.deduped).toBe(0);
    expect(res.providersUsed).toEqual(["jina", "keyless"]);
  });

  it("a failing provider does not fail the whole deep merge", async () => {
    process.env.TAVILY_API_KEY = ALL_KEYS.TAVILY_API_KEY;
    mockExaSearch.mockRejectedValueOnce(new Error("exa outage"));
    const res = await runDeepSearch("resilient", { limit: 10 });
    expect(res.results.length).toBeGreaterThan(0);
    expect(res.providersUsed).toContain("tavily");
  });

  it("surfaces per-provider errors instead of collapsing them into empty results", async () => {
    const res = await runDeepSearch("transparent", { limit: 10 });
    // jina always fails with a rate-limit error and keyless is blocked here too.
    expect(res.errors?.some((e) => e.includes("jina"))).toBe(true);
    expect(res.errors?.some((e) => e.includes("429"))).toBe(true);
  });

  it("records a throwing provider's failure reason in the error list", async () => {
    process.env.EXA_API_KEY = ALL_KEYS.EXA_API_KEY;
    mockExaSearch.mockRejectedValueOnce(new Error("exa outage"));
    const res = await runDeepSearch("throwing", { limit: 10 });
    expect(res.errors?.some((e) => e.includes("exa") && e.includes("outage"))).toBe(true);
    expect(res.errors?.some((e) => e.includes("jina"))).toBe(true);
  });

  it("passes the SERP page through to the keyless chain", async () => {
    await routeSearch({ query: "page me", intent: "general", options: { page: 2 } });
    expect(mockRelaiSearch).toHaveBeenCalledWith(
      "page me",
      expect.objectContaining({ page: 2 }),
    );
  });

  it("serves repeat identical searches from the cache without re-fetching", async () => {
    await runDeepSearch("cached query", { limit: 10 });
    await runDeepSearch("cached query", { limit: 10 });
    // Only jina (blocked, never cached) re-runs; keyless is served from cache.
    expect(mockRelaiSearch).toHaveBeenCalledTimes(1);
  });

  it("never caches a blocked run so a transient failure cannot linger", async () => {
    mockRelaiSearch.mockResolvedValueOnce({
      hits: [],
      engine: "none",
      query: "",
      tried: 0,
      errors: ["boom"],
    });
    await runDeepSearch("flaky", { limit: 10 });
    await runDeepSearch("flaky", { limit: 10 });
    expect(mockRelaiSearch).toHaveBeenCalledTimes(2);
  });
});
