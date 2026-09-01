/**
 * Surf search provider tests — health probing and cache semantics.
 *
 * Every RelAI search module is mocked so no network or key lookup happens.
 * Cache assertions rely on the shared in-memory cache being fresh per file
 * (vitest isolates each test file).
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

const mockRelaiSearch = vi.fn(async () => ({
  hits: [{ url: "https://keyless.example.com/a", title: "Keyless A", snippet: "ka", engine: "duckduckgo" }],
  engine: "duckduckgo",
  query: "",
  tried: 1,
  errors: [] as string[],
}));

const mockTavilySearch = vi.fn(async () => ({
  hits: [{ url: "https://tavily.example.com/1", title: "Tavily 1", snippet: "t1", engine: "tavily" }],
  provider: "tavily",
  blocked: false,
}));

const mockExaSearch = vi.fn(async () => ({
  hits: [{ url: "https://exa.example.com/1", title: "Exa 1", snippet: "e1", engine: "exa" }],
  provider: "exa",
  blocked: false,
}));

const mockSearxngSearch = vi.fn(async () => ({
  hits: [{ url: "https://searxng.example.com/1", title: "Searxng 1", snippet: "s1", engine: "searxng" }],
  provider: "searxng",
  blocked: false,
}));

const mockJinaSearch = vi.fn(async () => ({
  hits: [],
  provider: "jina",
  blocked: true,
  error: "jina: HTTP 429 (rate limited)",
}));

vi.mock("@/RelAI/web/search.server", () => ({ relaiSearch: mockRelaiSearch }));
vi.mock("@/RelAI/web/providers/tavily.server", () => ({ tavilySearch: mockTavilySearch }));
vi.mock("@/RelAI/web/providers/exa.server", () => ({ exaSearch: mockExaSearch }));
vi.mock("@/RelAI/web/providers/searxng.server", () => ({ searxngSearch: mockSearxngSearch }));
vi.mock("@/RelAI/web/providers/jina.server", () => ({ jinaSearch: mockJinaSearch }));

import { configuredProviders, getProvider, providerHealth } from "@/surf/search/providers";

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

function setKeys() {
  process.env.TAVILY_API_KEY = ALL_KEYS.TAVILY_API_KEY;
  process.env.EXA_API_KEY = ALL_KEYS.EXA_API_KEY;
  process.env.SEARXNG_BASE_URL = ALL_KEYS.SEARXNG_BASE_URL;
}

beforeEach(() => {
  clearKeys();
  vi.clearAllMocks();
});

describe("providerHealth", () => {
  it("reports live reachability for every configured provider", async () => {
    setKeys();
    const health = await providerHealth();
    expect(health.map((h) => h.name).sort()).toEqual(["exa", "jina", "keyless", "searxng", "tavily"]);
    const keyless = health.find((h) => h.name === "keyless");
    const jina = health.find((h) => h.name === "jina");
    expect(keyless?.ok).toBe(true);
    expect(jina?.ok).toBe(false);
    expect(jina?.error).toContain("429");
    // Every probe hit a real mock call — no provider is skipped.
    expect(mockRelaiSearch).toHaveBeenCalledOnce();
    expect(mockJinaSearch).toHaveBeenCalledOnce();
  });

  it("only probes providers that are actually configured", async () => {
    const health = await providerHealth();
    expect(health.map((h) => h.name).sort()).toEqual(["jina", "keyless"]);
    expect(mockTavilySearch).not.toHaveBeenCalled();
    expect(mockExaSearch).not.toHaveBeenCalled();
  });
});

describe("provider caching", () => {
  it("serves repeat identical searches from the cache", async () => {
    const keyless = getProvider("keyless");
    expect(keyless).toBeDefined();
    await keyless!.search("cache me");
    await keyless!.search("cache me");
    expect(mockRelaiSearch).toHaveBeenCalledTimes(1);
  });

  it("never caches a blocked run so a transient failure cannot linger", async () => {
    mockRelaiSearch.mockResolvedValueOnce({
      hits: [],
      engine: "none",
      query: "",
      tried: 0,
      errors: ["keyless blocked"],
    });
    const keyless = getProvider("keyless")!;
    const first = await keyless.search("flaky");
    expect(first.blocked).toBe(true);
    const second = await keyless.search("flaky");
    expect(second.blocked).toBe(false);
    expect(mockRelaiSearch).toHaveBeenCalledTimes(2);
  });

  it("caches API provider runs too", async () => {
    setKeys();
    const tavily = getProvider("tavily")!;
    await tavily.search("api cache");
    await tavily.search("api cache");
    expect(mockTavilySearch).toHaveBeenCalledTimes(1);
  });

  it("does not cache failed API provider runs", async () => {
    const jina = getProvider("jina")!;
    const first = await jina.search("failing");
    expect(first.blocked).toBe(true);
    expect(first.errors).toEqual(["jina: HTTP 429 (rate limited)"]);
    // Second call re-fetches because the failure was not cached.
    const second = await jina.search("failing");
    expect(second.blocked).toBe(true);
    expect(mockJinaSearch).toHaveBeenCalledTimes(2);
  });
});

describe("configuredProviders", () => {
  it("only returns providers whose keys are set", () => {
    setKeys();
    expect(configuredProviders().map((p) => p.name).sort()).toEqual(["exa", "jina", "keyless", "searxng", "tavily"]);
  });
});
