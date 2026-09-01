/**
 * relaiSearch / relaiHealthCheck — network-free unit tests.
 *
 * `relaiFetch` (http.server) and `rankHits` (retrieval.server) are mocked so
 * no HTTP or LLM calls ever leave the test. The jina provider is mocked too
 * because it is always "configured" and would otherwise lazy-import a live
 * network call.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("@/RelAI/web/http.server", () => ({
  relaiFetch: vi.fn(),
  isPrivateHost: vi.fn(() => false),
}));

vi.mock("@/RelAI/web/retrieval.server", () => ({
  rankHits: vi.fn(async () => []),
}));

vi.mock("@/RelAI/web/providers/jina.server", () => ({
  jinaSearch: vi.fn(async () => ({ hits: [], engine: "jina", error: "jina: HTTP 429 (rate limited)" })),
}));

import { relaiFetch } from "@/RelAI/web/http.server";
import type { FetchResult } from "@/RelAI/web/http.server";
import { relaiSearch, relaiHealthCheck } from "@/RelAI/web/search.server";

const mockedRelaiFetch = vi.mocked(relaiFetch);

function htmlResult(url: string, title: string): string {
  return (
    `<html><head><title>Search results</title></head><body><main class="results">` +
    `<a class="result__a" href="${url}">${title}</a>` +
    `</main></body></html>`
  );
}

function blockedResponse(engine: string): FetchResult {
  return {
    ok: false,
    status: 403,
    text: "",
    error: `${engine}: HTTP 403`,
    url: "",
    contentType: "text/html",
    blocked: true,
    attempts: 1,
    fetchedAt: new Date().toISOString(),
  };
}

function okResponse(text: string): FetchResult {
  return {
    ok: true,
    status: 200,
    text,
    error: undefined,
    url: "",
    contentType: "text/html",
    blocked: false,
    attempts: 1,
    fetchedAt: new Date().toISOString(),
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  mockedRelaiFetch.mockImplementation(async (url: string) => {
    const u = url.toLowerCase();
    if (u.includes("duckduckgo")) return okResponse(htmlResult("https://acme.io/", "Acme"));
    if (u.includes("search.brave.com")) return okResponse(htmlResult("https://globex.com/", "Globex"));
    return blockedResponse("html");
  });
});

describe("relaiSearch — aggregate fan-out", () => {
  it("tries every provider even after one returns hits, dedupes, and slices to limit", async () => {
    const res = await relaiSearch("saas startups", { aggregate: true, limit: 5 });
    expect(res.hits.map((h) => h.url)).toContain("https://acme.io/");
    expect(res.hits.map((h) => h.url)).toContain("https://globex.com/");
    // Fan-out ran the whole list (jina + 6 HTML engines), not first-hit-wins.
    expect(mockedRelaiFetch.mock.calls.length).toBeGreaterThanOrEqual(6);
    // Blocked engines surface as errors, not as empty success.
    expect(res.errors.some((e) => e.includes("403"))).toBe(true);
    expect(res.errors.length).toBeGreaterThan(0);
    expect(res.hits.length).toBeLessThanOrEqual(5);
  });

  it("dedupes identical URLs across providers", async () => {
    // Both engines return the same URL now.
    mockedRelaiFetch.mockImplementation(async () => okResponse(htmlResult("https://acme.io/", "Acme")));
    const res = await relaiSearch("saas", { aggregate: true, limit: 10 });
    expect(res.hits.filter((h) => h.url === "https://acme.io/").length).toBe(1);
  });

  it("surfaces API provider errors instead of treating them as no results", async () => {
    const res = await relaiSearch("saas", { aggregate: true });
    expect(res.errors.some((e) => e.includes("jina"))).toBe(true);
    expect(res.errors.some((e) => e.includes("429"))).toBe(true);
  });
});

describe("relaiSearch — pagination", () => {
  it("passes the SERP page through to each HTML engine's pagination param", async () => {
    await relaiSearch("saas", { aggregate: true, page: 2 });
    const urls = mockedRelaiFetch.mock.calls.map((c) => String(c[0]).toLowerCase());
    expect(urls.some((u) => u.includes("duckduckgo") && u.includes("s=10"))).toBe(true);
    expect(urls.some((u) => u.includes("brave") && u.includes("offset=10"))).toBe(true);
    expect(urls.some((u) => u.includes("bing") && u.includes("first=11"))).toBe(true);
    expect(urls.some((u) => u.includes("mojeek") && u.includes("page=2"))).toBe(true);
    expect(urls.some((u) => u.includes("google") && u.includes("start=10"))).toBe(true);
  });

  it("uses page 1 (no pagination params) by default", async () => {
    await relaiSearch("saas", { aggregate: true });
    const urls = mockedRelaiFetch.mock.calls.map((c) => String(c[0]).toLowerCase());
    expect(urls.some((u) => u.includes("duckduckgo") && u.includes("s=0"))).toBe(true);
  });
});

describe("relaiSearch — fast path unchanged", () => {
  it("returns first-hit-wins without aggregate and stops early", async () => {
    const res = await relaiSearch("saas");
    expect(res.hits.length).toBeGreaterThan(0);
    // Fast path: jina is first (configured), it returns an error, then the
    // first HTML engine (duckduckgo) that succeeds ends the loop.
    expect(res.engine).toBe("duckduckgo");
    expect(res.errors.some((e) => e.includes("jina"))).toBe(true);
  });
});

describe("relaiHealthCheck", () => {
  it("probes every provider and reports per-provider status", async () => {
    const report = await relaiHealthCheck();
    // 1 configured API (jina) + 6 HTML engines.
    expect(report.providers.length).toBe(7);
    expect(report.providers.some((p) => p.status === "ok")).toBe(true);
    const ddg = report.providers.find((p) => p.name === "duckduckgo");
    expect(ddg?.hits).toBe(1);
    expect(ddg?.status).toBe("ok");
    const blocked = report.providers.find((p) => p.name === "bing");
    expect(blocked?.status).toBe("error");
    expect(report.anyWorking).toBe(true);
    expect(report.totalHits).toBeGreaterThan(0);
  });
});
