/**
 * Surf research engine tests — evidence quality.
 *
 * The router, planner, platforms and guard are mocked so the loop runs
 * entirely in-memory: search results come from `routeSearch`, page text from a
 * fake adapter. The in-memory session store is real, so evidence output can be
 * asserted end-to-end.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import type { PageData, SurfPlan } from "../types";

const h = vi.hoisted(() => {
  const routeSearch = vi.fn(
    async (input: { query: string }): Promise<{ provider: string; query: string; results: Array<Record<string, string>>; blocked: boolean }> => ({
      provider: "mock",
      query: input.query,
      results: [],
      blocked: false,
    }),
  );
  const runDeepSearch = vi.fn(async () => ({
    provider: "mock",
    query: "",
    results: [] as Array<Record<string, string>>,
    blocked: false,
    providersUsed: [],
    deduped: 0,
  }));
  const searchPlatform = vi.fn(async () => ({ results: [], platform: "generic", native: false }));
  const adapterForUrl = vi.fn(
    (_url: string): { extractPage: () => Promise<PageData | null> } => ({
      extractPage: async () => null,
    }),
  );
  const planResearch = vi.fn(
    async (objective: string, depth: SurfPlan["depth"] = "quick"): Promise<SurfPlan> => ({
      objective,
      queries: ["acme funding round"],
      sources: ["https://acme.io/"],
      requiredInformation: [],
      verificationRequirements: [],
      depth,
    }),
  );
  return { routeSearch, runDeepSearch, searchPlatform, adapterForUrl, planResearch };
});

vi.mock("./planner", () => ({ planResearch: h.planResearch }));
vi.mock("../search/router", () => ({ routeSearch: h.routeSearch, runDeepSearch: h.runDeepSearch }));
vi.mock("../platforms", () => ({ searchPlatform: h.searchPlatform, adapterForUrl: h.adapterForUrl }));
vi.mock("../guard", () => ({ assertSafeUrl: (url: string) => url }));

import { startResearch, getResearch } from "./engine";

beforeEach(() => {
  vi.clearAllMocks();
  h.routeSearch.mockImplementation(async (input: { query: string }) => ({
    provider: "mock",
    query: input.query,
    results: [],
    blocked: false,
  }));
  h.adapterForUrl.mockReturnValue({
    extractPage: async () => null,
  });
});

function hit(url: string): Record<string, string> {
  return { url, title: url, snippet: "", provider: "mock", accessMode: "public" };
}

function pageData(url: string, text: string): PageData {
  return {
    url,
    title: "Page",
    text,
    links: [],
    headings: [],
    metadata: {},
    truncated: false,
    fetchedAt: new Date().toISOString(),
    status: 200,
    contentType: "text/html",
    accessMode: "public",
  };
}

async function runUntilSettled(id: string, timeoutMs = 3000): Promise<ReturnType<typeof getResearch>> {
  const start = Date.now();
  for (;;) {
    const session = getResearch(id);
    if (session && (session.status === "complete" || session.status === "failed")) return session;
    if (Date.now() - start > timeoutMs) throw new Error("research loop did not settle in time");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

describe("evidence extraction", () => {
  it("prefers objective-relevant sentences over boilerplate", async () => {
    const FILLER =
      "Acme Corporation has offices in several cities around the world. " +
      "The founding team began with a simple idea over a decade ago. " +
      "Operations have expanded steadily into new markets each year. " +
      "The company employs a growing number of people today. " +
      "Acme recently closed a $40 million funding round led by its investors. " +
      "Management has set ambitious targets for the coming years.";

    h.routeSearch.mockImplementation(async () => ({
      provider: "mock",
      query: "q",
      results: [hit("https://acme.io/")],
      blocked: false,
    }));
    h.adapterForUrl.mockReturnValue({
      extractPage: async () => pageData("https://acme.io/", FILLER),
    });

    const session = await startResearch({ objective: "how much funding did acme raise", userId: "u1", depth: "quick" });
    const done = await runUntilSettled(session.id);

    expect(done?.status).toBe("complete");
    expect(done?.evidence.length).toBeGreaterThan(0);
    expect(done?.evidence[0].excerpt).toContain("funding");
    expect(done?.evidence[0].excerpt).toContain("40 million");
  });

  it("suppresses near-duplicate sentences across pages", async () => {
    const BOILER =
      "This website uses cookies to improve your browsing experience. " +
      "Please accept cookies to continue using our service. " +
      "We also collect basic analytics to understand how visitors use our pages. " +
      "Our privacy policy explains everything in plain language.";

    h.routeSearch.mockImplementation(async () => ({
      provider: "mock",
      query: "q",
      results: [hit("https://one.io/"), hit("https://two.io/"), hit("https://three.io/")],
      blocked: false,
    }));
    h.adapterForUrl.mockReturnValue({
      extractPage: async () => pageData("https://one.io/", BOILER),
    });

    const session = await startResearch({ objective: "research funding for saas companies", userId: "u1", depth: "quick" });
    const done = await runUntilSettled(session.id);

    expect(done?.status).toBe("complete");
    // One page's worth of sentences — the identical boilerplate on the other
    // two pages must not duplicate it.
    expect(done?.evidence.length).toBe(4);
    // Every source is still tracked, even when it added no new evidence.
    expect(done?.sources.length).toBe(3);
  });

  it("keeps evidence when a page fails to load (honest per-page failure)", async () => {
    h.routeSearch.mockImplementation(async () => ({
      provider: "mock",
      query: "q",
      results: [hit("https://loads.io/"), hit("https://fails.io/")],
      blocked: false,
    }));
    h.adapterForUrl.mockImplementation((url: string) => ({
      extractPage: async () =>
        url === "https://loads.io/"
          ? pageData(
              url,
              "A detailed report was published about the funding environment this year. " +
                "Investors remain cautious but active in early stage deals across Europe. " +
                "The report covers valuation trends and expected market growth for 2026. " +
                "Startups in the sector should prepare for a slower fundraising cycle.",
            )
          : null,
    }));

    const session = await startResearch({ objective: "funding environment 2026", userId: "u1", depth: "quick" });
    const done = await runUntilSettled(session.id);

    expect(done?.status).toBe("complete");
    expect(done?.evidence.length).toBe(4);
    expect(done?.sources.map((s) => s.url)).toContain("https://fails.io/");
  });
});
