/**
 * Relay Surf — controlled tool registry (server-only).
 *
 * The safe surface exposed to Relay Brain, RVOS, Flow and the API. Every tool
 * validates its inputs, reports honest activity text, and gives the caller
 * structured data — never raw logs, raw HTML, or browser/JS control.
 *
 * The registry is the single source of truth for which Surf capabilities
 * exist; the server functions and REST routes both go through it.
 */
import type { SurfTool, SurfToolResult } from "./types";
import { routeSearch, runDeepSearch } from "./search/router";
import { providerSummary } from "./search/providers";
import { searchPlatform } from "./platforms";
import { createBrowseSession, navigate, currentPage, getBrowseSession, browseEnabled } from "./browse/browser";
import { findOnPage } from "./browse/find-on-page";
import { assertSafeUrl, clampInt, requireText } from "./guard";

/* ------------------------------------------------------------------ */
/*  Result helpers                                                     */
/* ------------------------------------------------------------------ */

function ok(data: unknown, activity?: string): SurfToolResult {
  return { ok: true, data, activity };
}

function fail(message: string): SurfToolResult {
  return { ok: false, data: null, error: message };
}

function platformFromSite(site: string): string {
  const s = site.toLowerCase();
  if (s.includes("reddit")) return "reddit";
  if (s.includes("github")) return "github";
  if (s.includes("youtube") || s.includes("youtu.be")) return "youtube";
  if (s.includes("x.com") || s.includes("twitter")) return "x";
  if (s.includes("instagram")) return "instagram";
  if (s.includes("linkedin")) return "linkedin";
  return "generic";
}

/* ------------------------------------------------------------------ */
/*  Tool definitions                                                   */
/* ------------------------------------------------------------------ */

const webSearchTool: SurfTool = {
  name: "web_search",
  description: "Search the open web. Results carry which provider actually served them.",
  category: "search",
  inputSchema: {
    type: "object",
    properties: {
      query: { type: "string", description: "Search query" },
      intent: { type: "string", enum: ["general", "technical", "news", "deep", "platform"] },
      limit: { type: "integer", description: "Max results (1-30)" },
      site: { type: "string", description: "Restrict results to a single site" },
    },
    required: ["query"],
  },
  async execute(input) {
    const { query, intent, limit, site } = input as { query: string; intent?: string; limit?: number; site?: string };
    const q = requireText(query);
    const lim = clampInt(limit, 10, 1, 30);
    const opts = { limit: lim, site: site && typeof site === "string" ? site : undefined };
    if (intent === "deep") {
      const run = await runDeepSearch(q, opts);
      return ok({ results: run.results, provider: run.provider, note: run.note }, `Found ${run.results.length} results (${run.provider})`);
    }
    const run = await routeSearch({ query: q, intent: (intent as never) ?? "general", options: opts });
    return ok({ results: run.results, provider: run.provider, note: run.note }, `Found ${run.results.length} results (${run.provider})`);
  },
};

const openUrlTool: SurfTool = {
  name: "open_url",
  description: "Open a URL in a browse session and return its clean text, links and headings.",
  category: "browse",
  inputSchema: {
    type: "object",
    properties: { url: { type: "string" }, sessionId: { type: "string" } },
    required: ["url"],
  },
  async execute(input) {
    const { url, sessionId } = input as { url: string; sessionId?: string };
    const safeUrl = assertSafeUrl(url);
    let sid = typeof sessionId === "string" && getBrowseSession(sessionId) ? sessionId : "";
    if (!sid) sid = createBrowseSession().id;
    const result = await navigate(sid, { kind: "goto", target: safeUrl });
    if (!result.ok) return fail(result.error ?? "Could not open URL.");
    return ok(
      {
        sessionId: sid,
        url: result.url,
        title: result.title,
        text: result.text,
        links: result.links,
        headings: result.headings,
        mode: browseEnabled().mode,
      },
      `Opened ${new URL(result.url).hostname}`,
    );
  },
};

const navigateTool: SurfTool = {
  name: "navigate",
  description: "Navigate the current browse session: back, forward, reload, or click a link (by URL, anchor text, or intent like pricing/docs).",
  category: "browse",
  inputSchema: {
    type: "object",
    properties: {
      sessionId: { type: "string" },
      action: { type: "string", enum: ["back", "forward", "reload", "click"] },
      target: { type: "string" },
    },
    required: ["sessionId", "action"],
  },
  async execute(input) {
    const { sessionId, action, target } = input as { sessionId: string; action: string; target?: string };
    if (!getBrowseSession(sessionId)) return fail("Browse session not found.");
    const kind = action === "back" || action === "forward" || action === "reload" ? action : "click";
    if (kind === "click" && !target) return fail("click requires a target (URL, anchor text, or intent).");
    const result = await navigate(sessionId, { kind, target: target as never });
    if (!result.ok) return fail(result.error ?? "Navigation failed.");
    return ok(
      { sessionId, url: result.url, title: result.title, text: result.text?.slice(0, 6000), links: result.links },
      `Navigated to ${new URL(result.url).hostname}`,
    );
  },
};

const extractPageTool: SurfTool = {
  name: "extract_page",
  description: "Extract the current page (or a URL) as clean text, links and headings — never raw HTML.",
  category: "browse",
  inputSchema: {
    type: "object",
    properties: {
      sessionId: { type: "string" },
      url: { type: "string" },
      maxChars: { type: "integer" },
    },
  },
  async execute(input) {
    const { sessionId, url, maxChars } = input as { sessionId?: string; url?: string; maxChars?: number };
    let page = typeof sessionId === "string" ? currentPage(sessionId) : null;
    if (!page && url) {
      const sid = createBrowseSession().id;
      const result = await navigate(sid, { kind: "goto", target: assertSafeUrl(url) });
      if (!result.ok) return fail(result.error ?? "Could not load page.");
      page = currentPage(sid);
    }
    if (!page) return fail("No page loaded — open a URL or provide one.");
    const max = clampInt(maxChars, 10_000, 500, 40_000);
    return ok(
      {
        url: page.url,
        title: page.title,
        text: page.text.slice(0, max),
        links: page.links.slice(0, 80),
        headings: page.headings.slice(0, 60),
        metadata: page.metadata,
        truncated: page.truncated || page.text.length > max,
      },
      `Extracted ${Math.min(page.text.length, max).toLocaleString()} chars`,
    );
  },
};

const findOnPageTool: SurfTool = {
  name: "find_on_page",
  description: "Locate a keyword or section in the loaded page and return the surrounding content with a jump target.",
  category: "browse",
  inputSchema: {
    type: "object",
    properties: { sessionId: { type: "string" }, query: { type: "string" } },
    required: ["sessionId", "query"],
  },
  async execute(input) {
    const { sessionId, query } = input as { sessionId: string; query: string };
    const page = currentPage(sessionId);
    if (!page) return fail("No page loaded in this session.");
    const result = findOnPage(page, requireText(query));
    return ok(
      { url: result.url, count: result.count, best: result.best, matches: result.matches.slice(0, 5) },
      result.count > 0 ? `Found ${result.count} matches for "${query}"` : `No matches for "${query}"`,
    );
  },
};

const getLinksTool: SurfTool = {
  name: "get_links",
  description: "List links on the current page, optionally filtered by intent (pricing, docs, faq…).",
  category: "browse",
  inputSchema: {
    type: "object",
    properties: { sessionId: { type: "string" }, intent: { type: "string" } },
    required: ["sessionId"],
  },
  async execute(input) {
    const { sessionId, intent } = input as { sessionId: string; intent?: string };
    const page = currentPage(sessionId);
    if (!page) return fail("No page loaded in this session.");
    const links = typeof intent === "string" && intent
      ? page.links.filter((l) => l.intent === intent || l.intent?.includes(intent))
      : page.links;
    return ok({ links: links.slice(0, 80), total: page.links.length }, `${links.length} links`);
  },
};

const searchSiteTool: SurfTool = {
  name: "search_site",
  description: "Search within a website or platform (Reddit, GitHub, YouTube, …). Honest about how results were reached.",
  category: "search",
  inputSchema: {
    type: "object",
    properties: {
      query: { type: "string" },
      site: { type: "string" },
      platform: { type: "string", enum: ["reddit", "x", "instagram", "youtube", "github", "linkedin", "generic"] },
      limit: { type: "integer" },
    },
    required: ["query"],
  },
  async execute(input) {
    const { query, site, platform, limit } = input as { query: string; site?: string; platform?: string; limit?: number };
    const q = requireText(query);
    const p = typeof platform === "string" && platform ? platform : typeof site === "string" && site ? platformFromSite(site) : "generic";
    const res = await searchPlatform({ platform: p as never, query: q, options: { limit: clampInt(limit, 8, 1, 30) } });
    return ok(
      { results: res.results, platform: res.platform, note: res.note, native: res.native },
      `Found ${res.results.length} results on ${res.platform}`,
    );
  },
};

const searchPlatformTool: SurfTool = {
  name: "search_platform",
  description: "Search a specific platform (Reddit, GitHub, YouTube, X, Instagram, LinkedIn).",
  category: "platform",
  inputSchema: {
    type: "object",
    properties: {
      platform: { type: "string", enum: ["reddit", "x", "instagram", "youtube", "github", "linkedin"] },
      query: { type: "string" },
      limit: { type: "integer" },
    },
    required: ["platform", "query"],
  },
  async execute(input) {
    const { platform, query, limit } = input as { platform: string; query: string; limit?: number };
    const res = await searchPlatform({
      platform: platform as never,
      query: requireText(query),
      options: { limit: clampInt(limit, 8, 1, 30) },
    });
    return ok(
      { results: res.results, platform: res.platform, note: res.note, native: res.native },
      `Found ${res.results.length} results on ${res.platform}`,
    );
  },
};

const providerStatusTool: SurfTool = {
  name: "provider_status",
  description: "List which search providers and browsing mode are actually configured on this deployment.",
  category: "search",
  inputSchema: { type: "object", properties: {} },
  async execute() {
    const providers = providerSummary();
    return ok(
      { providers, browserMode: browseEnabled().mode },
      `${providers.filter((p) => p.available).length} search providers active`,
    );
  },
};

const researchTool: SurfTool = {
  name: "research",
  description:
    "Run a complete multi-step web investigation (search, browse, extract, verify, synthesize) and get a cited report. Returns a researchId; poll research_get for progress.",
  category: "research",
  inputSchema: {
    type: "object",
    properties: {
      objective: { type: "string" },
      depth: { type: "string", enum: ["quick", "standard", "deep", "exhaustive"] },
    },
    required: ["objective"],
  },
  async execute() {
    // Executed with a user context in runSurfTool — never without one.
    return fail("research requires an authenticated user context; use research_run via the API/RPC.");
  },
};

export const SURF_TOOLS: SurfTool[] = [
  webSearchTool,
  providerStatusTool,
  openUrlTool,
  navigateTool,
  extractPageTool,
  findOnPageTool,
  getLinksTool,
  searchSiteTool,
  searchPlatformTool,
  researchTool,
];

export interface SurfToolContext {
  userId: string;
  workspaceId?: string;
}

/**
 * Execute a Surf tool by name. The `research` tool needs the caller's identity
 * (sessions are user-scoped) so it is wired here rather than in the registry.
 */
export async function runSurfTool(
  name: string,
  input: unknown,
  ctx: SurfToolContext,
): Promise<SurfToolResult> {
  const tool = SURF_TOOLS.find((t) => t.name === name);
  if (!tool) return fail(`Unknown Surf tool: ${name}`);

  const available = typeof tool.available === "function" ? await tool.available() : tool.available !== false;
  if (!available) return fail(`${name} is not available on this deployment.`);

  if (name === "research") {
    const { startResearch } = await import("./research/engine");
    const { objective, depth } = input as { objective?: string; depth?: string };
    const session = await startResearch({
      objective: requireText(objective, "objective"),
      userId: ctx.userId,
      workspaceId: ctx.workspaceId,
      depth: depth === "quick" || depth === "deep" || depth === "exhaustive" ? depth : "standard",
    });
    return ok(
      { researchId: session.id, status: session.status, plan: session.plan },
      "Research started — poll research_get for progress",
    );
  }

  return tool.execute(input);
}
