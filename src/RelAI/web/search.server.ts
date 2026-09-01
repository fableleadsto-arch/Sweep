/**
 * RelAI — open web layer (server-only).
 *
 * DuckDuckGo first, then a fallback chain so one blocked provider can never
 * take RelAI offline. Also exposes a page reader and an OSINT sweep that
 * fans out across public sources. No API keys, no paid providers.
 *
 * Each engine parser tries multiple patterns in order of specificity so a
 * site redesign never silently breaks searches.
 */

import { relaiFetch, isPrivateHost } from "./http.server";
import { htmlToMarkdown, jsonToMarkdown, htmlToText } from "./markdown.server";
import { rankHits } from "./retrieval.server";
import { normalizeBatch } from "./normalize.server";

export interface RelAIWebHit {
  url: string;
  title: string;
  snippet: string;
  engine: string;
  relevance?: number;
}

export interface RelAIPage {
  url: string;
  title: string;
  text: string;
  status: number;
  truncated: boolean;
}

type Engine = {
  name: string;
  build: (q: string, page: number) => string;
  parse: (html: string) => RelAIWebHit[];
};

/** Kept for backwards compatibility — delegates to the Markdown layer. */
export function stripHtml(input: string): string {
  return htmlToText(input);
}

/** Try all patterns and return the first non-empty match group. */
function firstMatch(text: string, patterns: RegExp[]): string {
  for (const re of patterns) {
    const m = re.exec(text);
    if (m?.[1]) return stripHtml(m[1]);
  }
  return "";
}

/**
 * Flexible DDG result parser.
 * The html.duckduckgo.com/html/ endpoint has changed structure over the years.
 * We try block-level extraction first, then link-based fallback.
 */
function parseDdg(html: string): RelAIWebHit[] {
  const results: RelAIWebHit[] = [];
  const seen = new Set<string>();

  // Strategy 1: Look for result blocks with class containing "result" and "results_links"
  const blocks = extractBlocks(html, [
    /<div[^>]*class="[^"]*result[^"]*results_links[^"]*"[^>]*>([\s\S]*?)<\/div>\s*<\/div>/gi,
    /<div[^>]*class="[^"]*result[^"]*"[^>]*data-nrn="[^"]*"[^>]*>([\s\S]*?)<\/div>\s*<!--\s*end\s*result\s*-->/gi,
    /<div[^>]*class="[^"]*result\b[^"]*"[^>]*>([\s\S]*?)(?:<\/div>\s*){2,3}(?=<div[^>]*class="(?:result|nav)[^"]*">)/i,
  ]);

  for (const block of blocks) {
    const url = extractUrl(block, [
      /<a[^>]+href="(https?:\/\/[^"]+)"[^>]*>/i,
      /<a[^>]+href="(\/\/[^"]+)"[^>]*>/i,
      /<a[^>]+href="([^"]+)"[^>]*class="[^"]*result__a[^"]*"[^>]*>/i,
    ]);
    const title = firstMatch(block, [
      /class="[^"]*result__a[^"]*"[^>]*>([\s\S]*?)<\/a>/i,
      /<a[^>]+href="https?:\/\/[^"]+"[^>]*>([\s\S]*?)<\/a>/i,
    ]);
    const snippet = firstMatch(block, [
      /class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\//i,
      /class="[^"]*snippet[^"]*"[^>]*>([\s\S]*?)<\//i,
      /<td[^>]*class="[^"]*snippet[^"]*"[^>]*>([\s\S]*?)<\/td>/i,
    ]);

    if (url && title && !seen.has(url)) {
      seen.add(url);
      results.push({ url, title, snippet, engine: "duckduckgo" });
    }
  }

  if (results.length > 0) return results;

  // Strategy 2: Find anchor tags with result-like classes
  const linkRe = /<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
  let m: RegExpExecArray | null;
  while ((m = linkRe.exec(html)) !== null) {
    const url = decodeRedirect(m[1]);
    const title = stripHtml(m[2]);
    if (!url.startsWith("http") || !title || seen.has(url)) continue;
    seen.add(url);
    const snippet = ddgSnippet(html, m.index);
    results.push({ url, title, snippet, engine: "duckduckgo" });
    if (results.length >= 40) break;
  }

  if (results.length > 0) return results;

  // Strategy 3: Any link inside a result-like div
  const anyResultRe = /<div[^>]*class="[^"]*\bresult\b[^"]*"[^>]*>([\s\S]*?)<\/div>/gi;
  while ((m = anyResultRe.exec(html)) !== null) {
    const block = m[1];
    const linkMatch = block.match(/<a[^>]+href="(https?:\/\/[^"]+)"[^>]*>([\s\S]*?)<\/a>/i);
    if (!linkMatch) continue;
    const url = decodeRedirect(linkMatch[1]);
    const title = stripHtml(linkMatch[2]);
    if (!url.startsWith("http") || !title || seen.has(url)) continue;
    seen.add(url);
    results.push({ url, title, snippet: "", engine: "duckduckgo" });
    if (results.length >= 40) break;
  }

  return results;
}

function extractBlocks(html: string, patterns: RegExp[]): string[] {
  for (const re of patterns) {
    const blocks = html.match(re);
    if (blocks && blocks.length > 1) return blocks;
  }
  return [];
}

function extractUrl(text: string, patterns: RegExp[]): string {
  for (const re of patterns) {
    const m = re.exec(text);
    if (m) {
      const url = m[1].startsWith("//") ? `https:${m[1]}` : m[1];
      return decodeRedirect(url);
    }
  }
  return "";
}

function ddgSnippet(html: string, _anchorIndex: number): string {
  return firstMatch(html.slice(_anchorIndex), [
    /class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\//i,
    /class="[^"]*snippet[^"]*"[^>]*>([\s\S]*?)<\//i,
  ]);
}

function parseDdgLite(html: string): RelAIWebHit[] {
  const results: RelAIWebHit[] = [];
  const rows = html.match(/<tr[^>]*>[\s\S]*?<\/tr>/gi) ?? [];

  for (const row of rows) {
    if (!row.includes("result-link") && !row.includes("result-snippet")) continue;
    const url = firstMatch(row, [
      /<a[^>]+href="([^"]+)"[^>]*class="[^"]*result-link[^"]*"[^>]*>/i,
      /<a[^>]+href="(https?:\/\/[^"]+)"[^>]*>/i,
    ]);
    const title = firstMatch(row, [
      /class="[^"]*result-link[^"]*"[^>]*>([\s\S]*?)<\/a>/i,
    ]);
    const snippet = firstMatch(row, [
      /class="[^"]*result-snippet[^"]*"[^>]*>([\s\S]*?)<\/t[dD]/i,
      /<td[^>]*>([\s\S]*?)<\/td>/i,
    ]);
    if (url && title) {
      results.push({ url, title, snippet, engine: "duckduckgo-lite" });
    }
  }

  return results;
}

function parseBrave(html: string): RelAIWebHit[] {
  const results: RelAIWebHit[] = [];
  const blocks = html.match(
    /<div[^>]*class="[^"]*(?:snippet|result|search-result)[^"]*"[^>]*>[\s\S]*?<a[^>]+href="https?:\/\/[^"]+"[^>]*>[\s\S]*?<\/a>[\s\S]*?<\/div>/gi,
  ) ?? [];

  for (const block of blocks) {
    const url = firstMatch(block, [
      /<a[^>]+href="(https?:\/\/[^"]+)"[^>]*>\s*<div[^>]*class="[^"]*title[^"]*"[^>]*>/i,
      /<a[^>]+href="(https?:\/\/[^"]+)"[^>]*>/i,
    ]);
    const title = firstMatch(block, [
      /class="[^"]*title[^"]*"[^>]*>([\s\S]*?)<\/div>/i,
    ]);
    const snippet = firstMatch(block, [
      /class="[^"]*snippet[^"]*"[^>]*>([\s\S]*?)<\/div>/i,
      /class="[^"]*description[^"]*"[^>]*>([\s\S]*?)<\//i,
    ]);
    if (url && title) {
      results.push({ url, title, snippet, engine: "brave" });
    }
  }

  if (results.length > 0) return results;

  // Fallback: any link with title-like sibling
  const simpleRe = /<a[^>]+href="(https?:\/\/[^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
  let m: RegExpExecArray | null;
  while ((m = simpleRe.exec(html)) !== null) {
    const url = m[1];
    const title = stripHtml(m[2]);
    if (url && title && !url.includes("brave.com")) {
      results.push({ url, title, snippet: "", engine: "brave" });
      if (results.length >= 30) break;
    }
  }

  return results;
}

function decodeBingRedirect(href: string): string {
  if (href.includes("/url?q=")) {
    try { return decodeURIComponent(href.split("/url?q=")[1]?.split("&")[0] ?? href); } catch { /* noop */ }
  }
  return href;
}

function bingCiteToUrl(cite: string): string {
  const clean = cite.replace(/<[^>]*>/g, "").trim();
  if (!clean.startsWith("http")) return "";
  const url = clean.replace(/\s*›\s*/g, "/").replace(/&amp;/g, "&").replace(/\s+/g, "");
  try { return new URL(url).href; } catch { return url; }
}

function parseBing(html: string): RelAIWebHit[] {
  const results: RelAIWebHit[] = [];

  const blocks = html.match(
    /<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>[\s\S]*?<\/li>/gi,
  ) ?? html.match(
    /<div[^>]*class="[^"]*b_algo[^"]*"[^>]*>[\s\S]*?<\/div>/gi,
  ) ?? [];

  for (const block of blocks) {
    const citeText = firstMatch(block, [
      /<cite[^>]*>([\s\S]*?)<\/cite>/i,
      /<div[^>]*class="[^"]*b_attribution[^"]*"[^>]*>[\s\S]*?<cite[^>]*>([\s\S]*?)<\/cite>/i,
    ]);
    const citeUrl = citeText ? bingCiteToUrl(citeText) : "";
    const rawUrl = firstMatch(block, [
      /<a[^>]+href="(https?:\/\/[^"]+)"[^>]*>/i,
    ]);
    const hrefUrl = rawUrl ? decodeBingRedirect(rawUrl) : "";
    const url = (citeUrl && citeUrl.startsWith("http")) ? citeUrl : hrefUrl;

    const title = firstMatch(block, [
      /<h2[^>]*>([\s\S]*?)<\/h2>/i,
    ]);
    const snippet = firstMatch(block, [
      /<p[^>]*>([\s\S]*?)<\/p>/i,
      /class="[^"]*b_caption[^"]*"[^>]*>[\s\S]*?<p[^>]*>([\s\S]*?)<\/p>/i,
    ]);
    if (url && title && url.startsWith("http")) {
      results.push({ url: stripHtml(url), title, snippet, engine: "bing" });
    }
  }

  if (results.length > 0) return results;

  const h2Re = /<h2[^>]*><a[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a><\/h2>/gi;
  let m: RegExpExecArray | null;
  while ((m = h2Re.exec(html)) !== null) {
    const rawUrl = m[1];
    const titleHtml = m[2];
    const citeInTitle = titleHtml.match(/<cite[^>]*>([\s\S]*?)<\/cite>/i);
    const url = citeInTitle ? bingCiteToUrl(citeInTitle[1]) : decodeBingRedirect(rawUrl);
    const title = stripHtml(titleHtml);
    if (url && title && url.startsWith("http")) {
      const snippet = bingSnippet(html, m.index);
      results.push({ url: stripHtml(url), title, snippet, engine: "bing" });
      if (results.length >= 30) break;
    }
  }

  return results;
}

function bingSnippet(html: string, anchorStart: number): string {
  return firstMatch(html.slice(anchorStart), [
    /<p[^>]*class="[^"]*b_lineclamp2[^"]*"[^>]*>([\s\S]*?)<\/p>/i,
    /<div[^>]*class="[^"]*b_caption[^"]*"[^>]*>[\s\S]*?<p[^>]*>([\s\S]*?)<\/p>/i,
  ]);
}

function parseMojeek(html: string): RelAIWebHit[] {
  const results: RelAIWebHit[] = [];

  const blocks = html.match(
    /<div[^>]*class="[^"]*result[^"]*"[^>]*>[\s\S]*?<\/div>\s*<\/div>/gi,
  ) ?? [];

  for (const block of blocks) {
    const url = firstMatch(block, [
      /<a[^>]+href="(https?:\/\/[^"]+)"[^>]*class="[^"]*ob[^"]*"[^>]*>/i,
      /<a[^>]+href="(https?:\/\/[^"]+)"[^>]*>/i,
    ]);
    const title = firstMatch(block, [
      /class="[^"]*ob[^"]*"[^>]*>([\s\S]*?)<\/a>/i,
      /<a[^>]+href="https?:\/\/[^"]+"[^>]*>([\s\S]*?)<\/a>/i,
    ]);
    const snippet = firstMatch(block, [
      /<p[^>]*class="[^"]*s[^"]*"[^>]*>([\s\S]*?)<\/p>/i,
      /class="[^"]*result-snippet[^"]*"[^>]*>([\s\S]*?)<\//i,
      /<p[^>]*>([\s\S]*?)<\/p>/i,
    ]);
    if (url && title) {
      results.push({ url, title, snippet, engine: "mojeek" });
    }
  }

  if (results.length > 0) return results;

  const classObRe = /<a[^>]+class="ob"[^>]+href="(https?:\/\/[^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
  let m: RegExpExecArray | null;
  while ((m = classObRe.exec(html)) !== null) {
    const url = m[1];
    const title = stripHtml(m[2]);
    if (url && title) {
      const snippet = mojeekSnippet(html, m.index);
      results.push({ url, title, snippet, engine: "mojeek" });
      if (results.length >= 30) break;
    }
  }

  return results;
}

function mojeekSnippet(html: string, anchorStart: number): string {
  return firstMatch(html.slice(anchorStart), [
    /<p[^>]*class="[^"]*s[^"]*"[^>]*>([\s\S]*?)<\/p>/i,
    /class="[^"]*result-snippet[^"]*"[^>]*>([\s\S]*?)<\//i,
  ]);
}

function parseGoogle(html: string): RelAIWebHit[] {
  const results: RelAIWebHit[] = [];

  // Google result blocks are div.g or similar
  const blocks = html.match(
    /<div[^>]*class="[^"]*g[^"]*"[^>]*>[\s\S]*?<\/div>\s*<\/div>/gi,
  ) ?? html.match(
    /<div[^>]*class="[^"]*[Gg]?[Ss]earch[^"]*Result[^"]*"[^>]*>[\s\S]*?<\/div>/gi,
  ) ?? [];

  for (const block of blocks) {
    const url = firstMatch(block, [
      /<a[^>]+href="(\/url\?q=[^"&]+)[^"]*"[^>]*>/i,
      /<a[^>]+href="(https?:\/\/[^"]+)"[^>]*>/i,
    ]);
    const decodedUrl = url?.startsWith("/url?q=")
      ? decodeURIComponent(url.slice(7).split("&")[0])
      : url ?? "";
    const title = firstMatch(block, [
      /<h3[^>]*>([\s\S]*?)<\/h3>/i,
      /class="[^"]*BNeawe[^"]*"[^>]*>([\s\S]*?)<\//i,
    ]);
    const snippet = firstMatch(block, [
      /class="[^"]*(?:VwiC3b|BNeawe|st[^"]*)[^"]*"[^>]*>([\s\S]*?)<\/div>/i,
      /<span[^>]*class="[^"]*aCOpRe[^"]*"[^>]*>([\s\S]*?)<\/span>/i,
    ]);
    if (decodedUrl && title && !decodedUrl.includes("google.com")) {
      results.push({ url: decodedUrl, title, snippet, engine: "google" });
    }
  }

  if (results.length > 0) return results;

  // Fallback: Simple URL extraction
  const urlRe = /<a[^>]+href="(\/url\?q=[^"&]+)[^"]*"[^>]*>([\s\S]*?)<\/a>/gi;
  let m: RegExpExecArray | null;
  while ((m = urlRe.exec(html)) !== null) {
    const raw = m[1];
    const title = stripHtml(m[2]);
    const url = decodeURIComponent(raw.slice(7).split("&")[0]);
    if (url && title && !url.includes("google.com")) {
      results.push({ url, title, snippet: googleSnippet(html, m.index), engine: "google" });
      if (results.length >= 30) break;
    }
  }

  return results;
}

function googleSnippet(html: string, anchorStart: number): string {
  return firstMatch(html.slice(anchorStart), [
    /<div[^>]*class="[^"]*(?:VwiC3b|BNeawe)[^"]*"[^>]*>([\s\S]*?)<\/div>/i,
    /<span[^>]*class="[^"]*aCOpRe[^"]*"[^>]*>([\s\S]*?)<\/span>/i,
  ]);
}

const ENGINES: Engine[] = [
  {
    name: "bing",
    build: (q, page) =>
      `https://www.bing.com/search?q=${encodeURIComponent(q)}&setlang=en&cc=US&first=${(page - 1) * 10 + 1}`,
    parse: parseBing,
  },
  {
    name: "duckduckgo",
    build: (q, page) =>
      `https://html.duckduckgo.com/html/?q=${encodeURIComponent(q)}&kl=wt-wt&s=${(page - 1) * 10}`,
    parse: parseDdg,
  },
  {
    name: "duckduckgo-lite",
    build: (q) => `https://lite.duckduckgo.com/lite/?q=${encodeURIComponent(q)}`,
    parse: parseDdgLite,
  },
  {
    name: "brave",
    build: (q, page) =>
      `https://search.brave.com/search?q=${encodeURIComponent(q)}&source=web&offset=${(page - 1) * 10}`,
    parse: parseBrave,
  },
  {
    name: "mojeek",
    build: (q, page) => `https://www.mojeek.com/search?q=${encodeURIComponent(q)}&page=${page}`,
    parse: parseMojeek,
  },
  {
    name: "google",
    build: (q, page) =>
      `https://www.google.com/search?q=${encodeURIComponent(q)}&hl=en&start=${(page - 1) * 10}`,
    parse: parseGoogle,
  },
];

/**
 * API-based search providers. These are tried after the free HTML engines.
 * Each provider has a `search()` function that returns hits or empty array.
 *
 * Circuit breaker: after the first rate-limit/credit error, skip the provider
 * for 60 seconds. This prevents 58+ lead-discovery calls from each wasting
 * time on a dead provider, while auto-resetting when the user adds/refreshes
 * an API key (server restarts also clear the breaker).
 */
const circuitBrokenUntil = new Map<string, number>();

const API_SEARCH_PROVIDERS = [
  {
    name: "tavily",
    configured: () => {
      if (!process.env.TAVILY_API_KEY) return false;
      const until = circuitBrokenUntil.get("tavily") ?? 0;
      if (until > Date.now()) return false;
      circuitBrokenUntil.delete("tavily");
      return true;
    },
    search: async (q: string, limit: number) => {
      try {
        const { tavilySearch } = await import("./providers/tavily.server");
        const res = await tavilySearch(q, { limit, includeAnswer: false });
        if (res.error && /limit|quota|credit|rate|exceeds/i.test(res.error)) circuitBrokenUntil.set("tavily", Date.now() + 60_000);
        return res;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (/limit|quota|credit|rate|exceeds/i.test(msg)) circuitBrokenUntil.set("tavily", Date.now() + 60_000);
        return { hits: [], error: msg };
      }
    },
  },
  {
    name: "exa",
    configured: () => {
      if (!process.env.EXA_API_KEY) return false;
      const until = circuitBrokenUntil.get("exa") ?? 0;
      if (until > Date.now()) return false;
      circuitBrokenUntil.delete("exa");
      return true;
    },
    search: async (q: string, limit: number) => {
      try {
        const { exaSearch } = await import("./providers/exa.server");
        const res = await exaSearch(q, { limit });
        if (res.error && /limit|quota|credit|rate|exceeds|429/i.test(res.error)) circuitBrokenUntil.set("exa", Date.now() + 60_000);
        return res;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (/limit|quota|credit|rate|exceeds|429/i.test(msg)) circuitBrokenUntil.set("exa", Date.now() + 60_000);
        return { hits: [], error: msg };
      }
    },
  },
  {
    name: "searxng",
    configured: () => Boolean(process.env.SEARXNG_BASE_URL),
    search: async (q: string, limit: number) => {
      const { searxngSearch } = await import("./providers/searxng.server");
      return searxngSearch(q, { limit });
    },
  },
  {
    name: "jina",
    configured: () => true, // Free tier always available
    search: async (q: string, limit: number) => {
      const { jinaSearch } = await import("./providers/jina.server");
      return jinaSearch(q, { limit });
    },
  },
];

function decodeRedirect(href: string): string {
  try {
    if (href.includes("duckduckgo.com/l/")) {
      const u = new URL(href.startsWith("//") ? `https:${href}` : href);
      return decodeURIComponent(u.searchParams.get("uddg") ?? href);
    }
    if (href.startsWith("/url?q=")) {
      return decodeURIComponent(href.slice(7).split("&")[0]);
    }
  } catch {
    /* fall through to the raw href */
  }
  return href;
}

async function fetchEngine(
  url: string,
  timeoutMs: number,
  retries = 1,
): Promise<{ ok: boolean; status: number; text: string; error?: string }> {
  const res = await relaiFetch(url, { timeoutMs, retries, cache: false });
  return { ok: res.ok, status: res.status, text: res.text, error: res.error };
}

const SEARCH_TIMEOUT_MS = 20_000;
const AGGREGATE_TIMEOUT_MS = 45_000;

/**
 * Rate-limit gate for Bing specifically. Bing returns stale/redirect results
 * when queried rapidly from the same IP. A 1.5s gap between calls keeps it
 * responsive without significantly slowing the overall search budget.
 */
const BING_MIN_INTERVAL_MS = 1_500;
let lastBingCallAt = 0;
async function bingGate(): Promise<void> {
  const now = Date.now();
  const elapsed = now - lastBingCallAt;
  if (elapsed < BING_MIN_INTERVAL_MS) {
    await new Promise((r) => setTimeout(r, BING_MIN_INTERVAL_MS - elapsed));
  }
  lastBingCallAt = Date.now();
}

type RunProvider =
  | { name: string; kind: "api"; search: () => Promise<{ hits: RelAIWebHit[]; error?: string }> }
  | { name: string; kind: "html"; engine: Engine; build: () => string };

/**
 * Build the provider run-list for one query: configured API providers FIRST
 * (reliable on datacenter IPs, better results), then the free HTML engines as
 * the keyless fallback.
 */
function buildRunProviders(searchQuery: string, limit: number, page: number): RunProvider[] {
  return [
    ...API_SEARCH_PROVIDERS.filter((p) => p.configured()).map((p) => ({
      name: p.name,
      kind: "api" as const,
      search: async () => {
        const r = await p.search(searchQuery, limit);
        return { hits: r.hits, error: r.error };
      },
    })),
    ...ENGINES.map((engine) => ({
      name: engine.name,
      kind: "html" as const,
      engine,
      build: () => engine.build(searchQuery, page),
    })),
  ];
}

async function runHtmlEngine(
  engine: Engine,
  build: () => string,
  timeoutMs: number,
  retries: number,
): Promise<{ hits: RelAIWebHit[]; error?: string }> {
  if (engine.name === "bing") await bingGate();
  const url = build();
  const res = await fetchEngine(url, timeoutMs, retries);
  if (!res.ok || !res.text) {
    return {
      hits: [],
      error: res.error
        ? `${engine.name}: ${res.error}`
        : `${engine.name}: HTTP ${res.status || "no response"}`,
    };
  }
  if (res.text.length < 100) {
    return { hits: [], error: `${engine.name}: response too short (${res.text.length} chars)` };
  }
  const parsed = engine.parse(res.text);
  if (parsed.length === 0) {
    return { hits: [], error: `${engine.name}: parsed 0 results from ${res.text.length} chars` };
  }
  return { hits: parsed };
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    promise,
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms),
    ),
  ]);
}

/**
 * Public web search. Tries engines in succession with a hard timeout so a
 * single blocked provider never takes RelAI offline or hangs a run.
 *
 * `aggregate` (lead discovery) fans out across ALL providers and aggregates
 * the deduped hits — one engine's blind spot must not starve the candidate
 * pool. The fast default path keeps first-hit-wins for chat/search.
 */
export async function relaiSearch(
  query: string,
  opts: {
    limit?: number;
    site?: string;
    rank?: boolean;
    mode?: string;
    /** SERP page to request (1 = first). HTML engines paginate via their own params. */
    page?: number;
    /**
     * Fan out across ALL providers (not first-hit-wins) and aggregate the
     * deduped hits up to `limit`. Used by lead discovery, where a single
     * engine's blind spot must not starve the candidate pool. Off by default
     * so chat/search keep the fast single-provider path.
     */
    aggregate?: boolean;
  } = {},
): Promise<{
  hits: RelAIWebHit[];
  engine: string;
  query: string;
  tried: number;
  errors: string[];
}> {
  const q = opts.site ? `site:${opts.site} ${query}` : query;
  const limit = Math.min(Math.max(opts.limit ?? 10, 1), 60);
  const page = Math.max(opts.page ?? 1, 1);
  const seen = new Set<string>();
  const allHits: RelAIWebHit[] = [];
  const errors: string[] = [];

  const searchQuery = opts.mode === "news"
    ? q + (q.includes("after:") || q.includes("before:") ? "" : " after:2025-01-01")
    : q;

  const searchStarted = Date.now();
  const budget = opts.aggregate ? AGGREGATE_TIMEOUT_MS : SEARCH_TIMEOUT_MS;

  const runProviders = buildRunProviders(searchQuery, limit, page);
  const htmlTimeout = opts.aggregate ? 8_000 : 10_000;
  const htmlRetries = opts.aggregate ? 1 : 2;

  let bestEngine = "";
  for (const provider of runProviders) {
    if (searchStarted + budget < Date.now()) {
      errors.push("search timeout reached");
      break;
    }
    // Fast path: stop as soon as we have enough. Aggregate mode runs every
    // provider so no single engine's result set decides the candidate pool.
    if (!opts.aggregate && allHits.length >= limit) break;

    let hits: RelAIWebHit[] = [];
    let providerError: string | undefined;
    try {
      if (provider.kind === "api") {
        const r = await provider.search();
        hits = r.hits;
        providerError = r.error;
      } else {
        const r = await runHtmlEngine(provider.engine, provider.build, htmlTimeout, htmlRetries);
        if (r.error) {
          errors.push(r.error);
          continue;
        }
        hits = r.hits;
      }
    } catch (err) {
      errors.push(`${provider.name}: ${err instanceof Error ? err.message : String(err)}`);
      continue;
    }

    if (providerError) errors.push(providerError);
    if (hits.length === 0 && !providerError) {
      errors.push(`${provider.name}: returned 0 results`);
      continue;
    }
    if (!bestEngine && hits.length > 0) bestEngine = provider.name;

    for (const hit of hits) {
      const key = hit.url.split("#")[0];
      if (seen.has(key)) continue;
      seen.add(key);
      allHits.push(hit);
      if (!opts.aggregate && allHits.length >= limit) break;
    }

    if (!opts.aggregate && allHits.length > 0) {
      if (opts.rank !== false) {
        try {
          const ranked = await rankHits(searchQuery, allHits, { limit });
          if (ranked.length > 0) {
            return {
              hits: ranked.map(({ relevance, ...h }) => ({ ...h, relevance })),
              engine: provider.name,
              query: searchQuery,
              tried: allHits.length,
              errors,
            };
          }
        } catch {
          /* ranking is an enhancement, never a failure mode */
        }
      }
      return {
        hits: allHits,
        engine: provider.name,
        query: searchQuery,
        tried: allHits.length,
        errors,
      };
    }
  }

  return {
    hits: opts.aggregate ? allHits.slice(0, limit) : allHits,
    engine: allHits.length > 0 ? (bestEngine || allHits[0].engine) : "none",
    query: searchQuery,
    tried: allHits.length,
    errors: errors.length > 0 ? errors : ["All engines returned no results"],
  };
}

export interface RelAIWebSearchResult {
  hits: RelAIWebHit[];
  engine: string;
  query: string;
  tried: number;
  errors: string[];
}

export interface ProviderHealth {
  name: string;
  configured: boolean;
  status: "ok" | "error" | "skipped";
  hits: number;
  error?: string;
  ms: number;
}

export interface RelAIHealthReport {
  providers: ProviderHealth[];
  totalHits: number;
  anyWorking: boolean;
  note: string;
}

/**
 * Probe every search provider with a small query and report per-provider
 * health. Used by the Lead Finder diagnostics panel; runs on-demand only.
 */
export async function relaiHealthCheck(): Promise<RelAIHealthReport> {
  const query = "technology startups funding";
  const probeLimit = 3;
  const providers = buildRunProviders(query, probeLimit, 1);
  const results: ProviderHealth[] = await Promise.all(
    providers.map(async (p) => {
      const started = Date.now();
      try {
        let hits = 0;
        let error: string | undefined;
        if (p.kind === "api") {
          const r = await withTimeout(p.search(), 10_000, p.name);
          hits = r.hits.length;
          error = r.error;
        } else {
          const r = await withTimeout(runHtmlEngine(p.engine, p.build, 10_000, 1), 10_000, p.name);
          hits = r.hits.length;
          error = r.error;
        }
        return {
          name: p.name,
          configured: true,
          status: error ? "error" : hits > 0 ? "ok" : "error",
          hits,
          error,
          ms: Date.now() - started,
        };
      } catch (err) {
        return {
          name: p.name,
          configured: true,
          status: "error",
          hits: 0,
          error: err instanceof Error ? err.message : String(err),
          ms: Date.now() - started,
        };
      }
    }),
  );

  const configured = API_SEARCH_PROVIDERS.filter((p) => p.configured()).map((p) => p.name);
  const totalHits = results.reduce((sum, r) => sum + r.hits, 0);
  const working = results.filter((r) => r.status === "ok");
  const note = working.length === 0
    ? "No search provider returned results. API providers will need API keys; HTML engines are often blocked on datacenter IPs."
    : `${working.length} of ${results.length} providers working (${configured.length} API key(s) configured).`;

  return {
    providers: results,
    totalHits,
    anyWorking: working.length > 0,
    note,
  };
}

/** Read one public page and return readable text. */
export async function relaiReadPage(
  url: string,
  opts: { maxChars?: number } = {},
): Promise<RelAIPage> {
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error(`Not a valid URL: ${url}`);
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("RelAI only reads http(s) URLs.");
  }
  if (isPrivateHost(parsed.hostname)) {
    throw new Error("RelAI will not fetch private or internal hosts.");
  }

  const max = Math.min(Math.max(opts.maxChars ?? 6000, 500), 20_000);
  const res = await relaiFetch(parsed.toString(), { timeoutMs: 15_000, retries: 2 });
  if (!res.ok) {
    throw new Error(
      res.blocked
        ? `${parsed.host} challenged or blocked the request after retries.`
        : `Could not read ${parsed.host} (HTTP ${res.status || "no response"}).`,
    );
  }

  const isJson =
    res.contentType.includes("json") || /^\s*[[{]/.test(res.text.slice(0, 200));
  const doc = isJson
    ? jsonToMarkdown(res.text, res.url)
    : htmlToMarkdown(res.text, res.url, { maxChars: max });

  const [normalized] = normalizeBatch([
    {
      url: doc.url,
      title: doc.meta.title,
      description: doc.meta.description,
      markdown: doc.markdown,
      text: doc.text,
      source: "read_url",
      fetchedAt: res.fetchedAt,
    },
  ]).docs;

  return {
    url: doc.url,
    title: normalized?.title ?? doc.meta.title,
    text: (normalized?.markdown ?? doc.markdown).slice(0, max),
    status: res.status,
    truncated: doc.truncated,
  };
}

export interface RelAIOsintSweep {
  target: string;
  queries: string[];
  hits: RelAIWebHit[];
  bySource: Record<string, RelAIWebHit[]>;
  emails: string[];
  pagesRead: Array<{ url: string; title: string; excerpt: string }>;
}

const OSINT_SOURCES = [
  "linkedin.com",
  "github.com",
  "x.com",
  "reddit.com",
  "crunchbase.com",
  "news.ycombinator.com",
];

/**
 * Keyless OSINT sweep: fans the target across the open web and the public
 * profile sources, then reads the strongest page for contact evidence.
 */
export async function relaiOsintSweep(
  target: string,
  opts: { readPages?: number } = {},
): Promise<RelAIOsintSweep> {
  const clean = target.trim();
  const queries = [
    clean,
    `"${clean}" founder OR CEO OR owner`,
    `"${clean}" contact email`,
    ...OSINT_SOURCES.map((s) => `site:${s} ${clean}`),
  ];

  const seen = new Set<string>();
  const hits: RelAIWebHit[] = [];
  const bySource: Record<string, RelAIWebHit[]> = {};

  const runs = await Promise.all(
    queries.map((q) => relaiSearch(q, { limit: 6 }).catch(() => null)),
  );

  for (const run of runs) {
    if (!run?.hits) continue;
    for (const hit of run.hits) {
      const key = hit.url.split("#")[0];
      if (seen.has(key)) continue;
      seen.add(key);
      hits.push(hit);
      let host = "web";
      try {
        host = new URL(hit.url).hostname.replace(/^www\./, "");
      } catch {
        /* keep the default bucket */
      }
      (bySource[host] ??= []).push(hit);
    }
  }

  const readCount = Math.min(Math.max(opts.readPages ?? 2, 0), 5);
  const pagesRead: RelAIOsintSweep["pagesRead"] = [];
  const emails = new Set<string>();

  for (const hit of hits.slice(0, readCount)) {
    try {
      const page = await relaiReadPage(hit.url, { maxChars: 8000 });
      pagesRead.push({
        url: page.url,
        title: page.title,
        excerpt: page.text.slice(0, 1200),
      });
      for (const e of page.text.match(
        /[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}/gi,
      ) ?? []) {
        if (!/\.(png|jpg|jpeg|gif|webp)$/i.test(e)) emails.add(e.toLowerCase());
      }
    } catch {
      /* an unreadable page is not a failed sweep */
    }
  }

  return {
    target: clean,
    queries,
    hits,
    bySource,
    emails: [...emails].slice(0, 20),
    pagesRead,
  };
}
