/**
 * Relay Surf — source quality scoring (server-only).
 *
 * Every source gets a score across relevance, authority, freshness,
 * directness, and an overall. The research engine prefers primary/official
 * sources when factual accuracy matters and community sources for sentiment.
 */
import { PLATFORMS, type SearchResult, type Source, type SourceScore, type SurfPlatform } from "../types";

/** Coerce a free-form platform string onto the known SurfPlatform set. */
function toSurfPlatform(platform?: string): SurfPlatform | undefined {
  return platform && (PLATFORMS as readonly string[]).includes(platform)
    ? (platform as SurfPlatform)
    : undefined;
}

/** Domain heuristics for authority — primary sources rank above aggregators. */
const AUTHORITY_DOMAINS: Record<string, number> = {
  "github.com": 0.9,
  "developer.mozilla.org": 0.95,
  "react.dev": 0.95,
  "nodejs.org": 0.9,
  "python.org": 0.9,
  "stackoverflow.com": 0.7,
  "en.wikipedia.org": 0.6,
};

const NEWS_DOMAINS = ["reddit.com", "news.ycombinator.com", "x.com", "twitter.com", "instagram.com"];

export function scoreSource(input: {
  url: string;
  title: string;
  query: string;
  type?: Source["type"];
  retrievedAt: string;
  publishedAt?: string;
  snippet?: string;
}): SourceScore {
  let host = "";
  try {
    host = new URL(input.url).hostname.replace(/^www\./, "");
  } catch {
    /* keep empty */
  }

  const relevance = scoreRelevance(input.url, input.title, input.query);
  const authority = scoreAuthority(host, input.type);
  const freshness = scoreFreshness(input.publishedAt);
  const directness = scoreDirectness(input.type, host);

  const overall =
    relevance * 0.35 + authority * 0.25 + freshness * 0.15 + directness * 0.25;

  return {
    relevance,
    authority,
    freshness,
    directness,
    overall: clamp(overall, 0, 1),
  };
}

export function attachScores(results: SearchResult[], query: string): Source[] {
  const now = new Date().toISOString();
  return results.map((r) => {
    const score = scoreSource({
      url: r.url,
      title: r.title,
      query,
      type: inferSourceType(r.url, r.metadata),
      retrievedAt: now,
      publishedAt: r.publishedAt,
      snippet: r.snippet,
    });
    return {
      title: r.title,
      url: r.url,
      platform: toSurfPlatform(r.platform),
      type: inferSourceType(r.url, r.metadata),
      accessMode: r.accessMode,
      retrievedAt: now,
      score,
    };
  });
}

function scoreRelevance(url: string, title: string, query: string): number {
  const terms = query.toLowerCase().split(/\s+/).filter((t) => t.length > 2).slice(0, 8);
  if (terms.length === 0) return 0.5;
  const haystack = `${url} ${title}`.toLowerCase();
  const hits = terms.filter((t) => haystack.includes(t)).length;
  return clamp(0.3 + (hits / terms.length) * 0.7, 0, 1);
}

function scoreAuthority(host: string, type?: Source["type"]): number {
  if (type === "official" || type === "documentation") return 0.9;
  if (type === "repository") return 0.8;
  if (type === "primary") return 0.85;
  if (AUTHORITY_DOMAINS[host]) return AUTHORITY_DOMAINS[host];
  if (type === "community") return 0.55;
  if (type === "secondary") return 0.4;
  return 0.6;
}

function scoreFreshness(publishedAt?: string): number {
  if (!publishedAt) return 0.5;
  const ageDays = (Date.now() - new Date(publishedAt).getTime()) / 86_400_000;
  if (ageDays < 0) return 0.6;
  if (ageDays <= 30) return 0.95;
  if (ageDays <= 180) return 0.75;
  if (ageDays <= 365) return 0.55;
  return 0.3;
}

function scoreDirectness(type?: Source["type"], host = ""): number {
  if (type === "official" || type === "primary") return 1;
  if (NEWS_DOMAINS.includes(host)) return 0.65;
  if (type === "community") return 0.7;
  if (type === "secondary") return 0.4;
  return 0.6;
}

export function inferSourceType(url: string, metadata?: Record<string, string>): Source["type"] {
  try {
    const host = new URL(url).hostname.replace(/^www\./, "");
    if (host === "github.com") return "repository";
    if (host === "reddit.com" || host === "x.com" || host === "twitter.com" || host === "instagram.com" || host === "youtube.com") {
      return "community";
    }
    if (metadata?.subreddit || metadata?.author) return "community";
  } catch {
    /* fall through */
  }
  if (/\/docs?\//i.test(url) || /\/(docs|documentation|manual|reference)/i.test(url)) return "documentation";
  if (/\/pricing|plans\b/i.test(url)) return "official";
  return "unknown";
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
