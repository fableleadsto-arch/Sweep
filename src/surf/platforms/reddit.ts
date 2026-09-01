/**
 * Relay Surf — Reddit adapter (server-only).
 *
 * Uses Reddit's public JSON endpoints (no key, no login) to search and read
 * threads where content is publicly accessible. Honest access reporting:
 * every result carries `accessMode: "public"` and platform `reddit`.
 *
 * Never bypasses authentication; private/quarantined subs simply return no
 * public data. When the JSON endpoints are blocked, the adapter falls back to
 * site-scoped web search and labels it as such.
 */
import { relaiFetch } from "@/RelAI/web/http.server";
import { assertSafeUrl } from "../guard";
import type { PageData, PlatformAdapter, PlatformSearchOptions, SearchResult } from "../types";

const REDDIT_JSON_HEADERS = { "User-Agent": "RelaySurf/1.0 (research agent)" };

interface RedditListingChild {
  data?: {
    title?: string;
    url?: string;
    permalink?: string;
    selftext?: string;
    /** Comment body (present on comment children in a thread listing). */
    body?: string;
    score?: number;
    num_comments?: number;
    subreddit?: string;
    created_utc?: number;
    author?: string;
    id?: string;
    is_self?: boolean;
  };
}

async function searchJson(query: string, subreddit?: string, limit = 10): Promise<SearchResult[]> {
  const params = new URLSearchParams({
    q: query,
    limit: String(Math.min(limit, 25)),
    sort: "relevance",
    raw_json: "1",
  });
  const base = subreddit
    ? `https://www.reddit.com/r/${encodeURIComponent(subreddit)}/search.json`
    : "https://www.reddit.com/search.json";
  const url = `${base}?${params.toString()}`;

  const res = await relaiFetch(url, {
    headers: REDDIT_JSON_HEADERS,
    timeoutMs: 12_000,
    retries: 1,
    cache: true,
  });
  if (!res.ok) return [];

  let data: { data?: { children?: RedditListingChild[] } };
  try {
    data = JSON.parse(res.text);
  } catch {
    return [];
  }

  const out: SearchResult[] = [];
  for (const child of data.data?.children ?? []) {
    const d = child.data;
    if (!d?.permalink) continue;
    out.push({
      url: `https://www.reddit.com${d.permalink}`,
      title: d.title ?? "",
      snippet: (d.selftext ?? "").slice(0, 300),
      provider: "reddit",
      accessMode: "public",
      platform: "reddit",
      metadata: {
        subreddit: d.subreddit ?? "",
        author: d.author ?? "",
        score: String(d.score ?? 0),
        comments: String(d.num_comments ?? 0),
        created: d.created_utc ? new Date(d.created_utc * 1000).toISOString() : "",
      },
    });
    if (out.length >= limit) break;
  }
  return out;
}

async function siteScopedFallback(query: string, limit: number): Promise<SearchResult[]> {
  const { relaiSearch } = await import("@/RelAI/web/search.server");
  const res = await relaiSearch(query, { site: "reddit.com", limit });
  return res.hits.map((h) => ({
    url: h.url,
    title: h.title,
    snippet: h.snippet,
    provider: "web-index",
    accessMode: "public",
    platform: "reddit",
  }));
}

export const redditAdapter: PlatformAdapter = {
  platform: "reddit",
  canHandle(url: string) {
    try {
      return new URL(url).hostname.replace(/^www\.|^old\.|^new\./, "") === "reddit.com";
    } catch {
      return false;
    }
  },
  async search(query, options: PlatformSearchOptions = {}) {
    try {
      const results = await searchJson(query, options.subreddit, options.limit ?? 10);
      if (results.length > 0) {
        return { results, accessMode: "public" };
      }
      // JSON search empty or blocked → honest site-scoped fallback.
      const fallback = await siteScopedFallback(query, options.limit ?? 10);
      return {
        results: fallback,
        accessMode: "public",
        note:
          fallback.length > 0
            ? "Reddit's public search API did not respond; results come from public web indexing of reddit.com."
            : "Reddit public search returned no results.",
      };
    } catch {
      const fallback = await siteScopedFallback(query, options.limit ?? 10);
      return {
        results: fallback,
        accessMode: "public",
        note: "Reddit public search is currently unavailable; results come from public web indexing.",
      };
    }
  },
  async extractPage(url) {
    const safeUrl = assertSafeUrl(url);
    if (!this.canHandle(safeUrl)) return null;
    const jsonUrl = safeUrl.replace(/\/$/, "") + ".json";
    const res = await relaiFetch(jsonUrl, {
      headers: REDDIT_JSON_HEADERS,
      timeoutMs: 12_000,
      retries: 1,
      cache: true,
    });
    if (!res.ok) return null;

    let parsed: { data?: { children?: RedditListingChild[] } };
    try {
      parsed = JSON.parse(res.text);
    } catch {
      return null;
    }
    const listing = parsed.data?.children ?? [];
    const post = listing[0]?.data;

    if (!post?.title) return null;

    const comments = listing
      .slice(1)
      .map((c) => c.data)
      .filter((c): c is NonNullable<RedditListingChild["data"]> => Boolean(c))
      .filter((c) => c.body && !c.body.startsWith("[deleted]"))
      .slice(0, 40)
      .map((c) => `- (${c.author ?? "anonymous"}) ${(c.body ?? "").slice(0, 600)}`)
      .join("\n");

    const text = [
      `# ${post.title}`,
      `r/${post.subreddit ?? ""} · u/${post.author ?? ""} · ${post.created_utc ? new Date(post.created_utc * 1000).toISOString() : ""}`,
      ``,
      (post.selftext ?? "").slice(0, 4000),
      ``,
      comments ? `## Comments (public)\n${comments}` : "## Comments\n(no public comments available)",
    ].join("\n");

    const page: PageData = {
      url: safeUrl,
      title: post.title,
      text,
      links: [],
      headings: [{ level: 1, text: post.title }],
      metadata: {
        subreddit: post.subreddit ?? "",
        author: post.author ?? "",
        score: String(post.score ?? 0),
        comments: String(post.num_comments ?? 0),
      },
      truncated: false,
      fetchedAt: new Date().toISOString(),
      status: res.status,
      contentType: "application/json",
      accessMode: "public",
    };
    return page;
  },
};
