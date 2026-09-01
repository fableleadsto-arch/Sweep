/**
 * RelAI async crawler (server-only).
 *
 * Reference architecture: crawl4ai. Breadth-first, bounded concurrency, one
 * output shape: normalized Markdown documents. It never emits raw HTML and
 * everything it yields has already been through the normalization step.
 */
import { relaiFetch, pooled, assertPublicUrl } from "./http.server";
import { htmlToMarkdown, jsonToMarkdown, type MarkdownDoc } from "./markdown.server";
import {
  canonicalizeUrl,
  normalizeBatch,
  type JsonValue,
  type NormalizedDoc,
} from "./normalize.server";

export interface CrawlOptions {
  maxPages?: number;
  maxDepth?: number;
  concurrency?: number;
  sameHostOnly?: boolean;
  include?: string;
  exclude?: string;
  maxChars?: number;
  budgetMs?: number;
}

export interface CrawlResult {
  root: string;
  docs: NormalizedDoc[];
  visited: number;
  blocked: string[];
  failed: Array<{ url: string; error: string }>;
  duplicatesDropped: number;
  elapsedMs: number;
}

const SKIP_EXT = /\.(pdf|zip|gz|png|jpe?g|gif|webp|svg|ico|mp4|mp3|wav|css|js|woff2?|ttf)(\?|$)/i;

/** Fetch one URL and return it as a Markdown document. */
export async function fetchAsMarkdown(
  url: string,
  opts: { maxChars?: number; retries?: number; timeoutMs?: number } = {},
): Promise<{ doc: MarkdownDoc; status: number; fetchedAt: string; blocked: boolean } | { error: string; status: number; blocked: boolean }> {
  const res = await relaiFetch(url, {
    timeoutMs: opts.timeoutMs ?? 15_000,
    retries: opts.retries ?? 2,
  });
  if (!res.ok) {
    return {
      error: res.error ?? `HTTP ${res.status}`,
      status: res.status,
      blocked: res.blocked,
    };
  }
  const isJson =
    res.contentType.includes("json") || /^\s*[[{]/.test(res.text.slice(0, 200));
  const doc = isJson
    ? jsonToMarkdown(res.text, res.url)
    : htmlToMarkdown(res.text, res.url, { maxChars: opts.maxChars });
  return { doc, status: res.status, fetchedAt: res.fetchedAt, blocked: false };
}

/**
 * Crawl from a root URL. Bounded by pages, depth, host and wall-clock budget
 * so a runaway site can never consume the request.
 */
export async function relaiCrawl(
  rootUrl: string,
  opts: CrawlOptions = {},
): Promise<CrawlResult> {
  const started = Date.now();
  const root = assertPublicUrl(rootUrl);
  const maxPages = Math.min(Math.max(opts.maxPages ?? 10, 1), 60);
  const maxDepth = Math.min(Math.max(opts.maxDepth ?? 1, 0), 4);
  const concurrency = Math.min(Math.max(opts.concurrency ?? 4, 1), 6);
  const budgetMs = Math.min(Math.max(opts.budgetMs ?? 60_000, 5_000), 180_000);
  const sameHostOnly = opts.sameHostOnly !== false;
  const include = opts.include ? safeRegex(opts.include) : null;
  const exclude = opts.exclude ? safeRegex(opts.exclude) : null;

  const seen = new Set<string>([canonicalizeUrl(root.toString())]);
  const blocked: string[] = [];
  const failed: Array<{ url: string; error: string }> = [];
  const raws: Array<{ url: string; title: string; description: string; markdown: string; text: string; fetchedAt: string; metadata: Record<string, JsonValue> }> = [];

  let frontier = [root.toString()];

  for (let depth = 0; depth <= maxDepth && frontier.length > 0; depth++) {
    if (Date.now() - started > budgetMs || raws.length >= maxPages) break;
    const batch = frontier.slice(0, maxPages - raws.length);
    frontier = [];

    const results = await pooled(batch, concurrency, async (url) => {
      if (Date.now() - started > budgetMs) return null;
      return { url, out: await fetchAsMarkdown(url, { maxChars: opts.maxChars }) };
    });

    for (const item of results) {
      if (!item) continue;
      const { url, out } = item;
      if ("error" in out) {
        if (out.blocked) blocked.push(url);
        else failed.push({ url, error: out.error });
        continue;
      }
      raws.push({
        url: out.doc.url,
        title: out.doc.meta.title,
        description: out.doc.meta.description,
        markdown: out.doc.markdown,
        text: out.doc.text,
        fetchedAt: out.fetchedAt,
        metadata: {
          depth,
          status: out.status,
          siteName: out.doc.meta.siteName,
          author: out.doc.meta.author,
          publishedAt: out.doc.meta.publishedAt,
          jsonLd: JSON.parse(
            JSON.stringify(out.doc.meta.jsonLd.slice(0, 3)),
          ) as JsonValue,
        },
      });

      if (depth === maxDepth) continue;
      for (const link of out.doc.links) {
        const canonical = canonicalizeUrl(link.url);
        if (seen.has(canonical) || SKIP_EXT.test(canonical)) continue;
        let host: string;
        try {
          host = new URL(canonical).hostname.replace(/^www\./, "");
        } catch {
          continue;
        }
        if (sameHostOnly && host !== root.hostname.replace(/^www\./, "")) continue;
        if (include && !include.test(canonical)) continue;
        if (exclude && exclude.test(canonical)) continue;
        seen.add(canonical);
        frontier.push(canonical);
      }
    }
  }

  const { docs, dropped } = normalizeBatch(
    raws.map((r) => ({ ...r, source: "crawl" })),
    { nearDistance: 3, minWords: 12 },
  );

  return {
    root: root.toString(),
    docs: docs.slice(0, maxPages),
    visited: raws.length,
    blocked,
    failed,
    duplicatesDropped: dropped,
    elapsedMs: Date.now() - started,
  };
}

function safeRegex(pattern: string): RegExp | null {
  try {
    return new RegExp(pattern, "i");
  } catch {
    return null;
  }
}
