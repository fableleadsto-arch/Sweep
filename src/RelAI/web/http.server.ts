/**
 * RelAI resilient HTTP layer (server-only).
 *
 * One gateway every crawl, search and extraction call goes through, so the
 * reliability rules live in exactly one place:
 *   - per-host token-bucket rate limiting + global concurrency ceiling
 *   - retry with exponential backoff and jitter on 429/5xx/network errors
 *   - user-agent rotation and optional proxy for blocked hosts
 *   - CAPTCHA / interstitial detection so a challenge page is never indexed
 *   - SSRF guard: no private, link-local or metadata hosts, ever
 *   - short-lived response cache so a re-read inside one run costs nothing
 *
 * Reference architecture: steel (sandboxed, proxied, rate-limited fetching).
 */

import { isPrivateHost } from "@/lib/url-safety";

export { isPrivateHost };

const USER_AGENTS = [
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36",
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
];

const HOST_MIN_INTERVAL_MS = 700;
const MAX_CONCURRENT = 6;
const CACHE_TTL_MS = 120_000;
const CACHE_MAX_ENTRIES = 200;

export interface FetchResult {
  ok: boolean;
  status: number;
  url: string;
  text: string;
  contentType: string;
  blocked: boolean;
  attempts: number;
  error?: string;
  fetchedAt: string;
}

export interface FetchOptions {
  timeoutMs?: number;
  retries?: number;
  method?: "GET" | "POST";
  body?: string;
  headers?: Record<string, string>;
  cache?: boolean;
  /** Bypass the polite delay — only for the first hit on a fresh host. */
  maxBytes?: number;
}

/* ------------------------------------------------------------------ *
 * Scheduling: per-host spacing plus a global in-flight ceiling.
 * ------------------------------------------------------------------ */

const lastHit = new Map<string, number>();
const hostFailures = new Map<string, number>();
let inFlight = 0;
const waiters: Array<() => void> = [];

async function acquireSlot(): Promise<void> {
  if (inFlight < MAX_CONCURRENT) {
    inFlight++;
    return;
  }
  await new Promise<void>((resolve) => waiters.push(resolve));
  inFlight++;
}

function releaseSlot(): void {
  inFlight = Math.max(0, inFlight - 1);
  const next = waiters.shift();
  if (next) next();
}

async function politeDelay(host: string): Promise<void> {
  // Hosts that have already blocked us get backed off harder.
  const penalty = (hostFailures.get(host) ?? 0) * 400;
  const gap = HOST_MIN_INTERVAL_MS + penalty;
  const wait = gap - (Date.now() - (lastHit.get(host) ?? 0));
  if (wait > 0) await sleep(wait);
  lastHit.set(host, Date.now());
}

export function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/* ------------------------------------------------------------------ *
 * Safety
 * ------------------------------------------------------------------ */

export function assertPublicUrl(raw: string): URL {
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    throw new Error(`Not a valid URL: ${raw}`);
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("RelAI only fetches http(s) URLs.");
  }
  if (isPrivateHost(parsed.hostname)) {
    throw new Error("RelAI will not fetch private or internal hosts.");
  }
  return parsed;
}

const CAPTCHA_MARKERS = [
  "captcha",
  "are you a robot",
  "unusual traffic",
  "verify you are human",
  "cf-browser-verification",
  "checking your browser before",
  "access denied",
  "enable javascript and cookies to continue",
];

/** A challenge page is a 200 that carries no content — treat it as a block. */
export function looksBlocked(status: number, text: string): boolean {
  if (status === 403 || status === 429 || status === 503) return true;
  const head = text.slice(0, 4000).toLowerCase();
  if (head.length < 200) return false;
  return CAPTCHA_MARKERS.some((m) => head.includes(m));
}

/* ------------------------------------------------------------------ *
 * Cache
 * ------------------------------------------------------------------ */

const cache = new Map<string, { at: number; value: FetchResult }>();

function cacheGet(key: string): FetchResult | null {
  const entry = cache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.at > CACHE_TTL_MS) {
    cache.delete(key);
    return null;
  }
  return entry.value;
}

function cacheSet(key: string, value: FetchResult): void {
  if (cache.size >= CACHE_MAX_ENTRIES) {
    const oldest = cache.keys().next().value;
    if (oldest) cache.delete(oldest);
  }
  cache.set(key, { at: Date.now(), value });
}

/* ------------------------------------------------------------------ *
 * The fetch itself
 * ------------------------------------------------------------------ */

function proxied(url: string): string {
  // RELAI_PROXY_URL is a prefix-style fetch proxy, e.g.
  // https://proxy.example.com/?url=  — left unset, RelAI fetches directly.
  const proxy = process.env.RELAI_PROXY_URL;
  if (!proxy) return url;
  return proxy.includes("{url}")
    ? proxy.replace("{url}", encodeURIComponent(url))
    : `${proxy}${encodeURIComponent(url)}`;
}

function headersFor(attempt: number, extra?: Record<string, string>): HeadersInit {
  return {
    "User-Agent": USER_AGENTS[attempt % USER_AGENTS.length],
    Accept: "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    ...extra,
  };
}

/**
 * Fetch a public URL with rate limiting, retries and block detection.
 * Never throws on a network failure — it reports it in the result.
 */
export async function relaiFetch(
  rawUrl: string,
  opts: FetchOptions = {},
): Promise<FetchResult> {
  const now = () => new Date().toISOString();
  let parsed: URL;
  try {
    parsed = assertPublicUrl(rawUrl);
  } catch (err) {
    return {
      ok: false,
      status: 0,
      url: rawUrl,
      text: "",
      contentType: "",
      blocked: false,
      attempts: 0,
      error: err instanceof Error ? err.message : String(err),
      fetchedAt: now(),
    };
  }

  const url = parsed.toString();
  const host = parsed.host;
  const method = opts.method ?? "GET";
  const cacheKey = `${method} ${url} ${opts.body ?? ""}`;
  const useCache = opts.cache !== false && method === "GET";
  if (useCache) {
    const hit = cacheGet(cacheKey);
    if (hit) return hit;
  }

  const retries = Math.min(Math.max(opts.retries ?? 2, 0), 4);
  const timeoutMs = opts.timeoutMs ?? 15_000;
  const maxBytes = opts.maxBytes ?? 2_000_000;

  let last: FetchResult = {
    ok: false,
    status: 0,
    url,
    text: "",
    contentType: "",
    blocked: false,
    attempts: 0,
    error: "not attempted",
    fetchedAt: now(),
  };

  for (let attempt = 0; attempt <= retries; attempt++) {
    await acquireSlot();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      await politeDelay(host);
      let requestUrl = url;
      let res!: Response;
      for (let redirects = 0; redirects <= 4; redirects++) {
        const target = attempt > 0 ? proxied(requestUrl) : requestUrl;
        res = await fetch(target, {
          method,
          headers: headersFor(attempt, opts.headers),
          body: opts.body,
          signal: controller.signal,
          redirect: "manual",
        });
        if (res.status < 300 || res.status >= 400) break;
        const location = res.headers.get("location");
        if (!location) throw new Error("Redirect response did not include a location.");
        if (redirects === 4) throw new Error("Too many redirects.");
        requestUrl = assertPublicUrl(new URL(location, requestUrl).toString()).toString();
      }
      const text = await readCapped(res, maxBytes);
      const blocked = looksBlocked(res.status, text);
      last = {
        ok: res.ok && !blocked,
        status: res.status,
        url: requestUrl,
        text,
        contentType: res.headers.get("content-type") ?? "",
        blocked,
        attempts: attempt + 1,
        error: blocked ? "blocked or challenged by the host" : undefined,
        fetchedAt: now(),
      };
      if (last.ok) {
        hostFailures.delete(host);
        if (useCache) cacheSet(cacheKey, last);
        return last;
      }
      hostFailures.set(host, (hostFailures.get(host) ?? 0) + 1);
      // 4xx that is not a throttle will not change on retry.
      if (!blocked && res.status >= 400 && res.status < 500 && res.status !== 429) {
        return last;
      }
    } catch (err) {
      const aborted = (err as { name?: string })?.name === "AbortError";
      last = {
        ok: false,
        status: 0,
        url,
        text: "",
        contentType: "",
        blocked: false,
        attempts: attempt + 1,
        error: aborted ? `timed out after ${timeoutMs}ms` : String(err),
        fetchedAt: now(),
      };
      hostFailures.set(host, (hostFailures.get(host) ?? 0) + 1);
    } finally {
      clearTimeout(timer);
      releaseSlot();
    }

    if (attempt < retries) {
      const backoff = 500 * 2 ** attempt + Math.random() * 400;
      await sleep(backoff);
    }
  }

  return last;
}

/** Run tasks with a bounded worker pool — the crawler's concurrency primitive. */
export async function pooled<T, R>(
  items: T[],
  limit: number,
  worker: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const size = Math.min(Math.max(limit, 1), MAX_CONCURRENT);
  const out = new Array<R>(items.length);
  let cursor = 0;
  await Promise.all(
    Array.from({ length: Math.min(size, items.length) }, async () => {
      while (cursor < items.length) {
        const i = cursor++;
        out[i] = await worker(items[i], i);
      }
    }),
  );
  return out;
}

/**
 * Read a response body as text, streaming with a hard byte cap.
 *
 * The old `await res.text()` buffered the entire body before slicing, so a
 * multi-GB response could balloon heap usage. This reader aborts as soon as
 * `maxBytes` is exceeded and never holds more than a capped chunk in memory.
 */
async function readCapped(res: Response, maxBytes: number): Promise<string> {
  const reader = res.body?.getReader();
  if (!reader) return res.text().catch(() => "");

  const decoder = new TextDecoder();
  let text = "";
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      text += decoder.decode(value, { stream: true });
      if (total >= maxBytes) {
        await reader.cancel();
        break;
      }
    }
  } catch {
    return text;
  }
  return text.length > maxBytes ? text.slice(0, maxBytes) : text;
}
