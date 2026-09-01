/**
 * Relay Surf — public research cache (server-only).
 *
 * Short-TTL in-memory cache for public, reusable research artifacts (search
 * results, extracted pages, document text). Content-keyed and bounded so idle
 * entries can't pile up. Only *public* data is cached — anything that could be
 * user-private is keyed by its content and never stored globally.
 */

interface CacheEntry<T> {
  value: T;
  expiresAt: number;
}

const store = new Map<string, CacheEntry<unknown>>();
const DEFAULT_TTL_MS = 5 * 60 * 1000;
const MAX_ENTRIES = 300;

export function cacheGet<T>(key: string): T | undefined {
  const entry = store.get(key) as CacheEntry<T> | undefined;
  if (!entry) return undefined;
  if (Date.now() > entry.expiresAt) {
    store.delete(key);
    return undefined;
  }
  return entry.value;
}

export function cacheSet<T>(key: string, value: T, ttlMs = DEFAULT_TTL_MS): void {
  if (store.size >= MAX_ENTRIES) {
    const oldest = [...store.entries()].sort((a, b) => a[1].expiresAt - b[1].expiresAt)[0];
    if (oldest) store.delete(oldest[0]);
  }
  store.set(key, { value, expiresAt: Date.now() + ttlMs });
}

export async function cacheWrap<T>(key: string, producer: () => Promise<T>, ttlMs = DEFAULT_TTL_MS): Promise<T> {
  const hit = cacheGet<T>(key);
  if (hit !== undefined) return hit;
  const value = await producer();
  cacheSet(key, value, ttlMs);
  return value;
}

export function cacheKey(...parts: Array<string | number | boolean | undefined>): string {
  return parts
    .filter((part): part is string | number | boolean => Boolean(part))
    .map(String)
    .join("|")
    .slice(0, 400);
}

export function cacheStats(): { entries: number; max: number } {
  return { entries: store.size, max: MAX_ENTRIES };
}
