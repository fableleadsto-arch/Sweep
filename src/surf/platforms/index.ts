/**
 * Relay Surf — platform adapter registry (server-only).
 *
 * Every platform adapter implements the same `PlatformAdapter` contract so the
 * research engine never has to know platform specifics. Search resolution:
 *   1. dedicated adapter's native/public search,
 *   2. honest fallbacks (web index) when native search isn't accessible,
 *   3. never a fabricated result.
 */
import type { PageData, PlatformAdapter, PlatformSearchOptions, SearchResult, SurfPlatform } from "../types";
import { redditAdapter } from "./reddit";
import { githubAdapter } from "./github";
import { youtubeAdapter } from "./youtube";
import { xAdapter } from "./x";
import { instagramAdapter } from "./instagram";
import { genericAdapter } from "./generic";

const ADAPTERS: PlatformAdapter[] = [
  redditAdapter,
  githubAdapter,
  youtubeAdapter,
  xAdapter,
  instagramAdapter,
  genericAdapter,
];

export function getAdapter(platform: SurfPlatform | string): PlatformAdapter | undefined {
  if (platform === "generic") return genericAdapter;
  return ADAPTERS.find((a) => a.platform === platform);
}

/** Find an adapter that handles a URL (generic always matches last). */
export function adapterForUrl(url: string): PlatformAdapter {
  for (const adapter of ADAPTERS) {
    if (adapter !== genericAdapter && adapter.canHandle(url)) return adapter;
  }
  return genericAdapter;
}

export interface PlatformSearchInput {
  platform?: SurfPlatform;
  query: string;
  options?: PlatformSearchOptions;
}

/** Search a platform (or any website) with honest access reporting. */
export async function searchPlatform(
  input: PlatformSearchInput,
): Promise<{ results: SearchResult[]; platform: SurfPlatform; note?: string; native: boolean }> {
  const platform = input.platform ?? "generic";
  const adapter = getAdapter(platform) ?? genericAdapter;
  const res = await adapter.search(input.query, input.options ?? {});
  return {
    results: res.results,
    platform: adapter.platform,
    note: res.note,
    native: adapter.platform === "reddit" || adapter.platform === "github" || res.accessMode === "authenticated",
  };
}

/** Extract a platform page via the adapter that handles its URL. */
export async function extractPlatformPage(url: string): Promise<PageData | null> {
  return adapterForUrl(url).extractPage(url);
}

export { genericAdapter };
