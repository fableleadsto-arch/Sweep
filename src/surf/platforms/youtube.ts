/**
 * Relay Surf — YouTube adapter (server-only).
 *
 * No public unauthenticated YouTube *search* API exists, so:
 *   - search: site-scoped web indexing of youtube.com (honest — labeled
 *     "web-index", never claimed as native YouTube search).
 *   - extract: YouTube's public oEmbed + watch-page metadata for titles,
 *     channels, and descriptions of a specific video.
 *   - transcript: best-effort caption extraction from the public watch page;
 *     reported as unavailable when captions aren't accessible.
 */
import { relaiFetch } from "@/RelAI/web/http.server";
import { assertSafeUrl } from "../guard";
import type { PlatformAdapter, PlatformSearchOptions, SearchResult } from "../types";

function videoIdFromUrl(url: string): string | null {
  try {
    const u = new URL(url);
    if (u.hostname === "youtu.be") return u.pathname.split("/")[1] ?? null;
    if (u.hostname.includes("youtube.com")) {
      return u.searchParams.get("v");
    }
  } catch {
    /* not a URL */
  }
  return null;
}

export const youtubeAdapter: PlatformAdapter = {
  platform: "youtube",
  canHandle(url: string) {
    try {
      const host = new URL(url).hostname;
      return host === "youtube.com" || host.endsWith(".youtube.com") || host === "youtu.be";
    } catch {
      return false;
    }
  },
  async search(query, options: PlatformSearchOptions = {}) {
    const { relaiSearch } = await import("@/RelAI/web/search.server");
    const limit = options.limit ?? 8;
    const res = await relaiSearch(query, { site: "youtube.com", limit: Math.min(limit * 2, 20) });
    const results: SearchResult[] = res.hits
      .filter((h) => /youtube\.com\/watch|youtu\.be\//i.test(h.url) || /youtube\.com\/(@|c\/|channel|user)/i.test(h.url))
      .slice(0, limit)
      .map((h) => ({
        url: h.url,
        title: h.title,
        snippet: h.snippet,
        provider: "web-index",
        accessMode: "public",
        platform: "youtube",
      }));
    return {
      results,
      accessMode: "public",
      note:
        "YouTube has no public unauthenticated search API. Results come from public web indexing of youtube.com.",
    };
  },
  async extractPage(url) {
    const safeUrl = assertSafeUrl(url);
    const videoId = videoIdFromUrl(safeUrl);
    if (!videoId) return null;

    // oEmbed is a stable public metadata endpoint.
    const oembed = await relaiFetch(
      `https://www.youtube.com/oembed?url=${encodeURIComponent(`https://www.youtube.com/watch?v=${videoId}`)}&format=json`,
      { timeoutMs: 10_000, retries: 1, cache: true },
    );
    let meta: { title?: string; author_name?: string; author_url?: string } = {};
    if (oembed.ok) {
      try {
        meta = JSON.parse(oembed.text);
      } catch {
        /* keep empty meta */
      }
    }

    // Description + captions from the public watch page (best-effort).
    const watch = await relaiFetch(`https://www.youtube.com/watch?v=${videoId}`, {
      timeoutMs: 15_000,
      retries: 1,
      cache: true,
    });

    let description = "";
    if (watch.ok) {
      const m = watch.text.match(/"shortDescription":"([\s\S]*?)"/);
      if (m) description = m[1].replace(/\\n/g, "\n").slice(0, 2000);
    }

    // Transcript: check for accessible caption tracks. The captions endpoint
    // is only available when the uploader enables them — reported honestly.
    let transcript = "";
    if (watch.ok) {
      const trackRe = /"captionTracks":\[[^\]]*\]/;
      const tracks = watch.text.match(trackRe);
      if (tracks) {
        const baseUrl = tracks[0].match(/"baseUrl":"([^"]+)"/);
        if (baseUrl) {
          const capRes = await relaiFetch(baseUrl[1].replace(/\\u0026/g, "&"), {
            timeoutMs: 10_000,
            retries: 1,
            cache: true,
          });
          if (capRes.ok) {
            const xml = capRes.text;
            const lines = [...xml.matchAll(/<text[^>]*>([\s\S]*?)<\/text>/g)]
              .map((t) => t[1].replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#39;/g, "'"))
              .slice(0, 400)
              .join(" ");
            if (lines.length > 40) transcript = lines;
          }
        }
      }
    }

    if (!meta.title && !watch.ok) return null;

    const text = [
      meta.title ? `# ${meta.title}` : `# Video ${videoId}`,
      meta.author_name ? `Channel: ${meta.author_name} (${meta.author_url ?? ""})` : "",
      ``,
      description ? `## Description\n${description}` : "",
      ``,
      transcript ? `## Transcript (public captions)\n${transcript}` : "## Transcript\n(no accessible public captions)",
    ]
      .filter(Boolean)
      .join("\n");

    return {
      url: safeUrl,
      title: meta.title ?? `Video ${videoId}`,
      text,
      links: [],
      headings: [{ level: 1, text: meta.title ?? `Video ${videoId}` }],
      metadata: {
        channel: meta.author_name ?? "",
        channelUrl: meta.author_url ?? "",
        videoId,
        transcript: transcript ? "available" : "unavailable",
      },
      truncated: transcript.length > 0,
      fetchedAt: new Date().toISOString(),
      status: 200,
      contentType: "text/html",
      accessMode: "public",
    };
  },
};
