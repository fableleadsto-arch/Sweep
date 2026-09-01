/**
 * Relay Surf — evidence store (server-only).
 *
 * Every web-derived claim flows through here as `Evidence`, so Relay Brain can
 * reason over cited evidence instead of raw search results. Handles:
 *   - deduplication (canonical URL + content hash)
 *   - provenance tracking (retrieval time, access mode, platform)
 *   - containment: session-scoped, bounded.
 */
import type { Evidence, Source } from "../types";

export class EvidenceStore {
  private evidence: Evidence[] = [];
  private sources = new Map<string, Source>();
  private urlSeen = new Set<string>();

  constructor(private maxEvidence = 200) {}

  /** Record a piece of evidence (deduped by source URL + claim). */
  add(input: {
    sourceUrl: string;
    sourceTitle: string;
    claim: string;
    excerpt: string;
    platform?: Evidence["platform"];
    timestamp?: string;
    accessMode?: Evidence["accessMode"];
    confidence?: number;
  }): Evidence | null {
    const key = `${input.sourceUrl.split("#")[0]}|${normalizeClaim(input.claim)}`;
    if (this.urlSeen.has(key)) return null;
    this.urlSeen.add(key);

    if (this.evidence.length >= this.maxEvidence) {
      this.evidence.shift();
    }

    const item: Evidence = {
      id: `ev_${this.evidence.length + 1}_${hashKey(input.sourceUrl).slice(0, 6)}`,
      sourceUrl: input.sourceUrl,
      sourceTitle: input.sourceTitle,
      platform: input.platform,
      excerpt: input.excerpt.slice(0, 800),
      claim: input.claim,
      timestamp: input.timestamp,
      accessMode: input.accessMode ?? "public",
      confidence: Math.min(1, Math.max(0, input.confidence ?? 0.6)),
    };
    this.evidence.push(item);

    // Track the underlying source too.
    if (!this.sources.has(input.sourceUrl)) {
      this.sources.set(input.sourceUrl, {
        title: input.sourceTitle,
        url: input.sourceUrl,
        platform: input.platform,
        accessMode: input.accessMode ?? "public",
        retrievedAt: new Date().toISOString(),
      });
    }
    return item;
  }

  /** Register a source even without evidence (e.g. a browsed page). */
  trackSource(source: Source): void {
    if (!this.sources.has(source.url)) this.sources.set(source.url, source);
  }

  all(): Evidence[] {
    return [...this.evidence];
  }

  listSources(): Source[] {
    return [...this.sources.values()];
  }

  count(): number {
    return this.evidence.length;
  }

  clear(): void {
    this.evidence = [];
    this.sources.clear();
    this.urlSeen.clear();
  }
}

/** Claims are normalized before dedup so "Pricing: $20" vs "$20 pricing" differ. */
function normalizeClaim(claim: string): string {
  return claim.toLowerCase().replace(/[^\w\s]/g, "").trim().slice(0, 120);
}

function hashKey(input: string): string {
  let hash = 5381;
  for (let i = 0; i < input.length; i++) {
    hash = (hash * 33) ^ input.charCodeAt(i);
  }
  return (hash >>> 0).toString(36);
}
