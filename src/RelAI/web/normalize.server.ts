/**
 * The single normalization step every scraped record passes through before it
 * reaches the index. Nothing else in RelAI is allowed to write documents.
 *
 * Responsibilities: canonicalize the URL, clean whitespace and scraping
 * artifacts, drop boilerplate lines, fingerprint the content, dedupe both
 * exactly and near-exactly, and stamp every record with a fetch time.
 */

export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

export interface NormalizedDoc {
  id: string;
  url: string;
  canonicalUrl: string;
  host: string;
  title: string;
  description: string;
  markdown: string;
  text: string;
  source: string;
  fingerprint: string;
  wordCount: number;
  fetchedAt: string;
  normalizedAt: string;
  metadata: Record<string, JsonValue>;
}

export interface RawDoc {
  url: string;
  title?: string;
  description?: string;
  markdown?: string;
  text?: string;
  source?: string;
  fetchedAt?: string;
  metadata?: Record<string, JsonValue>;
}

const TRACKING_PARAMS = /^(utm_|fbclid|gclid|mc_|ref|ref_src|igshid|si|spm|_hs|yclid|msclkid)/i;

/** Strip tracking noise and trailing slashes so two URLs to one page collapse. */
export function canonicalizeUrl(raw: string): string {
  try {
    const u = new URL(raw);
    u.hash = "";
    u.hostname = u.hostname.toLowerCase().replace(/^www\./, "");
    const keep = [...u.searchParams.entries()].filter(([k]) => !TRACKING_PARAMS.test(k));
    u.search = "";
    for (const [k, v] of keep.sort(([a], [b]) => a.localeCompare(b))) {
      u.searchParams.append(k, v);
    }
    if (u.pathname.length > 1 && u.pathname.endsWith("/")) {
      u.pathname = u.pathname.replace(/\/+$/, "");
    }
    if ((u.protocol === "https:" && u.port === "443") || (u.protocol === "http:" && u.port === "80")) {
      u.port = "";
    }
    return u.toString();
  } catch {
    return raw.trim();
  }
}

const BOILERPLATE = [
  /^(accept|manage) (all )?cookies?/i,
  /^(sign|log) ?in\b/i,
  /^subscribe( to)?\b/i,
  /^share (this|on)\b/i,
  /^advertisement$/i,
  /^skip to (main )?content/i,
  /^copyright ?©/i,
  /^all rights reserved/i,
  /^(privacy|cookie) policy$/i,
];

/** Whitespace, control chars, repeated punctuation and boilerplate lines. */
export function cleanContent(input: string): string {
  const lines = input
    // eslint-disable-next-line no-control-regex
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, " ")
    .replace(/\u00a0|\u200b|\ufeff/g, " ")
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .map((l) => l.replace(/[ \t]+/g, " ").trim());

  const kept: string[] = [];
  let lastLine = "";
  for (const line of lines) {
    if (BOILERPLATE.some((re) => re.test(line))) continue;
    // Collapse the "same line repeated by a broken template" artifact.
    if (line && line === lastLine) continue;
    kept.push(line);
    lastLine = line;
  }

  return kept
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/([!?.,])\1{3,}/g, "$1$1$1")
    .trim();
}

/** Stable 64-bit-ish hash, hex encoded. Used for exact-duplicate detection. */
export function hashString(input: string): string {
  let h1 = 0x811c9dc5;
  let h2 = 0x1000193;
  for (let i = 0; i < input.length; i++) {
    const c = input.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 16777619) >>> 0;
    h2 = Math.imul(h2 + c, 2654435761) >>> 0;
  }
  return h1.toString(16).padStart(8, "0") + h2.toString(16).padStart(8, "0");
}

/** SimHash over word shingles — near-duplicate detection across mirrors. */
export function simhash(text: string): bigint {
  const tokens = text.toLowerCase().match(/[a-z0-9]{3,}/g) ?? [];
  if (tokens.length === 0) return 0n;
  const shingles = new Map<string, number>();
  for (let i = 0; i < tokens.length; i++) {
    const key = `${tokens[i]} ${tokens[i + 1] ?? ""}`;
    shingles.set(key, (shingles.get(key) ?? 0) + 1);
  }
  const bits = new Array<number>(64).fill(0);
  for (const [gram, weight] of shingles) {
    const h = BigInt(`0x${hashString(gram)}`);
    for (let b = 0; b < 64; b++) {
      bits[b] += ((h >> BigInt(b)) & 1n) === 1n ? weight : -weight;
    }
  }
  let out = 0n;
  for (let b = 0; b < 64; b++) if (bits[b] > 0) out |= 1n << BigInt(b);
  return out;
}

export function hammingDistance(a: bigint, b: bigint): number {
  let x = a ^ b;
  let count = 0;
  while (x) {
    x &= x - 1n;
    count++;
  }
  return count;
}

/** Normalize one raw scrape into an indexable record. */
export function normalizeDoc(raw: RawDoc): NormalizedDoc {
  const canonicalUrl = canonicalizeUrl(raw.url);
  const markdown = cleanContent(raw.markdown ?? raw.text ?? "");
  const text = cleanContent(raw.text ?? markdown.replace(/[#>*_`[\]()]/g, " "));
  let host = "";
  try {
    host = new URL(canonicalUrl).hostname;
  } catch {
    /* keep it empty for non-URL sources */
  }
  const now = new Date().toISOString();

  return {
    id: hashString(canonicalUrl),
    url: raw.url,
    canonicalUrl,
    host,
    title: cleanContent(raw.title ?? "").slice(0, 300) || host || canonicalUrl,
    description: cleanContent(raw.description ?? "").slice(0, 600),
    markdown,
    text,
    source: raw.source ?? "web",
    fingerprint: hashString(text.slice(0, 4000)),
    wordCount: text.split(/\s+/).filter(Boolean).length,
    fetchedAt: raw.fetchedAt ?? now,
    normalizedAt: now,
    metadata: raw.metadata ?? {},
  };
}

/**
 * Normalize a batch and remove duplicates: same canonical URL, same content
 * fingerprint, or a SimHash within `nearDistance` bits of a kept document.
 */
export function normalizeBatch(
  raws: RawDoc[],
  opts: { nearDistance?: number; minWords?: number } = {},
): { docs: NormalizedDoc[]; dropped: number } {
  const nearDistance = opts.nearDistance ?? 3;
  const minWords = opts.minWords ?? 0;
  const byUrl = new Set<string>();
  const byFingerprint = new Set<string>();
  const hashes: bigint[] = [];
  const docs: NormalizedDoc[] = [];
  let dropped = 0;

  for (const raw of raws) {
    const doc = normalizeDoc(raw);
    if (doc.wordCount < minWords) {
      dropped++;
      continue;
    }
    if (byUrl.has(doc.canonicalUrl) || byFingerprint.has(doc.fingerprint)) {
      dropped++;
      continue;
    }
    if (doc.wordCount > 40) {
      const sh = simhash(doc.text);
      if (hashes.some((h) => hammingDistance(h, sh) <= nearDistance)) {
        dropped++;
        continue;
      }
      hashes.push(sh);
    }
    byUrl.add(doc.canonicalUrl);
    byFingerprint.add(doc.fingerprint);
    docs.push(doc);
  }

  return { docs, dropped };
}
