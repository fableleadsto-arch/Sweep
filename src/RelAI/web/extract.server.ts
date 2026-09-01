/**
 * Resilient structured extraction (server-only).
 *
 * Reference architecture: ScrapeGraphAI (graph-based: fetch → parse → reduce →
 * generate) with a Skyvern-style layout-resilient fallback. The point is that
 * a site redesign must not break extraction:
 *
 *   1. deterministic layer — JSON-LD, microdata, OpenGraph, meta tags
 *   2. heuristic layer     — regexes for the universal fields (emails, phones,
 *                            socials, prices) that survive any layout change
 *   3. semantic layer      — the model reads the normalized Markdown and fills
 *                            the requested schema, citing nothing it can't see
 *
 * Each layer records what it contributed, so an operator can see whether a
 * field came from the page's own structured data or from the model.
 */
import { geminiGenerate, relaiJson, hasRelAIKey } from "../core/gemini.server";
import { fetchAsMarkdown } from "./crawl.server";
import type { MarkdownDoc } from "./markdown.server";
import { cleanContent, type JsonValue } from "./normalize.server";

export type ExtractionLayer = "structured" | "heuristic" | "semantic";

export interface ExtractionResult {
  url: string;
  data: Record<string, JsonValue>;
  fieldSources: Record<string, ExtractionLayer>;
  layersUsed: ExtractionLayer[];
  confidence: number;
  missing: string[];
  notes: string[];
  fetchedAt: string;
}

const EMAIL_RE = /[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}/gi;
const PHONE_RE = /\+?\d[\d\s().-]{7,17}\d/g;
const SOCIAL_RE =
  /https?:\/\/(?:www\.)?(?:linkedin\.com|x\.com|twitter\.com|instagram\.com|facebook\.com|github\.com|tiktok\.com|youtube\.com)\/[^\s"'<>)]+/gi;

/* ---------------------- layer 1: structured data ---------------------- */

function flattenJsonLd(nodes: unknown[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const walk = (node: unknown, depth: number) => {
    if (depth > 3 || !node || typeof node !== "object") return;
    if (Array.isArray(node)) {
      for (const n of node) walk(n, depth + 1);
      return;
    }
    for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
      if (k.startsWith("@") && k !== "@type") continue;
      if (v == null) continue;
      if (typeof v === "object") {
        walk(v, depth + 1);
        continue;
      }
      const key = k === "@type" ? "type" : k;
      if (!(key in out)) out[key] = v;
    }
  };
  walk(nodes, 0);
  return out;
}

function structuredLayer(doc: MarkdownDoc): Record<string, JsonValue> {
  const ld = flattenJsonLd(doc.meta.jsonLd);
  const out: Record<string, JsonValue> = { ...(ld as Record<string, JsonValue>) };
  if (doc.meta.title) out.title ??= doc.meta.title;
  if (doc.meta.description) out.description ??= doc.meta.description;
  if (doc.meta.siteName) out.siteName ??= doc.meta.siteName;
  if (doc.meta.author) out.author ??= doc.meta.author;
  if (doc.meta.publishedAt) out.publishedAt ??= doc.meta.publishedAt;
  return out;
}

/* ------------------------ layer 2: heuristics ------------------------- */

function heuristicLayer(doc: MarkdownDoc): Record<string, JsonValue> {
  const body = `${doc.text}\n${doc.links.map((l) => l.url).join("\n")}`;
  const emails = uniq(
    (body.match(EMAIL_RE) ?? [])
      .map((e) => e.toLowerCase())
      .filter((e) => !/\.(png|jpe?g|gif|webp|svg)$/i.test(e)),
  ).slice(0, 10);
  const phones = uniq(
    (doc.text.match(PHONE_RE) ?? []).map((p) => p.replace(/\s+/g, " ").trim()),
  )
    .filter((p) => p.replace(/\D/g, "").length >= 9)
    .slice(0, 6);
  const socials = uniq(body.match(SOCIAL_RE) ?? []).slice(0, 12);
  const headings = (doc.markdown.match(/^#{1,3} .+$/gm) ?? [])
    .map((h) => h.replace(/^#+ /, "").trim())
    .slice(0, 15);

  const out: Record<string, JsonValue> = {};
  if (emails.length) out.emails = emails;
  if (phones.length) out.phones = phones;
  if (socials.length) out.socialProfiles = socials;
  if (headings.length) out.headings = headings;
  return out;
}

/* ------------------------- layer 3: semantic -------------------------- */

const SEMANTIC_SYSTEM = `You read one web page that has already been converted to clean Markdown and fill a requested JSON schema.

Rules:
- Use only what is present in the page. Never invent a name, number, email or URL.
- If a field is not present, set it to null. Do not guess.
- Values must be copied or minimally summarised from the page text.
- Return ONLY the JSON object with the requested keys, no commentary.`;

async function semanticLayer(
  doc: MarkdownDoc,
  fields: string[],
  instruction?: string,
): Promise<Record<string, JsonValue>> {
  if (!hasRelAIKey() || fields.length === 0) return {};
  const prompt = [
    `URL: ${doc.url}`,
    instruction ? `Goal: ${instruction}` : "",
    `Fields to fill: ${JSON.stringify(fields)}`,
    "",
    "PAGE (markdown):",
    doc.markdown.slice(0, 14_000),
  ]
    .filter(Boolean)
    .join("\n");

  try {
    const res = await geminiGenerate({
      system: SEMANTIC_SYSTEM,
      turns: [{ role: "user", text: prompt }],
      json: true,
      temperature: 0.1,
      maxTokens: 1400,
      timeoutMs: 45_000,
    });
    const parsed = relaiJson<Record<string, JsonValue>>(res.text, {});
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

/* ------------------------------ driver -------------------------------- */

/**
 * Extract structured data from an already-fetched Markdown document.
 * Layers run cheapest-first; the model is only asked for what is still empty.
 */
export async function extractFromDoc(
  doc: MarkdownDoc,
  opts: { fields?: string[]; instruction?: string; fetchedAt?: string } = {},
): Promise<ExtractionResult> {
  const fields = (opts.fields ?? []).map((f) => f.trim()).filter(Boolean).slice(0, 25);
  const data: Record<string, JsonValue> = {};
  const fieldSources: Record<string, ExtractionLayer> = {};
  const layersUsed: ExtractionLayer[] = [];
  const notes: string[] = [];

  const structured = structuredLayer(doc);
  if (Object.keys(structured).length > 0) layersUsed.push("structured");
  for (const [k, v] of Object.entries(structured)) {
    if (fields.length && !fields.includes(k)) continue;
    if (isEmpty(v)) continue;
    data[k] = v;
    fieldSources[k] = "structured";
  }

  const heuristic = heuristicLayer(doc);
  if (Object.keys(heuristic).length > 0) layersUsed.push("heuristic");
  for (const [k, v] of Object.entries(heuristic)) {
    if (fields.length && !fields.includes(k)) continue;
    if (k in data || isEmpty(v)) continue;
    data[k] = v;
    fieldSources[k] = "heuristic";
  }

  const stillMissing = fields.filter((f) => isEmpty(data[f]));
  const needsSemantic = fields.length === 0 ? false : stillMissing.length > 0;
  if (needsSemantic) {
    const semantic = await semanticLayer(doc, stillMissing, opts.instruction);
    if (Object.keys(semantic).length > 0) {
      layersUsed.push("semantic");
      for (const [k, v] of Object.entries(semantic)) {
        if (isEmpty(v)) continue;
        data[k] = typeof v === "string" ? cleanContent(v).slice(0, 2000) : v;
        fieldSources[k] = "semantic";
      }
    } else {
      notes.push("Semantic fallback returned nothing usable for the empty fields.");
    }
  }

  const missing = fields.filter((f) => isEmpty(data[f]));
  const filled = fields.length ? fields.length - missing.length : Object.keys(data).length;
  const denominator = fields.length || Math.max(Object.keys(data).length, 1);
  const structuredShare =
    Object.values(fieldSources).filter((s) => s !== "semantic").length /
    Math.max(Object.keys(fieldSources).length, 1);
  const confidence = Number(
    Math.min(1, (filled / denominator) * (0.7 + 0.3 * structuredShare)).toFixed(2),
  );

  if (doc.wordCount < 60) {
    notes.push("Page carried very little text — it may be JS-rendered or gated.");
  }

  return {
    url: doc.url,
    data,
    fieldSources,
    layersUsed,
    confidence,
    missing,
    notes,
    fetchedAt: opts.fetchedAt ?? new Date().toISOString(),
  };
}

/** Fetch a URL and extract in one call. */
export async function relaiExtract(
  url: string,
  opts: { fields?: string[]; instruction?: string } = {},
): Promise<ExtractionResult> {
  const out = await fetchAsMarkdown(url, { maxChars: 20_000 });
  if ("error" in out) {
    return {
      url,
      data: {},
      fieldSources: {},
      layersUsed: [],
      confidence: 0,
      missing: opts.fields ?? [],
      notes: [
        out.blocked
          ? `${url} challenged or blocked the request; retries and proxy fallback were exhausted.`
          : `Could not read ${url}: ${out.error}`,
      ],
      fetchedAt: new Date().toISOString(),
    };
  }
  return extractFromDoc(out.doc, { ...opts, fetchedAt: out.fetchedAt });
}

function isEmpty(v: unknown): boolean {
  if (v == null) return true;
  if (typeof v === "string") return v.trim() === "" || v.trim().toLowerCase() === "null";
  if (Array.isArray(v)) return v.length === 0;
  return false;
}

function uniq(list: string[]): string[] {
  return [...new Set(list.map((s) => s.trim()).filter(Boolean))];
}
