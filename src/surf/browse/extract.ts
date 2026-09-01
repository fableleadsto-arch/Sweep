/**
 * Relay Surf — page extraction (server-only).
 *
 * Turns raw fetched content (HTML/JSON) into the rich `PageData` shape used
 * across the subsystem: clean text, semantic link hints, headings, metadata,
 * structured data and an injection assessment. Never dumps raw HTML to the
 * model — everything flows through the Markdown layer first.
 */
import { htmlToMarkdown, jsonToMarkdown, type MarkdownDoc } from "@/RelAI/web/markdown.server";
import { assessInjection } from "../guard";
import type { HeadingData, LinkData, PageData } from "../types";

export interface ExtractInput {
  html?: string;
  json?: string;
  url: string;
  status?: number;
  contentType?: string;
  maxChars?: number;
  fetchedAt?: string;
}

/** Known anchor-text/href signals → link intent (for link following). */
const LINK_INTENTS: Array<{ re: RegExp; intent: string }> = [
  { re: /\bpricing\b/i, intent: "pricing" },
  { re: /\bdocs?\b|documentation|api\s*reference|developers?\b/i, intent: "documentation" },
  { re: /\bfaq\b|frequently\s+asked/i, intent: "faq" },
  { re: /\babout\b|our\s+story|who\s+we\s+are/i, intent: "about" },
  { re: /\bcontact\b|reach\s+us|support\b|help\b/i, intent: "contact" },
  { re: /\bgithub\.com\b|source\s+code|open\s+source/i, intent: "github" },
  { re: /\bblog\b|news\b|articles?\b/i, intent: "blog" },
  { re: /\bchangelog\b|release\s+notes\b|what'?s\s+new\b/i, intent: "changelog" },
  { re: /\blogin\b|sign\s*in\b/i, intent: "login" },
  { re: /\bsign\s*up\b|register|get\s+started|try\s+free/i, intent: "signup" },
  { re: /\bfeatures?\b|capabilities\b/i, intent: "features" },
  { re: /\bterms\b|privacy\b/i, intent: "legal" },
];

function linkIntent(text: string, url: string): string | undefined {
  const haystack = `${text} ${url}`.slice(0, 120);
  for (const { re, intent } of LINK_INTENTS) {
    if (re.test(haystack)) return intent;
  }
  return undefined;
}

/** Extract heading structure from Markdown. */
function headingsFromMarkdown(markdown: string): HeadingData[] {
  const out: HeadingData[] = [];
  const re = /^(#{1,6})\s+(.*)$/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(markdown)) !== null) {
    const level = m[1].length;
    const text = m[2].replace(/[`*_()[\]]/g, "").trim();
    if (!text) continue;
    const id = text
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, "")
      .trim()
      .replace(/\s+/g, "-");
    out.push({ level, text, id });
    if (out.length >= 80) break;
  }
  return out;
}

/** Split cleaned text into ~1800-char chunks at sentence boundaries. */
export function chunkText(text: string, maxChars = 1800): string[] {
  const chunks: string[] = [];
  let rest = text.trim();
  while (rest.length > 0) {
    const slice = rest.slice(0, maxChars);
    const boundary = findBoundary(slice);
    chunks.push(boundary.text);
    rest = rest.slice(boundary.consumed);
    if (chunks.length >= 40) break;
  }
  return chunks;
}

function findBoundary(slice: string): { text: string; consumed: number } {
  if (slice.length < 400) return { text: slice.trim(), consumed: slice.length };
  const candidates = [slice.lastIndexOf("\n\n"), slice.lastIndexOf("."), slice.lastIndexOf("?")];
  const cutoff = slice.length >= 1200 ? 1200 : 600;
  const boundary = Math.max(...candidates.filter((c) => c > cutoff));
  if (boundary < cutoff) return { text: slice.trim(), consumed: slice.length };
  return { text: slice.slice(0, boundary).trim(), consumed: boundary };
}

/** Extract the chunk(s) of text around a keyword match (find-on-page core). */
export function extractAroundMatch(
  text: string,
  query: string,
  windowChars = 1600,
): { heading?: string; section: string; start: number; end: number }[] {
  const needle = query.toLowerCase();
  const sections: Array<{ heading?: string; section: string; start: number; end: number }> = [];
  const lines = text.split(/\n+/);
  let currentHeading: string | undefined;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^#{1,6}\s/.test(line)) currentHeading = line.replace(/^#{1,6}\s+/, "").trim();
    if (!line.toLowerCase().includes(needle)) continue;

    let start = 0;
    let end = text.length;
    let consumed = 0;
    const target = Math.max(0, i - Math.floor(windowChars / 160));
    for (let j = 0; j < target; j++) consumed += lines[j].length + 1;
    start = consumed;
    let endConsumed = consumed;
    for (let j = target; j < Math.min(lines.length, target + Math.floor(windowChars / 120)); j++) {
      endConsumed += lines[j].length + 1;
    }
    end = endConsumed;
    const section = text.slice(start, end).trim();
    if (section) {
      sections.push({ heading: currentHeading, section, start, end });
      if (sections.length >= 5) break;
    }
  }
  return sections;
}

/**
 * Convert raw content into PageData. Content must be either HTML or JSON.
 */
export function extractPageData(input: ExtractInput): PageData {
  const maxChars = Math.min(Math.max(input.maxChars ?? 12_000, 500), 60_000);
  let doc: MarkdownDoc;
  const isJson = Boolean(input.json) || /^\s*[[{]/.test((input.html ?? "").slice(0, 200));
  if (input.json || isJson) {
    doc = jsonToMarkdown(input.json ?? input.html ?? "", input.url);
  } else {
    doc = htmlToMarkdown(input.html ?? "", input.url, { maxChars });
  }

  const links: LinkData[] = doc.links.map(({ url, text }) => {
    const intent = linkIntent(text, url);
    let external = false;
    try {
      external = new URL(url).hostname !== new URL(input.url).hostname;
    } catch {
      /* keep default */
    }
    return { url, text, intent, external };
  });

  const headings = headingsFromMarkdown(doc.markdown);
  const injection = assessInjection(doc.text);

  const metadata: Record<string, string> = {};
  const { meta } = doc;
  if (meta.description) metadata.description = meta.description;
  if (meta.siteName) metadata.siteName = meta.siteName;
  if (meta.author) metadata.author = meta.author;
  if (meta.publishedAt) metadata.publishedAt = meta.publishedAt;
  if (meta.canonical) metadata.canonical = meta.canonical;
  if (meta.lang) metadata.lang = meta.lang;

  return {
    url: input.url,
    title: meta.title,
    description: meta.description || undefined,
    text: doc.text,
    markdown: doc.markdown,
    links,
    headings,
    metadata,
    structuredData: meta.jsonLd.length ? meta.jsonLd : undefined,
    truncated: doc.truncated,
    fetchedAt: input.fetchedAt ?? new Date().toISOString(),
    status: input.status ?? 200,
    contentType: input.contentType ?? "text/html",
    accessMode: "public",
    injection,
  };
}

/** How many tokens a chunk budget allows given a max-token ceiling. */
export function chunkBudget(maxTokens: number): number {
  return Math.max(1, Math.floor(maxTokens / 8)); // ~8 tokens per chunk on average
}
