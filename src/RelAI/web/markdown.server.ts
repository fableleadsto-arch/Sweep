/**
 * HTML → clean Markdown (server-only).
 *
 * Reference architecture: crawl4ai. Instead of handing the model raw HTML we
 * pick the main content region, drop chrome (nav/aside/footer/script), and
 * emit normalized Markdown so every indexed document looks the same shape
 * regardless of which site it came from.
 */

export interface PageMeta {
  title: string;
  description: string;
  siteName: string;
  author: string;
  publishedAt: string | null;
  lang: string;
  canonical: string | null;
  jsonLd: unknown[];
}

export interface MarkdownDoc {
  url: string;
  markdown: string;
  text: string;
  meta: PageMeta;
  links: Array<{ url: string; text: string }>;
  wordCount: number;
  truncated: boolean;
}


function decodeEntities(input: string): string {
  return input
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#0?39;|&apos;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&mdash;/gi, "—")
    .replace(/&ndash;/gi, "–")
    .replace(/&hellip;/gi, "…")
    .replace(/&#(\d+);/g, (_, d) => safeCodePoint(Number(d)))
    .replace(/&#x([0-9a-f]+);/gi, (_, h) => safeCodePoint(parseInt(h, 16)));
}

function safeCodePoint(code: number): string {
  if (!Number.isFinite(code) || code < 9 || code > 0x10ffff) return " ";
  try {
    return String.fromCodePoint(code);
  } catch {
    return " ";
  }
}

/** Strip every tag and collapse whitespace — the plain-text baseline. */
export function htmlToText(html: string): string {
  return decodeEntities(
    stripNoise(html)
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " "),
  ).trim();
}

function stripNoise(html: string): string {
  return html
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<noscript[\s\S]*?<\/noscript>/gi, " ")
    .replace(/<svg[\s\S]*?<\/svg>/gi, " ")
    .replace(/<iframe[\s\S]*?<\/iframe>/gi, " ")
    .replace(/<form[\s\S]*?<\/form>/gi, " ");
}

/** Pick the densest plausible content region — <article>, <main>, else <body>. */
function mainRegion(html: string): string {
  const candidates: string[] = [];
  for (const re of [
    /<article[^>]*>([\s\S]*?)<\/article>/gi,
    /<main[^>]*>([\s\S]*?)<\/main>/gi,
    /<div[^>]+(?:id|class)="[^"]*(?:post|content|entry|story|markdown|body)[^"]*"[^>]*>([\s\S]*?)<\/div>/gi,
  ]) {
    let m: RegExpExecArray | null;
    while ((m = re.exec(html)) !== null) candidates.push(m[1]);
  }
  const body = /<body[^>]*>([\s\S]*?)<\/body>/i.exec(html)?.[1] ?? html;
  candidates.push(body);

  let best = body;
  let bestScore = 0;
  for (const c of candidates) {
    const score = htmlToText(c).length;
    if (score > bestScore) {
      bestScore = score;
      best = c;
    }
  }
  // A region that lost more than 85% of the body is likely a wrapper miss.
  return bestScore >= htmlToText(body).length * 0.15 ? best : body;
}

function dropChrome(html: string): string {
  return html
    .replace(/<nav[\s\S]*?<\/nav>/gi, " ")
    .replace(/<aside[\s\S]*?<\/aside>/gi, " ")
    .replace(/<footer[\s\S]*?<\/footer>/gi, " ")
    .replace(/<header[^>]*class="[^"]*(?:site|global|top)[^"]*"[\s\S]*?<\/header>/gi, " ");
}

function attr(tag: string, name: string): string {
  const m = new RegExp(`${name}\\s*=\\s*"([^"]*)"|${name}\\s*=\\s*'([^']*)'`, "i").exec(tag);
  return decodeEntities(m?.[1] ?? m?.[2] ?? "").trim();
}

function metaContent(html: string, key: string): string {
  const re = new RegExp(
    `<meta[^>]+(?:name|property)\\s*=\\s*["']${key}["'][^>]*>`,
    "i",
  );
  const tag = re.exec(html)?.[0];
  return tag ? attr(tag, "content") : "";
}

export function readMeta(html: string, url: string): PageMeta {
  const jsonLd: unknown[] = [];
  const ldRe = /<script[^>]+type\s*=\s*["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi;
  let m: RegExpExecArray | null;
  while ((m = ldRe.exec(html)) !== null) {
    try {
      const parsed = JSON.parse(m[1].trim());
      if (Array.isArray(parsed)) jsonLd.push(...parsed);
      else jsonLd.push(parsed);
    } catch {
      /* a malformed ld+json block is not a failure */
    }
  }

  const titleTag = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html)?.[1];
  const canonicalTag = /<link[^>]+rel\s*=\s*["']canonical["'][^>]*>/i.exec(html)?.[0];
  let host = url;
  try {
    host = new URL(url).hostname.replace(/^www\./, "");
  } catch {
    /* keep the raw url */
  }

  return {
    title:
      metaContent(html, "og:title") ||
      (titleTag ? decodeEntities(titleTag).replace(/\s+/g, " ").trim() : "") ||
      host,
    description:
      metaContent(html, "og:description") || metaContent(html, "description"),
    siteName: metaContent(html, "og:site_name") || host,
    author: metaContent(html, "author") || metaContent(html, "article:author"),
    publishedAt:
      metaContent(html, "article:published_time") ||
      metaContent(html, "datePublished") ||
      null,
    lang: /<html[^>]+lang\s*=\s*["']([^"']+)["']/i.exec(html)?.[1] ?? "",
    canonical: canonicalTag ? attr(canonicalTag, "href") || null : null,
    jsonLd,
  };
}

function collectLinks(html: string, base: string): Array<{ url: string; text: string }> {
  const out: Array<{ url: string; text: string }> = [];
  const seen = new Set<string>();
  const re = /<a[^>]+href\s*=\s*["']([^"'#][^"']*)["'][^>]*>([\s\S]*?)<\/a>/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(html)) !== null) {
    let abs: string;
    try {
      abs = new URL(decodeEntities(m[1]), base).toString();
    } catch {
      continue;
    }
    if (!/^https?:/.test(abs)) continue;
    const key = abs.split("#")[0];
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ url: key, text: htmlToText(m[2]).slice(0, 160) });
    if (out.length >= 300) break;
  }
  return out;
}

/** Convert a content region into Markdown with headings, lists and links. */
function toMarkdown(region: string, base: string): string {
  let s = stripNoise(region);

  s = s
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<hr\s*\/?>/gi, "\n\n---\n\n")
    .replace(/<\/(p|div|section|article|li|tr|blockquote)>/gi, "\n\n")
    .replace(/<li[^>]*>/gi, "\n- ")
    .replace(/<\/?(strong|b)>/gi, "**")
    .replace(/<\/?(em|i)>/gi, "_")
    .replace(/<code[^>]*>([\s\S]*?)<\/code>/gi, (_, c) => `\`${htmlToText(c)}\``)
    .replace(/<pre[^>]*>([\s\S]*?)<\/pre>/gi, (_, c) => `\n\n\`\`\`\n${htmlToText(c)}\n\`\`\`\n\n`)
    .replace(/<blockquote[^>]*>/gi, "\n> ");

  for (let level = 1; level <= 6; level++) {
    s = s.replace(
      new RegExp(`<h${level}[^>]*>([\\s\\S]*?)<\\/h${level}>`, "gi"),
      (_, inner) => `\n\n${"#".repeat(level)} ${htmlToText(inner)}\n\n`,
    );
  }

  s = s.replace(
    /<a[^>]+href\s*=\s*["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi,
    (_, href: string, inner: string) => {
      const label = htmlToText(inner);
      if (!label) return " ";
      let abs = href;
      try {
        abs = new URL(decodeEntities(href), base).toString();
      } catch {
        /* leave the raw href */
      }
      return `[${label}](${abs})`;
    },
  );

  s = s.replace(/<img[^>]*>/gi, (tag) => {
    const alt = attr(tag, "alt");
    return alt ? `![${alt}]` : " ";
  });

  s = decodeEntities(s.replace(/<[^>]+>/g, " "));

  return s
    .replace(/[ \t\u00a0]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/(^|\n)- (?=\n)/g, "$1")
    .trim();
}

/**
 * The single HTML → LLM-ready conversion used by every RelAI reader.
 */
export function htmlToMarkdown(
  html: string,
  url: string,
  opts: { maxChars?: number; keepChrome?: boolean } = {},
): MarkdownDoc {
  const max = Math.min(Math.max(opts.maxChars ?? 12_000, 500), 60_000);
  const meta = readMeta(html, url);
  const region = opts.keepChrome ? mainRegion(html) : dropChrome(mainRegion(html));
  const full = toMarkdown(region, url);
  const markdown = full.slice(0, max);
  const text = htmlToText(region).slice(0, max);

  return {
    url,
    markdown,
    text,
    meta,
    links: collectLinks(region, url),
    wordCount: full.split(/\s+/).filter(Boolean).length,
    truncated: full.length > max,
  };
}

/** JSON responses get a Markdown-ish rendering so the pipeline stays uniform. */
export function jsonToMarkdown(raw: string, url: string): MarkdownDoc {
  let pretty = raw;
  try {
    pretty = JSON.stringify(JSON.parse(raw), null, 2);
  } catch {
    /* keep the raw body */
  }
  const md = `\`\`\`json\n${pretty.slice(0, 12_000)}\n\`\`\``;
  let host = url;
  try {
    host = new URL(url).hostname;
  } catch {
    /* keep the raw url */
  }
  return {
    url,
    markdown: md,
    text: pretty.slice(0, 12_000),
    meta: {
      title: host,
      description: "",
      siteName: host,
      author: "",
      publishedAt: null,
      lang: "",
      canonical: null,
      jsonLd: [],
    },
    links: [],
    wordCount: pretty.split(/\s+/).length,
    truncated: pretty.length > 12_000,
  };
}
