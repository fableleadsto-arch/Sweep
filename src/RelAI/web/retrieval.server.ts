/**
 * Semantic retrieval and ranking (server-only).
 *
 * Reference architecture: mem0 / supermemory. Keyword matching alone ranks a
 * page that repeats the query above a page that answers it. This layer keeps
 * the existing keyword index and fuses it with dense embeddings:
 *
 *   BM25 (lexical) + cosine over embeddings (semantic) → reciprocal rank fusion
 *
 * Embeddings go through Gemini's native API or OpenAI as fallback.
 * When no key is present, retrieval degrades gracefully to pure BM25 — the
 * caller's contract does not change.
 */

const GEMINI_EMBEDDING_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent";
const OPENAI_EMBEDDING_ENDPOINT = "https://api.openai.com/v1/embeddings";
const EMBED_BATCH = 90;
const MAX_EMBED_CHARS = 6000;
const EMBED_DIMENSIONS = 768;

export interface RetrievableDoc {
  id: string;
  title?: string;
  text: string;
  url?: string;
  metadata?: Record<string, unknown>;
}

export interface RankedDoc<T extends RetrievableDoc = RetrievableDoc> {
  doc: T;
  score: number;
  lexicalScore: number;
  semanticScore: number;
  rank: number;
  matchedTerms: string[];
}

export interface RankOptions {
  limit?: number;
  /** 0 = pure keyword, 1 = pure semantic. Default 0.6. */
  semanticWeight?: number;
  minScore?: number;
}

/* ------------------------------ embeddings ---------------------------- */

const embedCache = new Map<string, number[]>();

export function embeddingsAvailable(): boolean {
  return Boolean(process.env.GEMINI_API_KEY || process.env.OPENAI_API_KEY);
}

/**
 * Embed a batch of strings. Returns null per input when unavailable.
 * Uses Gemini's embedding API by default, falls back to OpenAI.
 */
export async function embedTexts(inputs: string[]): Promise<Array<number[] | null>> {
  const out: Array<number[] | null> = inputs.map(() => null);
  if (inputs.length === 0) return out;

  // Try Gemini first
  if (process.env.GEMINI_API_KEY) {
    const result = await embedViaGemini(process.env.GEMINI_API_KEY, inputs, out);
    if (result) return result;
  }

  // Fallback to OpenAI
  if (process.env.OPENAI_API_KEY) {
    const result = await embedViaOpenAI(process.env.OPENAI_API_KEY, inputs, out);
    if (result) return result;
  }

  return out; // all null = embeddings unavailable
}

async function embedViaGemini(apiKey: string, inputs: string[], out: Array<number[] | null>): Promise<Array<number[] | null> | null> {
  try {
    const results = await Promise.all(inputs.map(async (text, index) => {
      const cached = embedCache.get(text);
      if (cached) { out[index] = cached; return; }
      const res = await fetch(`${GEMINI_EMBEDDING_ENDPOINT}?key=${apiKey}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: { parts: [{ text: text.slice(0, MAX_EMBED_CHARS) }] },
          outputDimensionality: EMBED_DIMENSIONS,
        }),
      });
      if (!res.ok) return;
      const json = await res.json() as { embedding?: { values?: number[] } };
      if (json.embedding?.values) {
        embedCache.set(text, json.embedding.values);
        out[index] = json.embedding.values;
      }
    }));
    return results ? out : null;
  } catch {
    return null;
  }
}

async function embedViaOpenAI(apiKey: string, inputs: string[], out: Array<number[] | null>): Promise<Array<number[] | null> | null> {
  try {
    const pending = inputs.map((text, index) => ({ index, text: text.slice(0, MAX_EMBED_CHARS) }));
    for (let i = 0; i < pending.length; i += EMBED_BATCH) {
      const batch = pending.slice(i, i + EMBED_BATCH);
      const res = await fetch(OPENAI_EMBEDDING_ENDPOINT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: "text-embedding-3-small",
          input: batch.map(b => b.text),
          dimensions: EMBED_DIMENSIONS,
        }),
      });
      if (!res.ok) continue;
      const json = await res.json() as { data?: Array<{ index: number; embedding: number[] }> };
      if (json.data) {
        for (const item of json.data) {
          const idx = batch.find(b => b.index === item.index)?.index ?? item.index;
          embedCache.set(pending[idx]?.text ?? "", item.embedding);
          out[idx] = item.embedding;
        }
      }
    }
    return out;
  } catch {
    return null;
  }
}

export function cosine(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/* -------------------------------- BM25 -------------------------------- */

const STOPWORDS = new Set(
  "a an and are as at be but by for from has have how i if in is it its of on or that the their there these this to was what when where which who will with you your".split(
    " ",
  ),
);

export function tokenize(input: string): string[] {
  return (input.toLowerCase().match(/[a-z0-9][a-z0-9'-]{1,}/g) ?? []).filter(
    (t) => !STOPWORDS.has(t),
  );
}

interface Bm25Score {
  score: number;
  matched: string[];
}

function bm25(query: string, docs: string[]): Bm25Score[] {
  const k1 = 1.5;
  const b = 0.75;
  const qTerms = [...new Set(tokenize(query))];
  const tokenized = docs.map(tokenize);
  const avgLen =
    tokenized.reduce((sum, d) => sum + d.length, 0) / Math.max(tokenized.length, 1) || 1;

  const df = new Map<string, number>();
  for (const term of qTerms) {
    df.set(term, tokenized.filter((d) => d.includes(term)).length);
  }

  return tokenized.map((terms) => {
    const freq = new Map<string, number>();
    for (const t of terms) freq.set(t, (freq.get(t) ?? 0) + 1);
    let score = 0;
    const matched: string[] = [];
    for (const term of qTerms) {
      const f = freq.get(term) ?? 0;
      if (f === 0) continue;
      matched.push(term);
      const n = df.get(term) ?? 0;
      const idf = Math.log(1 + (tokenized.length - n + 0.5) / (n + 0.5));
      score += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + (b * terms.length) / avgLen)));
    }
    return { score, matched };
  });
}

/* ------------------------------- ranking ------------------------------ */

function rankMap(scores: number[]): number[] {
  const order = scores
    .map((s, i) => ({ s, i }))
    .sort((a, b) => b.s - a.s)
    .map((x) => x.i);
  const ranks = new Array(scores.length).fill(scores.length);
  order.forEach((docIndex, position) => {
    ranks[docIndex] = position + 1;
  });
  return ranks;
}

/**
 * Rank documents against a query with hybrid lexical + semantic scoring,
 * fused by reciprocal rank so neither signal can dominate on scale alone.
 */
export async function rankDocuments<T extends RetrievableDoc>(
  query: string,
  docs: T[],
  opts: RankOptions = {},
): Promise<RankedDoc<T>[]> {
  if (docs.length === 0) return [];
  const limit = Math.min(Math.max(opts.limit ?? 20, 1), 100);
  const weight = Math.min(Math.max(opts.semanticWeight ?? 0.6, 0), 1);

  const corpus = docs.map((d) => `${d.title ?? ""}\n${d.text}`.slice(0, MAX_EMBED_CHARS));
  const lexical = bm25(query, corpus);

  let semantic = docs.map(() => 0);
  let semanticOn = false;
  if (weight > 0 && embeddingsAvailable()) {
    const [queryVec] = await embedTexts([query]);
    if (queryVec) {
      const docVecs = await embedTexts(corpus);
      if (docVecs.some(Boolean)) {
        semanticOn = true;
        semantic = docVecs.map((v) => (v ? cosine(queryVec, v) : 0));
      }
    }
  }

  const lexRanks = rankMap(lexical.map((l) => l.score));
  const semRanks = rankMap(semantic);
  const K = 20;

  const scored = docs.map((doc, i) => {
    const lexRr = 1 / (K + lexRanks[i]);
    const semRr = 1 / (K + semRanks[i]);
    const fused = semanticOn ? (1 - weight) * lexRr + weight * semRr : lexRr;
    return {
      doc,
      score: Number((fused * (K + 1)).toFixed(4)),
      lexicalScore: Number(lexical[i].score.toFixed(4)),
      semanticScore: Number(semantic[i].toFixed(4)),
      rank: 0,
      matchedTerms: lexical[i].matched,
    };
  });

  return scored
    .filter((s) => s.score >= (opts.minScore ?? 0))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((s, i) => ({ ...s, rank: i + 1 }));
}

/** Rank plain search hits — used to reorder the engine chain's raw output. */
export async function rankHits<T extends { url: string; title: string; snippet?: string }>(
  query: string,
  hits: T[],
  opts: RankOptions = {},
): Promise<Array<T & { relevance: number }>> {
  const ranked = await rankDocuments(
    query,
    hits.map((h, i) => ({
      id: String(i),
      title: h.title,
      text: `${h.title} ${h.snippet ?? ""} ${h.url}`,
      url: h.url,
    })),
    opts,
  );
  return ranked.map((r) => ({
    ...hits[Number(r.doc.id)],
    relevance: r.score,
  }));
}
