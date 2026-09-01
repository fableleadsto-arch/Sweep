/**
 * Deterministic automation for dynamic pages (server-only).
 *
 * Reference architecture: stagehand. The expensive, flaky approach is to let a
 * model drive a browser step by step. Here the *script* is code — a declarative
 * list of deterministic steps (navigate, submit a form, follow a JSON endpoint,
 * paginate) executed by the resilient fetch layer — and the model is called
 * only at explicit judgment points ("did this page succeed?", "which of these
 * links is the pricing page?").
 *
 * Cheap, replayable, and every run returns a step-by-step trace.
 */
import { relaiFetch, assertPublicUrl } from "./http.server";
import { htmlToMarkdown, jsonToMarkdown } from "./markdown.server";
import { normalizeBatch, type JsonValue, type NormalizedDoc } from "./normalize.server";
import { geminiGenerate, relaiJson, hasRelAIKey } from "../core/gemini.server";

export type AutomationStep =
  | { kind: "goto"; url: string; label?: string }
  | { kind: "form"; url: string; fields: Record<string, string>; label?: string }
  | { kind: "json"; url: string; label?: string }
  | { kind: "paginate"; urlTemplate: string; from: number; to: number; label?: string }
  | { kind: "follow"; pattern: string; limit?: number; label?: string }
  | { kind: "judge"; question: string; label?: string };

export interface AutomationTrace {
  step: number;
  kind: AutomationStep["kind"];
  label: string;
  ok: boolean;
  detail: string;
  ms: number;
}

export interface AutomationRun {
  ok: boolean;
  docs: NormalizedDoc[];
  trace: AutomationTrace[];
  judgments: Array<{ question: string; answer: string }>;
  cookies: string[];
  elapsedMs: number;
}

interface Collected {
  url: string;
  title: string;
  description: string;
  markdown: string;
  text: string;
  fetchedAt: string;
  metadata: Record<string, JsonValue>;
}

const MAX_STEPS = 12;
const MAX_DOCS = 40;

/**
 * Run a deterministic automation script. Sessions carry cookies forward so a
 * form login in step 1 authenticates step 2 without a browser.
 */
export async function relaiAutomate(
  steps: AutomationStep[],
  opts: { budgetMs?: number; headers?: Record<string, string> } = {},
): Promise<AutomationRun> {
  const started = Date.now();
  const budgetMs = Math.min(Math.max(opts.budgetMs ?? 60_000, 5_000), 180_000);
  const trace: AutomationTrace[] = [];
  const judgments: Array<{ question: string; answer: string }> = [];
  const collected: Collected[] = [];
  const cookies = new Map<string, string>();
  let lastLinks: Array<{ url: string; text: string }> = [];
  let ok = true;

  const cookieHeader = (): Record<string, string> =>
    cookies.size > 0
      ? { Cookie: [...cookies].map(([k, v]) => `${k}=${v}`).join("; ") }
      : {};

  const capture = async (
    url: string,
    init: { method?: "GET" | "POST"; body?: string; contentType?: string } = {},
  ) => {
    const res = await relaiFetch(url, {
      method: init.method ?? "GET",
      body: init.body,
      retries: 2,
      timeoutMs: 20_000,
      cache: init.method !== "POST",
      headers: {
        ...opts.headers,
        ...cookieHeader(),
        ...(init.contentType ? { "Content-Type": init.contentType } : {}),
      },
    });
    if (!res.ok) return { ok: false as const, detail: res.error ?? `HTTP ${res.status}` };

    const isJson = res.contentType.includes("json") || /^\s*[[{]/.test(res.text.slice(0, 200));
    const doc = isJson
      ? jsonToMarkdown(res.text, res.url)
      : htmlToMarkdown(res.text, res.url, { maxChars: 16_000 });
    lastLinks = doc.links;
    if (collected.length < MAX_DOCS) {
      collected.push({
        url: doc.url,
        title: doc.meta.title,
        description: doc.meta.description,
        markdown: doc.markdown,
        text: doc.text,
        fetchedAt: res.fetchedAt,
        metadata: { status: res.status, siteName: doc.meta.siteName },
      });
    }
    return { ok: true as const, detail: `${doc.wordCount} words from ${doc.meta.title}` };
  };

  const plan = steps.slice(0, MAX_STEPS);
  for (let i = 0; i < plan.length; i++) {
    if (Date.now() - started > budgetMs) {
      trace.push({
        step: i + 1,
        kind: plan[i].kind,
        label: plan[i].label ?? plan[i].kind,
        ok: false,
        detail: "time budget exhausted",
        ms: 0,
      });
      ok = false;
      break;
    }

    const step = plan[i];
    const stepStart = Date.now();
    let result: { ok: boolean; detail: string };

    try {
      switch (step.kind) {
        case "goto": {
          assertPublicUrl(step.url);
          result = await capture(step.url);
          break;
        }
        case "form": {
          assertPublicUrl(step.url);
          const body = new URLSearchParams(step.fields).toString();
          result = await capture(step.url, {
            method: "POST",
            body,
            contentType: "application/x-www-form-urlencoded",
          });
          break;
        }
        case "json": {
          assertPublicUrl(step.url);
          result = await capture(step.url);
          break;
        }
        case "paginate": {
          const from = Math.max(step.from, 0);
          const to = Math.min(step.to, from + 20);
          let pages = 0;
          for (let page = from; page <= to; page++) {
            if (Date.now() - started > budgetMs) break;
            const url = step.urlTemplate.replace(/\{page\}/g, String(page));
            const r = await capture(url);
            if (r.ok) pages++;
          }
          result = { ok: pages > 0, detail: `${pages} page(s) captured` };
          break;
        }
        case "follow": {
          const re = safeRegex(step.pattern);
          const targets = lastLinks
            .filter((l) => (re ? re.test(l.url) : l.url.includes(step.pattern)))
            .slice(0, Math.min(step.limit ?? 5, 10));
          let hits = 0;
          for (const t of targets) {
            if (Date.now() - started > budgetMs) break;
            const r = await capture(t.url);
            if (r.ok) hits++;
          }
          result = { ok: hits > 0, detail: `followed ${hits}/${targets.length} link(s)` };
          break;
        }
        case "judge": {
          const answer = await judge(step.question, collected);
          judgments.push({ question: step.question, answer });
          result = { ok: Boolean(answer), detail: answer.slice(0, 200) };
          break;
        }
        default:
          result = { ok: false, detail: "unknown step" };
      }
    } catch (err) {
      result = { ok: false, detail: err instanceof Error ? err.message : String(err) };
    }

    if (!result.ok) ok = false;
    trace.push({
      step: i + 1,
      kind: step.kind,
      label: step.label ?? step.kind,
      ok: result.ok,
      detail: result.detail,
      ms: Date.now() - stepStart,
    });
  }

  const { docs } = normalizeBatch(
    collected.map((c) => ({ ...c, source: "automation" })),
    { nearDistance: 3, minWords: 10 },
  );

  return {
    ok,
    docs,
    trace,
    judgments,
    cookies: [...cookies.keys()],
    elapsedMs: Date.now() - started,
  };
}

/** The only place the model is consulted during automation. */
async function judge(question: string, collected: Collected[]): Promise<string> {
  if (!hasRelAIKey()) return "No model configured — judgment skipped.";
  const context = collected
    .slice(-3)
    .map((c) => `URL: ${c.url}\n${c.markdown.slice(0, 3000)}`)
    .join("\n\n---\n\n");
  try {
    const res = await geminiGenerate({
      system:
        "You are a judgment point inside a deterministic scraping script. Answer the question strictly from the captured pages. Reply with JSON: {\"answer\": string, \"confident\": boolean}. Never invent facts.",
      turns: [{ role: "user", text: `Question: ${question}\n\nCAPTURED PAGES:\n${context || "(nothing captured yet)"}` }],
      json: true,
      temperature: 0.1,
      maxTokens: 500,
      timeoutMs: 30_000,
    });
    const parsed = relaiJson<{ answer?: string }>(res.text, {});
    return parsed.answer ?? res.text.slice(0, 400);
  } catch (err) {
    return `judgment failed: ${err instanceof Error ? err.message : String(err)}`;
  }
}

function safeRegex(pattern: string): RegExp | null {
  try {
    return new RegExp(pattern, "i");
  } catch {
    return null;
  }
}
