/**
 * Relay Surf — research engine (server-only).
 *
 * Runs the multi-step research loop for a session:
 *
 *   PLAN → SEARCH → OPEN → EXTRACT → ENOUGH? → synthesize
 *
 * with hard, configurable budgets (searches, pages, runtime, depth) checked at
 * every step, and early stoppage once enough independent evidence exists. Every
 * step is recorded as a human-readable `SurfAction` so the UI renders a live
 * activity stream — never raw logs. Pages that trip the prompt-injection scan
 * are quarantined and never enter the evidence pool.
 */
import type { SearchResult, Source, SurfPlan, SurfSession } from "../types";
import { resolveLimits, isBudgetExhausted } from "./limits";
import { planResearch } from "./planner";
import {
  createSurfSession,
  getSurfSession,
  getSessionStore,
  recordAction,
  finishAction,
  setSurfSessionStatus,
  syncEvidenceToSession,
} from "./session";
import { routeSearch, runDeepSearch } from "../search/router";
import { searchPlatform, adapterForUrl } from "../platforms";
import { attachScores, scoreSource } from "../evidence/score";
import { assertSafeUrl } from "../guard";

export interface StartResearchInput {
  objective: string;
  userId: string;
  workspaceId?: string;
  depth?: SurfPlan["depth"];
  maxSearches?: number;
  maxPages?: number;
  maxRuntimeMs?: number;
}

const DEEP_DEPTHS: SurfPlan["depth"][] = ["deep", "exhaustive"];

/**
 * Start a research run and return the session immediately. The loop runs in
 * the background (the Express process stays alive until it finishes); callers
 * poll the session via `getSurfSession` / the RPC layer.
 */
export async function startResearch(input: StartResearchInput): Promise<SurfSession> {
  const objective = input.objective.trim();
  if (!objective) throw new Error("A research objective is required.");

  const plan = await planResearch(objective, input.depth);
  const session = createSurfSession({
    userId: input.userId,
    workspaceId: input.workspaceId,
    objective,
    plan,
  });

  void runResearchLoop(session.id, {
    maxSearches: input.maxSearches,
    maxPages: input.maxPages,
    maxRuntimeMs: input.maxRuntimeMs,
  });

  return session;
}

interface LoopOptions {
  maxSearches?: number;
  maxPages?: number;
  maxRuntimeMs?: number;
}

async function runResearchLoop(sessionId: string, opts: LoopOptions): Promise<void> {
  const session = getSurfSession(sessionId);
  if (!session || !session.plan) {
    setSurfSessionStatus(sessionId, "failed", "Session or plan missing at loop start.");
    return;
  }

  const plan = session.plan;
  const limits = resolveLimits(plan.depth, opts);

  let searches = 0;
  let pages = 0;
  const startedAt = Date.now();

  try {
    const queries = [...plan.queries];
    let queueIndex = 0;

    while (queueIndex < queries.length) {
      const budget = isBudgetExhausted({ limits, searches, pages, depth: 0, startedAt });
      if (budget.exhausted) break;
      if (stopEarly(sessionId)) break;

      const query = queries[queueIndex++];
      searches++;

      const action = recordAction(sessionId, "search", `Searching: "${query.slice(0, 80)}"`);
      const deep = DEEP_DEPTHS.includes(plan.depth);
      let results: SearchResult[] = [];
      let provider = "";
      let note = "";

      try {
        if (deep) {
          const run = await runDeepSearch(query, { limit: 8 });
          results = run.results;
          provider = run.provider;
          note = run.note ?? "";
        } else {
          const run = await routeSearch({ query, intent: "general", options: { limit: 8 } });
          results = run.results;
          provider = run.provider;
          note = run.note ?? "";
        }
      } catch (err) {
        finishAction(action, "error", err instanceof Error ? err.message : String(err));
        continue;
      }

      if (results.length === 0) {
        finishAction(action, "error", note || "No results from any configured provider.");
        continue;
      }
      finishAction(action, "done", `${results.length} results via ${provider}`);

      for (const hit of results) {
        const budget = isBudgetExhausted({ limits, searches, pages, depth: 0, startedAt });
        if (budget.exhausted) break;
        if (stopEarly(sessionId)) break;
        pages++;

        await processHit(sessionId, hit);
      }

      // Re-query when the evidence pool is thin and the query queue is empty.
      if (queueIndex >= queries.length && !stopEarly(sessionId) && sessionEvidenceCount(sessionId) < 3 && searches < limits.maxSearches) {
        const followUp = followUpQuery(plan, queries);
        if (followUp) queries.push(followUp);
      }
    }

    // Platform-scoped pass: when the plan names a platform and general search
    // produced nothing, ask the platform adapter directly (honest about mode).
    if (sessionEvidenceCount(sessionId) === 0 && searches < limits.maxSearches) {
      const platform = (plan.sources as string[]).find((s) =>
        ["reddit", "github", "youtube", "x", "instagram", "linkedin"].includes(s),
      );
      if (platform) {
        const action = recordAction(sessionId, "platform_search", `Searching ${platform} directly…`);
        try {
          const res = await searchPlatform({ platform: platform as never, query: plan.objective, options: { limit: 5 } });
          if (res.results.length === 0) {
            finishAction(action, "error", res.note ?? `No accessible search on ${platform}.`);
          } else {
            finishAction(action, "done", `${res.results.length} results on ${platform}`);
            for (const hit of res.results.slice(0, limits.maxPages)) {
              if (pages >= limits.maxPages) break;
              pages++;
              await processHit(sessionId, hit);
            }
          }
        } catch (err) {
          finishAction(action, "error", err instanceof Error ? err.message : String(err));
        }
      }
    }

    syncEvidenceToSession(sessionId);
    setSurfSessionStatus(sessionId, "complete");
  } catch (err) {
    syncEvidenceToSession(sessionId);
    setSurfSessionStatus(sessionId, "failed", err instanceof Error ? err.message : String(err));
  }
}

/** Fetch, scan and extract evidence from a single result URL. */
async function processHit(sessionId: string, hit: SearchResult): Promise<void> {
  const store = getSessionStore(sessionId);
  if (!store) return;

  // Score the source from the hit (relevance against the objective).
  const [scored] = attachScores([hit], getSurfSession(sessionId)?.objective ?? "");
  const source: Source = { ...scored };
  store.trackSource(source);

  let url = hit.url;
  try {
    url = assertSafeUrl(hit.url);
  } catch {
    return;
  }

  const action = recordAction(sessionId, "open", `Opening: ${source.title.slice(0, 60)}`, { provider: hit.provider });
  let page;
  try {
    page = await adapterForUrl(url).extractPage(url);
  } catch {
    page = null;
  }

  if (!page) {
    finishAction(action, "error", "Page could not be loaded.");
    return;
  }
  finishAction(action, "done", `${page.text.length.toLocaleString()} chars extracted`);

  // Quarantine injection-suspect pages — never let them into the evidence pool.
  if (page.injection?.suspect) {
    recordAction(sessionId, "extract", `Quarantined: ${source.title.slice(0, 60)} (possible prompt injection)`);
    return;
  }

  extractEvidence(store, source, page.text, getSurfSession(sessionId)?.objective ?? "");
}

const STOPWORDS = new Set([
  "the", "and", "for", "with", "that", "this", "from", "have", "what", "when",
  "where", "about", "into", "over", "their", "they", "them", "there", "than",
  "then", "will", "would", "should", "could", "also", "were", "been", "being",
  "which", "while", "your", "our",
]);

function objectiveKeywords(objective: string): string[] {
  return objective
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((word) => word.length > 3 && !STOPWORDS.has(word))
    .slice(0, 12);
}

/** Prefer sentences that mention the objective and carry concrete detail. */
function scoreSentence(sentence: string, keywords: string[], index: number): number {
  const lower = sentence.toLowerCase();
  let score = 0;
  for (const keyword of keywords) {
    if (lower.includes(keyword)) score += 1;
  }
  if (/\d/.test(sentence)) score += 0.5;
  if (sentence.length >= 120) score += 0.25;
  // Page leads usually state context first; prefer it, weakly.
  if (index < 4) score += 0.25;
  return score;
}

function normalizeText(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, "").replace(/\s+/g, " ").trim();
}

/** Drop sentences that merely restate evidence already captured (boilerplate). */
function isNearDuplicate(store: NonNullable<ReturnType<typeof getSessionStore>>, sentence: string): boolean {
  const norm = normalizeText(sentence);
  if (norm.length < 40) return true;
  return store.all().some((existing) => {
    const current = normalizeText(existing.excerpt);
    if (!current || current.length < 40) return false;
    return current.includes(norm) || norm.includes(current);
  });
}

/** Pull objective-relevant sentences from a page into the evidence store. */
function extractEvidence(
  store: NonNullable<ReturnType<typeof getSessionStore>>,
  source: Source,
  text: string,
  objective: string,
): void {
  const keywords = objectiveKeywords(objective);
  const candidates = text
    .split(/(?<=[.!?])\s+|\n+/)
    .map((s) => s.trim())
    .filter((s) => s.length >= 40 && s.length <= 320)
    .map((sentence, index) => ({ sentence, index }))
    .sort((a, b) => scoreSentence(b.sentence, keywords, b.index) - scoreSentence(a.sentence, keywords, a.index))
    .slice(0, 8);

  let added = 0;
  for (const { sentence } of candidates) {
    if (added >= 4) break;
    if (store.count() >= 12) break;
    if (isNearDuplicate(store, sentence)) continue;
    store.add({
      sourceUrl: source.url,
      sourceTitle: source.title,
      platform: source.platform,
      claim: sentence.slice(0, 200),
      excerpt: sentence,
      accessMode: source.accessMode,
      confidence: source.score?.overall ?? 0.6,
    });
    added++;
  }
}

/** Stop once we have a decent multi-source evidence base. */
function stopEarly(sessionId: string): boolean {
  const store = getSessionStore(sessionId);
  if (!store) return false;
  const distinctSources = new Set(store.all().map((e) => e.sourceUrl)).size;
  return store.count() >= 6 && distinctSources >= 3;
}

function sessionEvidenceCount(sessionId: string): number {
  return getSessionStore(sessionId)?.count() ?? 0;
}

function followUpQuery(plan: SurfPlan, queries: string[]): string | null {
  const platform = (plan.sources as string[]).find((s) =>
    ["reddit", "github", "youtube", "x"].includes(s),
  );
  if (!platform) return null;
  const variant = `${plan.objective} ${platform}`;
  return queries.includes(variant) ? null : variant;
}

/** Get a live snapshot of a session (actions + evidence + optional report). */
export function getResearch(id: string): SurfSession | undefined {
  const session = getSurfSession(id);
  if (session) syncEvidenceToSession(id);
  return session;
}

export { scoreSource };
