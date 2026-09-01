/**
 * Relay Surf — server functions (RPC surface).
 *
 * These `createServerFn`s are auto-discovered by `server/rpc.ts`
 * (`registerRpcModules` walks `src/` for `*.functions.ts`) and exposed at
 * `/api/rpc/surf/functions/*`. Every function authenticates via
 * `requireSupabaseAuth` and goes through the shared RPC dispatch hardening.
 *
 * IMPORTANT: surf internals are never imported statically — each handler
 * lazy-loads `./surf.server`, the server-only facade. The client build stubs
 * `.server` specifiers, so this keeps the browser bundle free of `node:`/
 * Playwright/provider code, matching the convention used by every other
 * `*.functions.ts` in the repo.
 */
import { z } from "zod";
import { createServerFn } from "@/lib/rpc/runtime";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import type { SupabaseClient } from "@supabase/supabase-js";
import { assertSafeUrl, clampInt, requireText } from "./guard";
import type { SurfSession } from "./types";

const LIMIT = z.number().int().min(1).max(30).optional();

async function workspaceIdOf(supabase: SupabaseClient, userId: string): Promise<string | undefined> {
  try {
    const { data } = await supabase
      .from("workspace_members")
      .select("workspace_id")
      .eq("user_id", userId)
      .order("created_at", { ascending: true })
      .limit(1)
      .maybeSingle();
    return (data as { workspace_id?: string } | null)?.workspace_id;
  } catch {
    return undefined;
  }
}

/* ------------------------------------------------------------------ */
/*  Search                                                             */
/* ------------------------------------------------------------------ */

export const surfSearch = createServerFn("surf/functions/surfSearch", { method: "POST" })
  .middleware([requireSupabaseAuth])
  .validator((input: unknown) =>
    z
      .object({
        query: z.string().trim().min(1).max(500),
        intent: z.enum(["general", "technical", "news", "deep", "platform"]).default("general"),
        limit: LIMIT,
        site: z.string().max(120).optional(),
        page: z.number().int().min(0).max(100).optional(),
      })
      .parse(input),
  )
  .handler(async ({ data }) => {
    const opts = { limit: data.limit, site: data.site || undefined, page: data.page ?? undefined };
    const { routeSearch, runDeepSearch } = await import("./surf.server");
    if (data.intent === "deep") {
      return runDeepSearch(data.query, opts);
    }
    return routeSearch({ query: data.query, intent: data.intent, options: opts });
  });

export const surfSearchSite = createServerFn("surf/functions/surfSearchSite", { method: "POST" })
  .middleware([requireSupabaseAuth])
  .validator((input: unknown) =>
    z
      .object({
        query: z.string().trim().min(1).max(500),
        site: z.string().max(120).optional(),
        platform: z.enum(["reddit", "x", "instagram", "youtube", "github", "linkedin", "generic"]).optional(),
        limit: LIMIT,
      })
      .parse(input),
  )
  .handler(async ({ data }) => {
    const platform =
      data.platform ??
      (data.site ? guessPlatformFromSite(data.site) : "generic");
    const { searchPlatform } = await import("./surf.server");
    return searchPlatform({
      platform: platform as never,
      query: data.query,
      options: { limit: data.limit },
    });
  });

export const surfSearchPlatform = createServerFn("surf/functions/surfSearchPlatform", { method: "POST" })
  .middleware([requireSupabaseAuth])
  .validator((input: unknown) =>
    z
      .object({
        platform: z.enum(["reddit", "x", "instagram", "youtube", "github", "linkedin"]),
        query: z.string().trim().min(1).max(500),
        limit: LIMIT,
      })
      .parse(input),
  )
  .handler(async ({ data }) => {
    const { searchPlatform } = await import("./surf.server");
    return searchPlatform({
      platform: data.platform,
      query: data.query,
      options: { limit: data.limit },
    });
  });

/* ------------------------------------------------------------------ */
/*  Browse                                                             */
/* ------------------------------------------------------------------ */

export const surfOpen = createServerFn("surf/functions/surfOpen", { method: "POST" })
  .middleware([requireSupabaseAuth])
  .validator((input: unknown) => z.object({ url: z.string(), sessionId: z.string().optional() }).parse(input))
  .handler(async ({ data }) => {
    const safeUrl = assertSafeUrl(data.url);
    const { createBrowseSession, getBrowseSession, navigate } = await import("./surf.server");
    let sid = typeof data.sessionId === "string" && getBrowseSession(data.sessionId) ? data.sessionId : "";
    if (!sid) sid = createBrowseSession().id;
    const result = await navigate(sid, { kind: "goto", target: safeUrl });
    if (!result.ok) throw new Error(result.error ?? "Could not open URL.");
    return { sessionId: sid, url: result.url, title: result.title, text: result.text, links: result.links, headings: result.headings };
  });

export const surfNavigate = createServerFn("surf/functions/surfNavigate", { method: "POST" })
  .middleware([requireSupabaseAuth])
  .validator((input: unknown) =>
    z
      .object({
        sessionId: z.string().min(1),
        action: z.enum(["back", "forward", "reload", "click"]),
        target: z.string().optional(),
      })
      .parse(input),
  )
  .handler(async ({ data }) => {
    const { getBrowseSession, navigate } = await import("./surf.server");
    if (!getBrowseSession(data.sessionId)) throw new Error("Browse session not found.");
    const kind = data.action === "back" || data.action === "forward" || data.action === "reload" ? data.action : "click";
    const result = await navigate(data.sessionId, { kind, target: data.target });
    if (!result.ok) throw new Error(result.error ?? "Navigation failed.");
    return { sessionId: data.sessionId, url: result.url, title: result.title, text: result.text, links: result.links };
  });

export const surfExtractPage = createServerFn("surf/functions/surfExtractPage", { method: "POST" })
  .middleware([requireSupabaseAuth])
  .validator((input: unknown) =>
    z.object({ sessionId: z.string().optional(), url: z.string().optional(), maxChars: z.number().int().min(500).max(40_000).optional() }).parse(input),
  )
  .handler(async ({ data }) => {
    const { currentPage, createBrowseSession, navigate } = await import("./surf.server");
    let page = data.sessionId ? currentPage(data.sessionId) : null;
    if (!page && data.url) {
      const sid = createBrowseSession().id;
      const result = await navigate(sid, { kind: "goto", target: assertSafeUrl(data.url) });
      if (!result.ok) throw new Error(result.error ?? "Could not load page.");
      page = currentPage(sid);
    }
    if (!page) throw new Error("No page loaded — open a URL or provide one.");
    const max = data.maxChars ?? 10_000;
    return {
      url: page.url,
      title: page.title,
      text: page.text.slice(0, max),
      links: page.links.slice(0, 80),
      headings: page.headings.slice(0, 60),
      metadata: page.metadata,
      truncated: page.truncated || page.text.length > max,
    };
  });

export const surfFindOnPage = createServerFn("surf/functions/surfFindOnPage", { method: "POST" })
  .middleware([requireSupabaseAuth])
  .validator((input: unknown) => z.object({ sessionId: z.string().min(1), query: z.string().trim().min(1).max(200) }).parse(input))
  .handler(async ({ data }) => {
    const { currentPage, findOnPage } = await import("./surf.server");
    const page = currentPage(data.sessionId);
    if (!page) throw new Error("No page loaded in this session.");
    return findOnPage(page, data.query);
  });

/* ------------------------------------------------------------------ */
/*  Research                                                           */
/* ------------------------------------------------------------------ */

export const surfResearchStart = createServerFn("surf/functions/surfResearchStart", { method: "POST" })
  .middleware([requireSupabaseAuth])
  .validator((input: unknown) =>
    z
      .object({
        objective: z.string().trim().min(3).max(2000),
        depth: z.enum(["quick", "standard", "deep", "exhaustive"]).default("standard"),
      })
      .parse(input),
  )
  .handler(async ({ data, context }) => {
    const { supabase, userId } = context;
    const { startResearch } = await import("./surf.server");
    const session = await startResearch({
      objective: data.objective,
      userId,
      workspaceId: await workspaceIdOf(supabase, userId),
      depth: data.depth,
    });
    return { id: session.id, status: session.status, plan: session.plan };
  });

export const surfResearchGet = createServerFn("surf/functions/surfResearchGet", { method: "GET" })
  .middleware([requireSupabaseAuth])
  .validator((input: unknown) => z.object({ id: z.string().min(1) }).parse(input))
  .handler(async ({ data, context }) => {
    const { getResearch, synthesizeReport } = await import("./surf.server");
    const session = getResearch(data.id);
    if (!session || session.userId !== context.userId) {
      throw new Error("Research session not found.");
    }
    const report =
      session.status === "complete" || session.status === "failed"
        ? await synthesizeReport(session)
        : undefined;
    return snapshot(session, report);
  });

export const surfResearchList = createServerFn("surf/functions/surfResearchList", { method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const { listSurfSessions } = await import("./surf.server");
    const sessions = listSurfSessions(context.userId, 10).map((s) => ({
      id: s.id,
      objective: s.objective,
      status: s.status,
      startedAt: s.startedAt,
      completedAt: s.completedAt,
      evidenceCount: s.evidence.length,
      sourceCount: s.sources.length,
    }));
    return { sessions };
  });

/* ------------------------------------------------------------------ */
/*  Status                                                             */
/* ------------------------------------------------------------------ */

export const surfProviderStatus = createServerFn("surf/functions/surfProviderStatus", { method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async () => {
    const { providerSummary, browseEnabled } = await import("./surf.server");
    return { providers: providerSummary(), browserMode: browseEnabled().mode };
  });

export const surfProviderHealth = createServerFn("surf/functions/surfProviderHealth", { method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async () => {
    const { providerHealth, browseEnabled } = await import("./surf.server");
    const [health, browserMode] = await Promise.all([providerHealth(), browseEnabled().mode]);
    return { health, browserMode };
  });

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function guessPlatformFromSite(site: string): string {
  const s = site.toLowerCase();
  if (s.includes("reddit")) return "reddit";
  if (s.includes("github")) return "github";
  if (s.includes("youtube") || s.includes("youtu.be")) return "youtube";
  if (s.includes("x.com") || s.includes("twitter")) return "x";
  if (s.includes("instagram")) return "instagram";
  if (s.includes("linkedin")) return "linkedin";
  return "generic";
}

function snapshot(
  session: SurfSession,
  report?: unknown,
): Record<string, unknown> {
  return {
    id: session.id,
    objective: session.objective,
    plan: session.plan,
    sources: session.sources.map((s) => ({ ...s, title: s.title.slice(0, 200) })),
    evidence: session.evidence.map((e) => ({ ...e, excerpt: e.excerpt.slice(0, 400) })),
    actions: session.actions,
    startedAt: session.startedAt,
    completedAt: session.completedAt,
    status: session.status,
    error: session.error,
    report,
  };
}

export { requireText, clampInt };
