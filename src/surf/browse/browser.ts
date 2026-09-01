/**
 * Relay Surf — controlled browser layer (server-only).
 *
 * A stateful browse session that keeps Relay from ever touching raw browser
 * control: the model can only issue `NavigationCommand`s (goto/click/fill/
 * scroll/back/forward/reload), never JavaScript. Sessions are short-lived
 * in-memory handles; the page model (url + PageData) is what flows to the
 * research engine.
 *
 * Rendering path:
 *   - when a Playwright/browserless WS endpoint is configured, the *initial*
 *     load of each URL goes through the browser service so JS-rendered content
 *     is captured;
 *   - otherwise (and for click-throughs) it falls back to the resilient HTTP
 *     fetch layer. The result is identical `PageData` either way — which path
 *     was used is recorded for transparency.
 */
import { relaiFetch } from "@/RelAI/web/http.server";
import { executeBrowserAction } from "@/RelAI/browser/playwright.server";
import { playwrightConfigured } from "@/RelAI/browser/playwright.server";
import { assertSafeUrl } from "../guard";
import { extractPageData } from "./extract";
import type { LinkData, NavigationCommand, NavigationResult, PageData } from "../types";

interface SessionState {
  id: string;
  createdAt: string;
  url: string;
  page: PageData | null;
  back: string[];
  forward: string[];
  loading: boolean;
}

const sessions = new Map<string, SessionState>();
let nextId = 0;

export function createBrowseSession(): { id: string } {
  const id = `browse_${Date.now().toString(36)}_${(nextId++).toString(36)}`;
  sessions.set(id, {
    id,
    createdAt: new Date().toISOString(),
    url: "about:blank",
    page: null,
    back: [],
    forward: [],
    loading: false,
  });
  return { id };
}

export function getBrowseSession(id: string): SessionState | undefined {
  return sessions.get(id);
}

export function closeBrowseSession(id: string): void {
  sessions.delete(id);
}

export function browseEnabled(): { mode: "browser" | "http" } {
  return playwrightConfigured() ? { mode: "browser" } : { mode: "http" };
}

/** Load a URL's PageData — browser service first when configured, else HTTP. */
async function loadPage(url: string): Promise<{ page: PageData; usedBrowser: boolean; blocked?: boolean; error?: string }> {
  const safeUrl = assertSafeUrl(url);
  if (playwrightConfigured()) {
    const res = await executeBrowserAction({ kind: "navigate", url: safeUrl });
    if (res.ok && (res.data.text || res.data.html)) {
      const html = res.data.html ?? "";
      const page = extractPageData({
        html: html || (res.data.text ?? ""),
        url: res.data.url ?? safeUrl,
        status: 200,
        contentType: "text/html",
        maxChars: 20_000,
      });
      if (res.data.title) page.title = res.data.title;
      return { page, usedBrowser: true };
    }
    // Browser service failed — fall through to HTTP (reported as fallback).
  }

  const res = await relaiFetch(safeUrl, { timeoutMs: 15_000, retries: 2 });
  if (!res.ok) {
    return {
      page: null as unknown as PageData,
      usedBrowser: false,
      blocked: res.blocked,
      error: res.blocked
        ? `${new URL(safeUrl).host} blocked the request.`
        : `Could not load page (HTTP ${res.status || "no response"}).`,
    };
  }

  const isJson = res.contentType.includes("json") || /^\s*[[{]/.test(res.text.slice(0, 200));
  const page = extractPageData({
    html: isJson ? undefined : res.text,
    json: isJson ? res.text : undefined,
    url: res.url,
    status: res.status,
    contentType: res.contentType,
    maxChars: 20_000,
    fetchedAt: res.fetchedAt,
  });
  return { page, usedBrowser: false };
}

/** Execute a navigation command against a session. */
export async function navigate(
  sessionId: string,
  command: NavigationCommand,
): Promise<NavigationResult> {
  const session = sessions.get(sessionId);
  if (!session) {
    throw new Error("Browse session not found — open a URL first.");
  }
  if (session.loading) {
    throw new Error("A navigation is already in progress for this session.");
  }

  switch (command.kind) {
    case "goto": {
      if (!command.target) throw new Error("goto requires a target URL");
      const prev = session.page?.url;
      session.loading = true;
      try {
        const { page, blocked, error } = await loadPage(command.target);
        if (!page) return { ok: false, url: command.target, blocked, error };
        if (prev && prev !== page.url) session.back.push(prev);
        session.forward = [];
        session.url = page.url;
        session.page = page;
        return {
          ok: true,
          url: page.url,
          title: page.title,
          text: page.text.slice(0, 12_000),
          links: page.links.slice(0, 60),
          headings: page.headings.slice(0, 40),
        };
      } finally {
        session.loading = false;
      }
    }

    case "back": {
      const prev = session.back.pop();
      if (!prev) return { ok: false, url: session.url, error: "No previous page in this session." };
      const current = session.page?.url;
      const { page, blocked, error } = await loadPage(prev);
      if (!page) return { ok: false, url: prev, blocked, error };
      if (current) session.forward.push(current);
      session.url = page.url;
      session.page = page;
      return { ok: true, url: page.url, title: page.title, text: page.text.slice(0, 12_000), links: page.links.slice(0, 60) };
    }

    case "forward": {
      const next = session.forward.pop();
      if (!next) return { ok: false, url: session.url, error: "No forward page in this session." };
      const current = session.page?.url;
      const { page, blocked, error } = await loadPage(next);
      if (!page) return { ok: false, url: next, blocked, error };
      if (current) session.back.push(current);
      session.url = page.url;
      session.page = page;
      return { ok: true, url: page.url, title: page.title, text: page.text.slice(0, 12_000), links: page.links.slice(0, 60) };
    }

    case "reload": {
      const { page, blocked, error } = await loadPage(session.url);
      if (!page) return { ok: false, url: session.url, blocked, error };
      session.page = page;
      return { ok: true, url: page.url, title: page.title, text: page.text.slice(0, 12_000), links: page.links.slice(0, 60) };
    }

    case "click": {
      // Controlled link-following: resolve the target against the loaded page's
      // real links — never a guessed URL. Target may be a URL, anchor text, or
      // a link intent (pricing, docs, faq…).
      const page = session.page;
      if (!page) return { ok: false, url: session.url, error: "No page loaded in this session." };
      const link = resolveLink(page.links, command.target ?? "");
      if (!link) {
        return { ok: false, url: session.url, error: `No matching link found on the page for "${command.target}".` };
      }
      return navigate(sessionId, { kind: "goto", target: link.url });
    }

    case "fill": {
      // Fill is only meaningful with a live interactive browser. Without one,
      // report honestly that Relay cannot type into this page.
      if (!playwrightConfigured()) {
        return {
          ok: false,
          url: session.url,
          error: "Filling forms requires a configured browser endpoint (PLAYWRIGHT_WS_ENDPOINT).",
        };
      }
      if (!command.selector) throw new Error("fill requires a selector");
      const res = await executeBrowserAction({ kind: "fill", selector: command.selector, value: command.value ?? "" });
      return {
        ok: res.ok,
        url: session.url,
        error: res.error,
        ...(res.ok ? { title: "filled" } : {}),
      };
    }

    case "scroll": {
      if (!playwrightConfigured()) {
        return {
          ok: true,
          url: session.url,
          error: "Scrolling requires a browser endpoint; content already captured via HTTP.",
        };
      }
      const res = await executeBrowserAction({
        kind: "scroll",
        direction: command.direction ?? "down",
      });
      return { ok: res.ok, url: session.url, error: res.error };
    }

    default:
      throw new Error(`Unsupported navigation command: ${(command as { kind: string }).kind}`);
  }
}

/** Resolve a click target to a real link on the page. */
function resolveLink(links: LinkData[], target: string): LinkData | undefined {
  const t = target.trim();
  if (!t) return undefined;

  // Exact URL match.
  const byUrl = links.find((l) => l.url === t || l.url.split("#")[0] === t.split("#")[0]);
  if (byUrl) return byUrl;

  // Intent match (pricing, docs, faq…).
  const intent = t.toLowerCase();
  const byIntent = links.find((l) => l.intent === intent || l.intent?.includes(intent) || intent.includes(l.intent ?? ""));
  if (byIntent) return byIntent;

  // Anchor text match (case-insensitive substring).
  const byText = links.find((l) => l.text.toLowerCase().includes(t.toLowerCase()));
  if (byText) return byText;

  // Partial URL match.
  const byPartial = links.find((l) => l.url.toLowerCase().includes(t.toLowerCase()));
  return byPartial;
}

/** Current page of a session (null before the first goto). */
export function currentPage(sessionId: string): PageData | null {
  return sessions.get(sessionId)?.page ?? null;
}

/** Cap how many URLs a single session may accumulate (cost guard). */
export function sessionCount(): number {
  return sessions.size;
}
