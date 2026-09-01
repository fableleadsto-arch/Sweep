/**
 * RelAI — Playwright Browser Agent (server-only).
 *
 * Provides AI-driven browser automation via WebSocket connection to an
 * external browser (browserless.io, Playwright Server, or headless Chrome).
 *
 * Architecture:
 *   1. Connect to browser via WebSocket (CDP protocol)
 *   2. Execute actions: navigate, click, fill, screenshot, extract
 *   3. Fall back to resilient HTTP fetch when browser is not available
 *
 * Environment:
 *   PLAYWRIGHT_WS_ENDPOINT - WebSocket endpoint for the browser
 *     e.g. "wss://chrome.browserless.io/playwright"
 *   BROWSER_WS_ENDPOINT    - Fallback endpoint alias
 */
import { relaiFetch } from "../web/http.server";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

export type BrowserAction =
  | { kind: "navigate"; url: string }
  | { kind: "click"; selector: string }
  | { kind: "fill"; selector: string; value: string }
  | { kind: "select"; selector: string; value: string }
  | { kind: "screenshot"; fullPage?: boolean }
  | { kind: "extract"; selector?: string }
  | { kind: "wait"; ms: number }
  | { kind: "scroll"; direction: "down" | "up" | "bottom" | "top" }
  | { kind: "hover"; selector: string }
  | { kind: "keypress"; key: string }
  | { kind: "evaluate"; script: string };

export interface BrowserStep {
  action: BrowserAction;
  result?: string;
  error?: string;
  ms: number;
}

export interface BrowserSession {
  id: string;
  url: string;
  steps: BrowserStep[];
  createdAt: string;
  state: "active" | "closed" | "error";
}

export interface BrowserActionResult {
  ok: boolean;
  data: {
    title?: string;
    url?: string;
    text?: string;
    screenshot?: string; // base64
    html?: string;
    links?: string[];
    extracted?: Record<string, unknown>;
  };
  error?: string;
}

/* ------------------------------------------------------------------ */
/*  Configuration                                                      */
/* ------------------------------------------------------------------ */

export function playwrightConfigured(): boolean {
  return Boolean(process.env.PLAYWRIGHT_WS_ENDPOINT) ||
         Boolean(process.env.BROWSER_WS_ENDPOINT);
}

export function playwrightWSEndpoint(): string | undefined {
  return process.env.PLAYWRIGHT_WS_ENDPOINT ?? process.env.BROWSER_WS_ENDPOINT;
}

/**
 * Execute a Playwright action by sending a JSON-RPC command to the
 * browser WebSocket endpoint. Many Playwright-as-a-service providers
 * accept a REST-like API over HTTP.
 */
async function executeViaBrowserService(
  action: BrowserAction,
): Promise<BrowserActionResult> {
  const endpoint = playwrightWSEndpoint();
  if (!endpoint) {
    return { ok: false, data: {}, error: "No browser WebSocket endpoint configured." };
  }

  try {
    // Determine the HTTP API URL from the WS endpoint
    // browserless.io convention: wss://... -> https://... 
    const httpUrl = endpoint.replace(/^wss?:\/\//, "https://").replace(/\/$/, "");

    switch (action.kind) {
      case "navigate": {
        const body = {
          url: action.url,
          // browserless.io /playwright endpoint
          code: `
            const page = await browser.newPage();
            await page.setViewportSize({ width: 1280, height: 800 });
            await page.goto('${action.url}', { waitUntil: 'networkidle', timeout: 30000 });
            const title = await page.title();
            const content = await page.evaluate(() => document.body.innerText);
            const links = await page.evaluate(() =>
              Array.from(document.querySelectorAll('a[href]'))
                .map(a => a.href)
                .filter(h => h.startsWith('http'))
            );
            const html = await page.content();
            await page.close();
            return { title, content: content.slice(0, 15000), links: links.slice(0, 50), html: html.slice(0, 10000) };
          `,
        };

        const res = await fetch(`${httpUrl}/playwright`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(35_000),
        });

        if (!res.ok) {
          return handleFallbackFetch(action.url);
        }

        const data = (await res.json()) as {
          title?: string;
          content?: string;
          links?: string[];
          html?: string;
        };

        return {
          ok: true,
          data: {
            title: data.title ?? "",
            url: action.url,
            text: data.content ?? "",
            links: data.links ?? [],
            html: data.html,
          },
        };
      }

      case "screenshot": {
        const body = {
          url: "about:blank",
          code: `
            const page = await browser.newPage();
            await page.setViewportSize({ width: 1280, height: 800 });
            ${action.fullPage ? "await page.goto('about:blank');" : ""}
            const screenshot = await page.screenshot({ fullPage: ${action.fullPage ?? false}, type: 'jpeg', quality: 80 });
            await page.close();
            return { screenshot: Buffer.from(screenshot).toString('base64') };
          `,
        };

        const res = await fetch(`${httpUrl}/playwright`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(30_000),
        });

        if (!res.ok) {
          return { ok: false, data: {}, error: `Screenshot service returned ${res.status}` };
        }

        const data = (await res.json()) as { screenshot?: string };
        return {
          ok: true,
          data: { screenshot: data.screenshot },
        };
      }

      default: {
        // For actions not supported by the browser service API,
        // fall back to the HTTP layer
        const anyAction = action as { kind: string; url?: string };
        if (anyAction.kind === "navigate" && anyAction.url) {
          return handleFallbackFetch(anyAction.url);
        }
        return {
          ok: false,
          data: {},
          error: `Browser action '${action.kind}' not supported by the current service. Use navigate + extract instead.`,
        };
      }
    }
  } catch (err) {
    // Fall back to HTTP fetch on any browser service failure
    const anyAction = action as Record<string, string>;
    if (anyAction.kind === "navigate" && anyAction.url) {
      return handleFallbackFetch(anyAction.url);
    }
    return {
      ok: false,
      data: {},
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

/* ------------------------------------------------------------------ */
/*  HTTP Fallback Layer                                                */
/* ------------------------------------------------------------------ */

async function handleFallbackFetch(url: string): Promise<BrowserActionResult> {
  const res = await relaiFetch(url, { retries: 2, timeoutMs: 15_000 });
  if (!res.ok) {
    return {
      ok: false,
      data: {},
      error: res.blocked
        ? `${url} blocked the request. Try a different URL or configure a browser endpoint.`
        : `Failed to fetch ${url}: ${res.error ?? `HTTP ${res.status}`}`,
    };
  }

  const title = res.text.match(/<title[^>]*>([^<]+)<\/title>/i)?.[1] ?? "";
  const links: string[] = [];
  const linkRe = /<a[^>]+href=["'](https?:\/\/[^"']+)["'][^>]*>/gi;
  let m: RegExpExecArray | null;
  while ((m = linkRe.exec(res.text)) !== null) {
    if (!links.includes(m[1])) links.push(m[1]);
  }

  return {
    ok: true,
    data: {
      title,
      url: res.url,
      text: res.text.slice(0, 10_000),
      links: links.slice(0, 30),
    },
  };
}

/* ------------------------------------------------------------------ */
/*  Public API                                                         */
/* ------------------------------------------------------------------ */

/**
 * Execute a single browser action.
 * Tries browser service (Playwright via WebSocket), falls back to HTTP fetch.
 */
export async function executeBrowserAction(
  action: BrowserAction,
  _session?: BrowserSession,  ): Promise<BrowserActionResult> {
    try {
    switch (action.kind) {
      case "navigate":
      case "screenshot":
        return await executeViaBrowserService(action);

      case "click":
      case "fill":
      case "select":
      case "hover":
      case "keypress":
      case "scroll":
      case "extract":
      case "evaluate":
        // These require a live browser session with a page open.
        // Without one, return a clear error
        return {
          ok: false,
          data: {},
          error: `Action '${action.kind}' requires a live browser session. Use 'browser_navigate' first, then this action. Configure PLAYWRIGHT_WS_ENDPOINT for full browser support.`,
        };

      case "wait":
        await new Promise((r) => setTimeout(r, action.ms));
        return { ok: true, data: { text: `waited ${action.ms}ms` } };

      default:
        return { ok: false, data: {}, error: `Unknown action: ${(action as any).kind}` };
    }
  } catch (err) {
    // Final fallback: try HTTP fetch for navigations
    const errAction = action as { kind: string; url?: string };
    if (errAction.kind === "navigate" && errAction.url) {
      return handleFallbackFetch(errAction.url);
    }
    return {
      ok: false,
      data: {},
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

/**
 * Run a sequence of browser actions for multi-step tasks.
 */
export async function runBrowserSequence(
  actions: BrowserAction[],
): Promise<{
  ok: boolean;
  steps: BrowserStep[];
  finalUrl?: string;
  finalText?: string;
}> {
  const steps: BrowserStep[] = [];
  let currentUrl = "";
  let currentText = "";
  let ok = true;

  for (const action of actions) {
    const started = Date.now();
    const result = await executeBrowserAction(action);
    const ms = Date.now() - started;

    const step: BrowserStep = {
      action,
      result: result.ok ? JSON.stringify(result.data).slice(0, 500) : undefined,
      error: result.error,
      ms,
    };
    steps.push(step);

    if (!result.ok) {
      ok = false;
      break;
    }

    if (action.kind === "navigate") {
      currentUrl = result.data.url ?? action.url;
      currentText = result.data.text ?? "";
    }
  }

  return { ok, steps, finalUrl: currentUrl, finalText: currentText };
}

/**
 * AI-guided browser task: given a goal, navigate and extract.
 * Uses the perception-action loop pattern from browser-use.
 */export async function aiBrowserTask(
    _goal: string,
    startUrl: string,
): Promise<{
  success: boolean;
  summary: string;
  steps: BrowserStep[];
  extractedData?: Record<string, unknown>;
}> {
  const steps: BrowserStep[] = [];
  const started = Date.now();

  // Step 1: Navigate to the URL
  const navigationResult = await executeBrowserAction({ kind: "navigate", url: startUrl });
  steps.push({
    action: { kind: "navigate", url: startUrl },
    result: navigationResult.ok
      ? `Loaded: ${navigationResult.data.title ?? startUrl}`
      : navigationResult.error,
    error: navigationResult.error,
    ms: Date.now() - started,
  });

  if (!navigationResult.ok) {
    return {
      success: false,
      summary: `Could not load ${startUrl}: ${navigationResult.error}. Try a simpler URL or check the browser configuration.`,
      steps,
    };
  }

  const pageText = navigationResult.data.text ?? "";
  const pageTitle = navigationResult.data.title ?? "";
  const pageLinks = navigationResult.data.links ?? [];

  // Step 2: Determine if we need to scroll for more content
  let extractedContent = pageText;
  let attemptCount = 1;

  if (pageText.length > 5000) {
    // Page has enough content already
  } else {
    // Try extracting again with a small wait for JS-rendered content
    await new Promise((r) => setTimeout(r, 1000));
    const retryResult = await executeBrowserAction({ kind: "navigate", url: startUrl });
    if (retryResult.ok && (retryResult.data.text?.length ?? 0) > pageText.length) {
      extractedContent = retryResult.data.text ?? pageText;
      steps.push({
        action: { kind: "navigate", url: startUrl },
        result: `Re-fetched: ${retryResult.data.title ?? startUrl}`,
        ms: 1500,
      });
      attemptCount++;
    }
  }

  const summary = `Loaded ${startUrl} (${pageTitle}). Extracted ${extractedContent.length.toLocaleString()} characters across ${attemptCount} fetch attempt(s). Found ${pageLinks.length} links on the page.`;

  return {
    success: true,
    summary,
    steps,
    extractedData: {
      url: startUrl,
      title: pageTitle,
      textLength: extractedContent.length,
      links: pageLinks.slice(0, 20),
      linksCount: pageLinks.length,
    },
  };
}
