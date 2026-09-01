/**
 * Relay Surf — research limits (server-only).
 *
 * Configurable cost ceilings. The research engine checks these at every step
 * and stops early when enough evidence exists, so a simple question never
 * burns through 100 pages.
 */
import { DEPTH_LIMITS, DEFAULT_SURF_LIMITS, type SurfLimits, type SurfPlan } from "../types";

export function resolveLimits(depth: SurfPlan["depth"], overrides?: Partial<SurfLimits>): SurfLimits {
  const base = { ...DEFAULT_SURF_LIMITS, ...(DEPTH_LIMITS[depth] ?? {}) };
  return { ...base, ...(overrides ?? {}) };
}

export function isBudgetExhausted(input: {
  limits: SurfLimits;
  searches: number;
  pages: number;
  depth: number;
  startedAt: number;
}): { exhausted: boolean; reason?: string } {
  if (input.searches >= input.limits.maxSearches) {
    return { exhausted: true, reason: `search budget reached (${input.limits.maxSearches})` };
  }
  if (input.pages >= input.limits.maxPages) {
    return { exhausted: true, reason: `page budget reached (${input.limits.maxPages})` };
  }
  if (input.depth >= input.limits.maxNavigationDepth) {
    return { exhausted: true, reason: `navigation depth reached (${input.limits.maxNavigationDepth})` };
  }
  if (Date.now() - input.startedAt >= input.limits.maxRuntimeMs) {
    return { exhausted: true, reason: "runtime budget reached" };
  }
  return { exhausted: false };
}
