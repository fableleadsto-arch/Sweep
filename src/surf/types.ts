/**
 * Relay Surf — core contracts (shared, pure types + constants).
 *
 * Relay Surf is the web-surfing and research subsystem of Relay AI. It sits
 * below Relay Brain and above the raw web: every capability — SEARCH, BROWSE,
 * RESEARCH — is expressed as a controlled `SurfTool` with validated inputs.
 *
 * This module is types + constants only, so it is safe to import from client
 * and server code. Server-only implementations live in the sibling *.server.ts
 * modules and are always lazy-imported.
 */

/* ------------------------------------------------------------------ */
/*  Tools                                                              */
/* ------------------------------------------------------------------ */

export type SurfToolCategory =
  | "search"
  | "browse"
  | "research"
  | "platform"
  | "evidence"
  | "document";

export interface SurfTool {
  name: string;
  description: string;
  category: SurfToolCategory;
  /** Which surfaces are allowed to call this tool ("" = everywhere). */
  restrictedTo?: string[];
  /** Runtime availability gate — an unconfigured provider is never advertised. */
  available?: boolean | (() => boolean | Promise<boolean>);
  inputSchema: {
    type: "object";
    properties: Record<string, { type: string; description?: string; enum?: string[] }>;
    required?: string[];
  };
  execute(input: unknown): Promise<SurfToolResult>;
}

export interface SurfToolResult {
  ok: boolean;
  data: unknown;
  /** Human-readable note shown in the live activity stream (never raw logs). */
  activity?: string;
  error?: string;
}

/* ------------------------------------------------------------------ */
/*  Search                                                             */
/* ------------------------------------------------------------------ */

export type SearchAccessMode = "public" | "authenticated" | "unavailable";

export interface SearchResult {
  url: string;
  title: string;
  snippet?: string;
  /** Which provider/engine produced this hit. */
  provider: string;
  /** How the content was reached — public indexing vs authenticated. */
  accessMode: SearchAccessMode;
  /** Present when the hit came through an authenticated connector. */
  platform?: string;
  /** Present when the provider knows when the page was published. */
  publishedAt?: string;
  relevance?: number;
  metadata?: Record<string, string>;
}

export interface SearchOptions {
  limit?: number;
  site?: string;
  timeRange?: "day" | "week" | "month" | "year";
  after?: string;
  before?: string;
  country?: string;
  language?: string;
  /** SERP page to request (1 = first). Only the keyless chain paginates. */
  page?: number;
  /**
   * Opt into the keyless provider's full fan-out (aggregate across every HTML
   * engine + API provider), the same mode lead discovery uses. Off by default
   * so chat/search keep the fast first-hit-wins path.
   */
  aggregate?: boolean;
}

export interface SearchProvider {
  name: string;
  /** Whether this provider is actually configured on this deployment. */
  configured(): boolean;
  /** Whether this provider is reachable/authorized right now. */
  health?(): Promise<boolean>;
  /**
   * Run a search. Returns the results plus honest metadata about how the run
   * went (which engine served it, whether it was blocked, a human note) so the
   * router and UI never guess.
   */
  search(query: string, options?: SearchOptions): Promise<SearchRun>;
  capabilities?: {
    news?: boolean;
    siteFilter?: boolean;
    technical?: boolean;
    platforms?: string[];
  };
}

/** What a single provider run returns (before routing wraps it). */
export interface SearchRun {
  /** Which provider/engine served this run. */
  provider: string;
  results: SearchResult[];
  /** Zero results because the provider blocked/errored — surfaced honestly. */
  blocked?: boolean;
  /** Human-readable note about the run (fallback reason, provider warning…). */
  note?: string;
  /** Per-provider failure messages (blocked engines, API errors). */
  errors?: string[];
}

export interface SearchRunResult {
  provider: string;
  query: string;
  results: SearchResult[];
  /** Zero results because the provider blocked/errored — surfaced honestly. */
  blocked?: boolean;
  note?: string;
  /** Per-provider failure messages surfaced to the caller/UI. */
  errors?: string[];
}

/* ------------------------------------------------------------------ */
/*  Page / browse                                                      */
/* ------------------------------------------------------------------ */

export interface LinkData {
  url: string;
  text: string;
  /** Semantic hints derived from anchor text + href (pricing, docs, faq…). */
  intent?: string;
  /** Whether the link points outside the current host. */
  external?: boolean;
}

export interface HeadingData {
  level: number;
  text: string;
  id?: string;
}

export interface PageData {
  url: string;
  title: string;
  description?: string;
  text: string;
  markdown?: string;
  links: LinkData[];
  headings: HeadingData[];
  metadata: Record<string, string>;
  structuredData?: unknown;
  /** True when the page was truncated to fit a token budget. */
  truncated: boolean;
  fetchedAt: string;
  status: number;
  contentType: string;
  accessMode: SearchAccessMode;
  blocked?: boolean;
  /** Injection-scan result: does the page try to override instructions? */
  injection?: InjectionAssessment;
}

export interface InjectionAssessment {
  suspect: boolean;
  /** Matched patterns, e.g. "ignore previous instructions". */
  signals: string[];
}

/** A section of a page extracted by find-on-page. */
export interface PageSection {
  heading?: string;
  headingLevel?: number;
  headingId?: string;
  text: string;
  /** Character offset into the cleaned page text. */
  start: number;
  end: number;
  /** Browsing context: the session's current URL. */
  url: string;
}

export interface NavigationCommand {
  kind: "goto" | "click" | "fill" | "scroll" | "back" | "forward" | "reload";
  /** For click: a selector or anchor text label resolved against page links. */
  target?: string;
  value?: string;
  selector?: string;
  direction?: "down" | "up" | "bottom" | "top";
}

export interface NavigationResult {
  ok: boolean;
  url: string;
  title?: string;
  text?: string;
  links?: LinkData[];
  headings?: HeadingData[];
  error?: string;
  blocked?: boolean;
}

/* ------------------------------------------------------------------ */
/*  Platforms                                                          */
/* ------------------------------------------------------------------ */

export type SurfPlatform = "reddit" | "x" | "instagram" | "youtube" | "github" | "linkedin" | "generic";

export interface PlatformSearchOptions {
  limit?: number;
  subreddit?: string;
  timeRange?: SearchOptions["timeRange"];
}

export interface PlatformAdapter {
  platform: SurfPlatform;
  /** Whether the adapter can handle a URL (posts, profiles, repos…). */
  canHandle(url: string): boolean;
  /**
   * Native/platform search where publicly accessible. Returns empty (with a
   * note) when there is no accessible native search — callers then fall back
   * to general web search and MUST NOT claim a native search happened.
   */
  search(query: string, options?: PlatformSearchOptions): Promise<{ results: SearchResult[]; note?: string; accessMode: SearchAccessMode }>;
  /** Extract a platform page (post + comments, repo README, video…). */
  extractPage(url: string): Promise<PageData | null>;
}

export const PLATFORMS: SurfPlatform[] = ["reddit", "x", "instagram", "youtube", "github", "linkedin", "generic"];

/* ------------------------------------------------------------------ */
/*  Evidence & sources                                                 */
/* ------------------------------------------------------------------ */

export type SourceType = "official" | "primary" | "community" | "secondary" | "news" | "documentation" | "repository" | "social" | "unknown";

export interface Source {
  title: string;
  url: string;
  platform?: SurfPlatform;
  type?: SourceType;
  accessMode?: SearchAccessMode;
  retrievedAt: string;
  score?: SourceScore;
}

export interface SourceScore {
  relevance: number;
  authority: number;
  freshness: number;
  directness: number;
  overall: number;
}

export interface Evidence {
  id: string;
  sourceUrl: string;
  sourceTitle: string;
  platform?: SurfPlatform;
  excerpt: string;
  claim: string;
  timestamp?: string;
  accessMode?: SearchAccessMode;
  confidence: number;
}

export interface Citation {
  index: number;
  claim: string;
  source: Source;
}

/* ------------------------------------------------------------------ */
/*  Research session                                                   */
/* ------------------------------------------------------------------ */

export type SurfSessionStatus = "running" | "complete" | "failed" | "cancelled";

export interface SurfAction {
  id: string;
  kind: "plan" | "search" | "open" | "navigate" | "extract" | "site_search" | "platform_search" | "verify" | "synthesize" | "error";
  description: string;
  status: "running" | "done" | "error";
  provider?: string;
  startedAt: string;
  endedAt?: string;
  ms?: number;
  detail?: string;
}

export interface SurfSession {
  id: string;
  userId: string;
  workspaceId?: string;
  objective: string;
  plan?: SurfPlan;
  sources: Source[];
  evidence: Evidence[];
  actions: SurfAction[];
  startedAt: string;
  completedAt?: string;
  status: SurfSessionStatus;
  error?: string;
}

export interface SurfPlan {
  objective: string;
  queries: string[];
  sources: string[];
  requiredInformation: string[];
  verificationRequirements: string[];
  depth: "quick" | "standard" | "deep" | "exhaustive";
}

export interface SurfLimits {
  maxSearches: number;
  maxPages: number;
  maxNavigationDepth: number;
  maxRuntimeMs: number;
  maxTokens: number;
  maxConcurrentPages: number;
}

export const DEFAULT_SURF_LIMITS: SurfLimits = {
  maxSearches: 8,
  maxPages: 12,
  maxNavigationDepth: 3,
  maxRuntimeMs: 120_000,
  maxTokens: 24_000,
  maxConcurrentPages: 2,
};

export const DEPTH_LIMITS: Record<SurfPlan["depth"], Partial<SurfLimits>> = {
  quick: { maxSearches: 2, maxPages: 3, maxNavigationDepth: 1, maxRuntimeMs: 30_000, maxTokens: 8_000 },
  standard: { maxSearches: 5, maxPages: 8, maxNavigationDepth: 2, maxRuntimeMs: 75_000, maxTokens: 16_000 },
  deep: { maxSearches: 10, maxPages: 16, maxNavigationDepth: 3, maxRuntimeMs: 180_000, maxTokens: 32_000 },
  exhaustive: { maxSearches: 20, maxPages: 30, maxNavigationDepth: 5, maxRuntimeMs: 420_000, maxTokens: 64_000 },
};

/* ------------------------------------------------------------------ */
/*  Research report                                                    */
/* ------------------------------------------------------------------ */

export interface ResearchReport {
  objective: string;
  executiveSummary: string;
  keyFindings: Array<{ finding: string; citations: number[] }>;
  comparison?: Array<{ label: string; values: Record<string, string>; citations: number[] }>;
  contradictions: Array<{ topic: string; sources: Array<{ source: Source; claim: string }> }>;
  uncertainty: string[];
  recommendations: string[];
  sources: Source[];
  durationMs: number;
  actionsTaken: number;
  truncated: boolean;
}

export interface SurfSnapshot {
  url: string;
  title: string;
  text?: string;
  takenAt: string;
  type: "page" | "document";
}
