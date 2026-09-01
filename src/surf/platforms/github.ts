/**
 * Relay Surf — GitHub adapter (server-only).
 *
 * Uses the public GitHub REST API (no token) to search repositories, issues,
 * PRs and discussions, and to read repository metadata + README. Public access
 * only; nothing authenticated is claimed. Unauthenticated search is
 * rate-limited by GitHub (~10 req/min) — the adapter shares the app's polite
 * HTTP layer so this stays within limits.
 */
import { relaiFetch } from "@/RelAI/web/http.server";
import { assertSafeUrl } from "../guard";
import type { PageData, PlatformAdapter, SearchResult } from "../types";

const API = "https://api.github.com";
const HEADERS = { Accept: "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28" };

interface GhItem {
  full_name?: string;
  html_url?: string;
  name?: string;
  description?: string;
  stargazers_count?: number;
  created_at?: string;
  updated_at?: string;
  owner?: { login?: string };
  title?: string;
  body?: string;
  number?: number;
}

async function ghGet(path: string): Promise<{ data?: unknown; ok: boolean; status: number }> {
  const res = await relaiFetch(`${API}${path}`, {
    headers: HEADERS,
    timeoutMs: 12_000,
    retries: 1,
    cache: true,
  });
  if (!res.ok) return { ok: false, status: res.status };
  try {
    return { data: JSON.parse(res.text), ok: true, status: res.status };
  } catch {
    return { ok: false, status: res.status };
  }
}

function mapRepo(item: GhItem): SearchResult {
  return {
    url: item.html_url ?? `https://github.com/${item.full_name}`,
    title: item.full_name ?? item.name ?? "",
    snippet: item.description ?? "",
    provider: "github",
    accessMode: "public",
    platform: "github",
    metadata: {
      stars: String(item.stargazers_count ?? 0),
      updated: item.updated_at ?? "",
    },
  };
}

export const githubAdapter: PlatformAdapter = {
  platform: "github",
  canHandle(url: string) {
    try {
      const host = new URL(url).hostname.replace(/^www\./, "");
      return host === "github.com";
    } catch {
      return false;
    }
  },
  async search(query, options = {}) {
    const limit = Math.min(options.limit ?? 8, 30);
    const results: SearchResult[] = [];

    // Repositories first — the primary GitHub search surface.
    const repo = await ghGet(
      `/search/repositories?q=${encodeURIComponent(query)}&per_page=${limit}`,
    );
    if (repo.ok && Array.isArray(repo.data)) {
      for (const item of (repo.data as { items?: GhItem[] }).items ?? []) {
        results.push(mapRepo(item));
        if (results.length >= limit) break;
      }
    }

    // Issues + PRs, capped well under the API rate budget.
    if (results.length < limit) {
      const issues = await ghGet(
        `/search/issues?q=${encodeURIComponent(`${query} type:issue`)}&per_page=${Math.min(limit - results.length, 10)}`,
      );
      if (issues.ok && Array.isArray(issues.data)) {
        for (const item of (issues.data as { items?: GhItem[] }).items ?? []) {
          if (results.length >= limit) break;
          results.push({
            url: item.html_url ?? "",
            title: `[${item.full_name}] ${item.title ?? ""}`,
            snippet: (item.body ?? "").slice(0, 300),
            provider: "github",
            accessMode: "public",
            platform: "github",
            metadata: { number: String(item.number ?? ""), owner: item.owner?.login ?? "" },
          });
        }
      }
    }

    return {
      results,
      accessMode: "public",
      note: results.length === 0 ? "GitHub public search returned no results (may be rate-limited)." : undefined,
    };
  },
  async extractPage(url) {
    const safeUrl = assertSafeUrl(url);
    const parsed = new URL(safeUrl);
    if (parsed.hostname !== "github.com") return null;

    // Repo README + metadata.
    const segments = parsed.pathname.split("/").filter(Boolean);
    if (segments.length >= 2) {
      const [owner, repo] = segments;
      const meta = await ghGet(`/repos/${owner}/${repo}`);
      if (meta.ok && (meta.data as { full_name?: string })?.full_name) {
        const m = meta.data as {
          full_name: string;
          description?: string;
          stargazers_count?: number;
          forks_count?: number;
          language?: string;
          default_branch?: string;
          html_url?: string;
        };
        const readme = await ghGet(`/repos/${owner}/${repo}/readme`);
        let readmeText = "";
        if (readme.ok) {
          const r = readme.data as { content?: string; encoding?: string };
          if (r.encoding === "base64" && r.content) {
            readmeText = Buffer.from(r.content, "base64").toString("utf-8").slice(0, 6000);
          }
        }
        const text = [
          `# ${m.full_name}`,
          m.description ? `${m.description}` : "",
          `Stars: ${m.stargazers_count ?? 0} · Forks: ${m.forks_count ?? 0} · Language: ${m.language ?? "unknown"}`,
          ``,
          readmeText ? `## README\n${readmeText}` : "## README\n(no public README found)",
        ]
          .filter(Boolean)
          .join("\n");

        const page: PageData = {
          url: m.html_url ?? safeUrl,
          title: m.full_name,
          text,
          links: [],
          headings: [
            { level: 1, text: m.full_name },
            { level: 2, text: "README" },
          ],
          metadata: {
            stars: String(m.stargazers_count ?? 0),
            forks: String(m.forks_count ?? 0),
            language: m.language ?? "",
            defaultBranch: m.default_branch ?? "",
          },
          truncated: readmeText.length >= 6000,
          fetchedAt: new Date().toISOString(),
          status: 200,
          contentType: "application/json",
          accessMode: "public",
        };
        return page;
      }
    }
    return null;
  },
};
