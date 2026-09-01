/**
 * Relay Surf — research planner (server-only).
 *
 * Turns a user question into a `SurfPlan`: objective, candidate queries, the
 * kinds of sources to prefer, required information and verification
 * requirements. A deterministic heuristic produces a plan instantly (works
 * with zero configuration); an optional AI refinement pass improves plans for
 * complex questions when a provider is configured. Both paths are guarded so
 * planning can never fail a research run.
 */
import type { SurfPlan } from "../types";

type PlatformName = string;

const PLATFORM_TERMS: Array<{ platform: PlatformName; terms: string[] }> = [
  { platform: "reddit", terms: ["reddit", "subreddit", "r/"] },
  { platform: "x", terms: [" twitter", " x.com", "tweet"] },
  { platform: "instagram", terms: ["instagram", " insta "] },
  { platform: "youtube", terms: ["youtube", "video", "youtuber", " tutorial"] },
  { platform: "github", terms: ["github", "repository", "open-source", "open source", "npm", " library"] },
  { platform: "linkedin", terms: ["linkedin"] },
];

function detectPlatform(question: string): string | undefined {
  const q = ` ${question.toLowerCase()} `;
  return PLATFORM_TERMS.find(({ terms }) => terms.some((t) => q.includes(t)))?.platform;
}

function detectDepth(question: string): SurfPlan["depth"] {
  const q = question.toLowerCase();
  if (/exhaustive|everything|leave no stone/i.test(q)) return "exhaustive";
  if (/deep|thorough|comprehensive|detailed/i.test(q)) return "deep";
  if (/quick|brief|short answer/i.test(q)) return "quick";
  return "standard";
}

const SHOPPING_TERMS = /\b(best|top|cheap|affordable|worth|alternative|vs\.?|pricing|price|cost|compare)\b/i;

function heuristicPlan(question: string): SurfPlan {
  const platform = detectPlatform(question);
  const depth = detectDepth(question);
  const q = question.trim();
  const shopping = SHOPPING_TERMS.test(q);

  const queries = [q];
  if (platform && !q.toLowerCase().includes(platform)) {
    queries.push(`${q} ${platform}`);
  }
  if (/ vs\.?\b/i.test(q)) {
    const [a, b] = q.split(/\bvs\.?\b/i).map((s) => s.trim());
    if (a && b) {
      queries.push(`${a} ${b} comparison`);
      queries.push(`${a} pricing`);
      queries.push(`${b} pricing`);
    }
  }
  if (shopping) {
    queries.push(`${q} alternatives`);
    queries.push(`${q} reviews`);
  }
  if (queries.length < 3) {
    queries.push(`${q} documentation`);
    queries.push(`${q} how it works`);
  }

  const sources = platform
    ? [platform, "official documentation", "pricing pages", "community discussions"]
    : shopping
      ? ["official documentation", "pricing pages", "comparison articles", "community discussions"]
      : ["official documentation", "primary sources", "community discussions", "news"];

  return {
    objective: q,
    queries: queries.slice(0, 6),
    sources,
    requiredInformation: ["key facts", "current status / date", "exact numbers when relevant (pricing, metrics)"],
    verificationRequirements: [
      "prefer official sources for factual claims",
      "cross-check claims across at least two independent sources",
      "report disagreement instead of choosing a side",
    ],
    depth,
  };
}

/**
 * Ask the AI gateway to design a better plan for a complex question. Purely
 * an optimization — any failure falls back to the heuristic plan.
 */
async function aiPlan(question: string, heuristic: SurfPlan): Promise<SurfPlan | null> {
  try {
    const { getAIProvider } = await import("@/lib/ai/gateway.server");
    const provider = getAIProvider({ defaultModel: "google/gemini-2.5-flash" });
    const result = await provider.complete({
      system:
        "You design a web research plan for a research agent called Relay Surf. " +
        "Return ONLY a JSON object with exactly these keys: " +
        'objective (string), queries (array of 3-6 distinct search queries), ' +
        'sources (array of source-type strings), required_information (array of strings), ' +
        'verification_requirements (array of strings), depth (one of "quick"|"standard"|"deep"|"exhaustive"). ' +
        "No markdown, no commentary — raw JSON only.",
      messages: [{ role: "user", content: `Research question: ${question}` }],
      temperature: 0.2,
      maxTokens: 600,
      jsonMode: true,
    });
    const parsed = JSON.parse(result.text) as Partial<SurfPlan>;
    if (!Array.isArray(parsed.queries) || parsed.queries.length === 0) return null;
    return {
      objective: typeof parsed.objective === "string" && parsed.objective.trim() ? parsed.objective : heuristic.objective,
      queries: parsed.queries.map((q) => String(q)).slice(0, 8),
      sources: Array.isArray(parsed.sources) ? parsed.sources.map(String) : heuristic.sources,
      requiredInformation: Array.isArray(parsed.requiredInformation)
        ? parsed.requiredInformation.map(String)
        : heuristic.requiredInformation,
      verificationRequirements: Array.isArray(parsed.verificationRequirements)
        ? parsed.verificationRequirements.map(String)
        : heuristic.verificationRequirements,
      depth: parsed.depth === "quick" || parsed.depth === "deep" || parsed.depth === "exhaustive" || parsed.depth === "standard"
        ? parsed.depth
        : heuristic.depth,
    };
  } catch {
    return null;
  }
}

/**
 * Build a research plan for a question. Deterministic heuristic first, AI
 * refinement only when the question clearly benefits (complex phrasing or a
 * deep/exhaustive depth request).
 */
export async function planResearch(question: string, depth?: SurfPlan["depth"]): Promise<SurfPlan> {
  const q = question.trim();
  const heuristic = heuristicPlan(q);
  if (depth) heuristic.depth = depth;

  const isComplex =
    /\b(vs\.?|compare|alternatives|reviews|best|top|deep|thorough)\b/i.test(q) ||
    q.length > 140 ||
    heuristic.depth === "deep" ||
    heuristic.depth === "exhaustive";

  if (!isComplex) return heuristic;
  return (await aiPlan(q, heuristic)) ?? heuristic;
}

export { detectPlatform, detectDepth, heuristicPlan };
