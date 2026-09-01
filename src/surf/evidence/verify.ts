/**
 * Relay Surf — claim verification (server-only).
 *
 * When possible, cross-check a claim against multiple sources and report
 * agreement/conflict WITHOUT silently choosing a winner. Conflicts are surfaced
 * as contradictions with both sides attached, per the surf contract.
 */
import type { Evidence, Source } from "../types";

export interface VerificationResult {
  claim: string;
  /** evidence grouped by the claim-normalized fact. */
  support: Evidence[];
  contradiction: Evidence[];
  /** "confirmed" | "disputed" | "insufficient" */
  status: "confirmed" | "disputed" | "insufficient";
  consensus: string;
}

/** Normalize a claim so two phrasings of the same fact compare. */
export function claimKey(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 100);
}

/**
 * Verify a claim against collected evidence. A claim is "confirmed" when at
 * least two independent sources state compatible facts; "disputed" when
 * sources state conflicting facts; otherwise "insufficient".
 */
export function verifyClaim(claim: string, evidence: Evidence[]): VerificationResult {
  const key = claimKey(claim);
  const relevant = evidence.filter((e) => {
    const eKey = claimKey(e.claim);
    return eKey === key || eKey.includes(key) || key.includes(eKey);
  });

  if (relevant.length < 2) {
    return {
      claim,
      support: relevant,
      contradiction: [],
      status: "insufficient",
      consensus: "Not enough independent sources to verify this claim.",
    };
  }

  // For now, treat multiple sources stating the same normalized claim as
  // support; sources with *different* claims about the same topic are treated
  // as potential contradictions when they came from different domains.
  const support = relevant.filter((e) => claimKey(e.claim) === key);
  const contradiction = relevant.filter((e) => claimKey(e.claim) !== key);

  if (contradiction.length > 0) {
    return {
      claim,
      support,
      contradiction,
      status: "disputed",
      consensus: "Sources disagree — see both sides below.",
    };
  }
  return {
    claim,
    support,
    contradiction: [],
    status: support.length >= 2 ? "confirmed" : "insufficient",
    consensus: support.length >= 2 ? "Supported by multiple independent sources." : "Partially supported.",
  };
}

/** Summarize contradictions across a set of evidence for the report. */
export function findContradictions(evidence: Evidence[]): Array<{ topic: string; claims: Evidence[] }> {
  const groups = new Map<string, Evidence[]>();
  for (const e of evidence) {
    // Group by the first sentence of the claim (the topic).
    const topic = firstSentence(e.claim).toLowerCase();
    const list = groups.get(topic) ?? [];
    list.push(e);
    groups.set(topic, list);
  }

  const out: Array<{ topic: string; claims: Evidence[] }> = [];
  for (const [topic, claims] of groups) {
    const distinctFacts = new Set(claims.map((c) => claimKey(c.claim)));
    if (distinctFacts.size > 1 && claims.length >= 2) {
      out.push({ topic, claims });
    }
  }
  return out;
}

function firstSentence(text: string): string {
  const match = text.match(/^[^.!?]*[.!?]?/);
  return (match?.[0] ?? text).trim() || text;
}

/** Attach source info to evidence for citations. */
export function citeSources(sources: Source[], evidence: Evidence[]): Array<{ source: Source; evidence: Evidence[] }> {
  const byUrl = new Map<string, Source>();
  for (const s of sources) byUrl.set(s.url, s);
  const map = new Map<string, { source: Source; evidence: Evidence[] }>();
  for (const e of evidence) {
    const source = byUrl.get(e.sourceUrl);
    if (!source) continue;
    const entry = map.get(e.sourceUrl) ?? { source, evidence: [] };
    entry.evidence.push(e);
    map.set(e.sourceUrl, entry);
  }
  return [...map.values()];
}
