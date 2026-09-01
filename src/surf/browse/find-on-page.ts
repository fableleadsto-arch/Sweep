/**
 * Relay Surf — find on page (server-only).
 *
 * Implements `find_on_page` for long documentation/article pages: locate a
 * keyword or section in the already-loaded page and return the surrounding
 * content plus a jump target, without re-fetching.
 */
import { extractAroundMatch } from "./extract";
import type { PageData, PageSection } from "../types";

export interface FindOnPageResult {
  url: string;
  query: string;
  matches: PageSection[];
  /** Best single section — the one the caller should read first. */
  best?: PageSection;
  /** Total occurrences in the cleaned text. */
  count: number;
}

export function findOnPage(page: PageData, query: string): FindOnPageResult {
  const needle = query.trim();
  const text = page.text ?? "";
  const lower = text.toLowerCase();
  const count = lower.split(needle.toLowerCase()).length - 1;

  const sections = extractAroundMatch(text, needle, 1600).map((s) => ({
    heading: s.heading,
    headingLevel: page.headings.find((h) => h.text === s.heading)?.level,
    headingId: page.headings.find((h) => h.text === s.heading)?.id,
    text: s.section,
    start: s.start,
    end: s.end,
    url: page.url,
  }));

  return {
    url: page.url,
    query,
    matches: sections,
    best: sections[0],
    count,
  };
}
