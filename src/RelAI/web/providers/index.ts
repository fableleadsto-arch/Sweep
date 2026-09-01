/**
 * RelAI — Web Search Providers Index.
 *
 * Central export point for all search engine providers.
 * Adding a new search engine: create the provider file and add it here.
 */
export {
  tavilySearch,
  tavilyExtract,
  tavilyConfigured,
} from "./tavily.server";
export type {
  TavilySearchOptions,
  TavilySearchResult,
} from "./tavily.server";

export {
  exaSearch,
  exaAnswer,
  exaGetContents,
  exaConfigured,
} from "./exa.server";
export type {
  ExaSearchOptions,
  ExaSearchResult,
} from "./exa.server";

export {
  jinaSearch,
  jinaReadUrl,
  jinaConfigured,
} from "./jina.server";

export {
  searxngSearch,
  searxngConfigured,
} from "./searxng.server";
export type {
  SearXNGOptions,
} from "./searxng.server";
