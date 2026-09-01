/**
 * Relay Surf — server-only facade.
 *
 * Every surf implementation module (search, browse, platforms, research) pulls
 * server-only code (Playwright, relaiFetch, provider SDKs) into the module
 * graph. The client build stubs any `.server` specifier to a throw-on-use
 * placeholder, so server functions must lazy-load *this* module — never the
 * internals directly — to keep `node:`/.server code out of the browser bundle.
 *
 * The server (`server/rpc.ts`, `server/surf-routes.ts`) executes the real
 * modules, so server behavior is unaffected.
 */
export { routeSearch, runDeepSearch } from "./search/router";
export { searchPlatform } from "./platforms";
export {
  createBrowseSession,
  getBrowseSession,
  navigate,
  currentPage,
  browseEnabled,
} from "./browse/browser";
export { findOnPage } from "./browse/find-on-page";
export { startResearch, getResearch } from "./research/engine";
export { synthesizeReport } from "./research/report";
export { listSurfSessions } from "./research/session";
export { providerSummary, providerHealth } from "./search/providers";
