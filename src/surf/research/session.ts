/**
 * Relay Surf — research session manager (server-only).
 *
 * Sessions let Relay continue research instead of starting from zero, and give
 * the UI a live stream of human-readable actions (never raw logs). Sessions
 * are in-memory and bounded.
 */
import { EvidenceStore } from "../evidence/store";
import type { SurfAction, SurfPlan, SurfSession, SurfSessionStatus } from "../types";

interface InternalSession extends SurfSession {
  _store: EvidenceStore;
}

const sessions = new Map<string, InternalSession>();
let counter = 0;

export function createSurfSession(input: {
  userId: string;
  workspaceId?: string;
  objective: string;
  plan?: SurfPlan;
}): SurfSession {
  const id = `surf_${Date.now().toString(36)}_${(counter++).toString(36)}`;
  const session: InternalSession = {
    id,
    userId: input.userId,
    workspaceId: input.workspaceId,
    objective: input.objective,
    plan: input.plan,
    sources: [],
    evidence: [],
    actions: [],
    startedAt: new Date().toISOString(),
    status: "running",
    _store: new EvidenceStore(),
  };
  sessions.set(id, session);
  return publicSession(session);
}

export function getSurfSession(id: string): SurfSession | undefined {
  const session = sessions.get(id);
  return session ? publicSession(session) : undefined;
}

export function getSessionStore(id: string): EvidenceStore | undefined {
  return sessions.get(id)?._store;
}

export function updateSurfSession(id: string, patch: Partial<SurfSession>): void {
  const session = sessions.get(id);
  if (!session) return;
  Object.assign(session, patch);
}

export function recordAction(
  id: string,
  kind: SurfAction["kind"],
  description: string,
  opts: { provider?: string } = {},
): SurfAction {
  const session = sessions.get(id);
  const action: SurfAction = {
    id: `act_${session?.actions.length ?? 0}_${Date.now().toString(36)}`,
    kind,
    description,
    status: "running",
    provider: opts.provider,
    startedAt: new Date().toISOString(),
  };
  session?.actions.push(action);
  return action;
}

export function finishAction(action: SurfAction, status: "done" | "error", detail?: string): void {
  action.status = status;
  action.endedAt = new Date().toISOString();
  action.detail = detail;
  if (action.startedAt) action.ms = Date.now() - new Date(action.startedAt).getTime();
}

export function setSurfSessionStatus(id: string, status: SurfSessionStatus, error?: string): void {
  const session = sessions.get(id);
  if (!session) return;
  session.status = status;
  session.error = error;
  if (status === "complete" || status === "failed" || status === "cancelled") {
    session.completedAt = new Date().toISOString();
  }
}

export function listSurfSessions(userId: string, limit = 10): SurfSession[] {
  return [...sessions.values()]
    .filter((s) => s.userId === userId)
    .sort((a, b) => b.startedAt.localeCompare(a.startedAt))
    .slice(0, limit)
    .map(publicSession);
}

export function deleteSurfSession(id: string): void {
  sessions.delete(id);
}

export function syncEvidenceToSession(id: string): void {
  const session = sessions.get(id);
  if (!session) return;
  const store = session._store;
  session.evidence = store.all();
  session.sources = store.listSources();
}

function publicSession(session: InternalSession): SurfSession {
  return {
    id: session.id,
    userId: session.userId,
    workspaceId: session.workspaceId,
    objective: session.objective,
    plan: session.plan,
    sources: session.sources,
    evidence: session.evidence,
    actions: session.actions,
    startedAt: session.startedAt,
    completedAt: session.completedAt,
    status: session.status,
    error: session.error,
  };
}
