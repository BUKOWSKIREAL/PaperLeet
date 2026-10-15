import { parseSaved } from "./reports";
import { isPassed } from "./types";
import type { Problem, Report, Saved } from "./types";

export const WORKSPACE_KEY = "paperleet.offline.workspace.v1";
type StorageLike = Pick<Storage, "getItem" | "setItem">;
export type WorkspaceSnapshot = { records: Record<string, Saved>; error: boolean };

export function readWorkspace(storage: StorageLike): WorkspaceSnapshot {
  try {
    const raw = storage.getItem(WORKSPACE_KEY);
    if (raw === null) return { records: {}, error: false };
    const value: unknown = JSON.parse(raw);
    if (!value || typeof value !== "object" || Array.isArray(value)) throw Error("Invalid workspace");
    return { records: parseSaved(value), error: false };
  } catch {
    // Never automatically replace unreadable storage with an empty workspace.
    return { records: {}, error: true };
  }
}
export function writeWorkspace(storage: StorageLike, records: Record<string, Saved>): boolean {
  try {
    storage.setItem(WORKSPACE_KEY, JSON.stringify(records));
    return true;
  } catch { return false; }
}

export function withPracticeReport(saved: Saved, problem: Problem, report: Report): Saved {
  if (report.problem !== problem.id) return saved;
  return {
    ...saved,
    report,
    updated: new Date().toISOString(),
    acceptedReport: isPassed(problem, report) ? report
      : isPassed(problem, saved.acceptedReport) ? saved.acceptedReport
      : isPassed(problem, saved.report) ? saved.report : undefined,
  };
}
