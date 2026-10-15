import { useEffect, useRef, useState } from "react";
import { readWorkspace, withPracticeReport, writeWorkspace } from "./offlineStorage";
import type { Problem, Report, Saved } from "./types";

/** Offline drafts have no account or remote sync lifecycle. Every edit stays here. */
export function usePracticeWorkspace() {
  const [snapshot, setSnapshot] = useState(() => {
    try { return readWorkspace(localStorage); }
    catch { return { records: {}, error: true }; }
  });
  const latest = useRef(snapshot.records);
  function persist(records: Record<string, Saved>) {
    latest.current = records;
    let success = false;
    try { success = writeWorkspace(localStorage, records); } catch { /* Keep the editable in-memory copy. */ }
    setSnapshot({ records, error: !success });
  }
  useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (snapshot.error && Object.keys(latest.current).length) {
        event.preventDefault();
        event.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", beforeUnload);
    return () => window.removeEventListener("beforeunload", beforeUnload);
  }, [snapshot.error]);
  return {
    saved: snapshot.records,
    getSaved: (id: string): Saved => latest.current[id] || {},
    ready: true,
    loading: false,
    error: snapshot.error,
    update(id: string, change: Partial<Saved>) {
      persist({ ...latest.current, [id]: { ...latest.current[id], ...change, updated: new Date().toISOString() } });
    },
    applyReport(problem: Problem, report: Report) {
      // A completed job never replaces edits made while that job was running.
      persist({ ...latest.current, [problem.id]: withPracticeReport(latest.current[problem.id] || {}, problem, report) });
    },
    retry() { persist(latest.current); },
  };
}
