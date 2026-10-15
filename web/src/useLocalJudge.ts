import { useEffect, useRef, useState } from "react";
import { JobRequestError, requestJob, watchJob } from "./judgeJobs";
import type { JudgeJob } from "./judgeJobs";
import type { Report } from "./types";

import { clientId, recallRun, rememberRun, runId } from "./judgeSession";
import type { PendingRun } from "./judgeSession";
type View = { pending?: PendingRun; job?: JudgeJob; reconnecting: boolean; error: string };
function session(): Pick<Storage, "getItem" | "setItem" | "removeItem"> {
  try { return sessionStorage; }
  catch { return { getItem: () => null, setItem() {}, removeItem() {} }; }
}
const empty = (): View => ({ reconnecting: false, error: "" });

export function useLocalJudge(onReport: (report: Report) => void, ready: boolean) {
  const [client] = useState(() => clientId(session()));
  const [view, setView] = useState<View>(() => empty());
  const controller = useRef<AbortController | undefined>(undefined);
  const callback = useRef(onReport);
  callback.current = onReport;
  const visible = view;

  async function attach(pending: PendingRun, signal: AbortSignal) {
    try {
      const job = await watchJob(pending.id, client, signal,
        (job) => {
          if (job.problem !== pending.problem) throw new JobRequestError("The judge returned a different module", 422);
          if (!signal.aborted) setView({ pending, job, reconnecting: false, error: "" });
        },
        (connected) => { if (!signal.aborted) setView((previous) => ({ ...previous, reconnecting: !connected })); });
      if (signal.aborted) return;
      rememberRun(session());
      setView({ job, reconnecting: false, error: job.status === "failed" ? job.error || "Judge job failed" : "" });
      if (job.status === "succeeded" && job.report) callback.current(job.report);
    } catch (error) {
      if (signal.aborted) return;
      rememberRun(session());
      setView({ reconnecting: false, error: error instanceof Error ? error.message : "Judge job unavailable" });
    }
  }

  useEffect(() => {
    controller.current?.abort();
    setView(empty());
    if (!ready) return;
    const pending = recallRun(session());
    if (pending) {
      const next = new AbortController();
      controller.current = next;
      setView({ ...empty(), pending });
      void attach(pending, next.signal);
    }
    return () => controller.current?.abort();
  }, [ready]);

  async function submit(problem: string, code: string) {
    if (!ready || visible.pending) return;
    const pending = { id: runId(), problem };
    const next = new AbortController();
    controller.current?.abort();
    controller.current = next;
    rememberRun(session(), pending);
    setView({ ...empty(), pending });
    try {
      const job = await requestJob("", client, next.signal, { ...pending, code });
      if (next.signal.aborted) return;
      if (job.id !== pending.id || job.problem !== pending.problem) throw new JobRequestError("The judge returned a different submission", 502);
      setView({ ...empty(), pending, job });
      void attach(pending, next.signal);
    } catch (error) {
      if (next.signal.aborted) return;
      if (!(error instanceof JobRequestError) || error.status >= 500) {
        // Creation may have succeeded before the reply was lost. Query its
        // stable ID rather than submit the same source a second time.
        setView({ ...empty(), pending, reconnecting: true });
        void attach(pending, next.signal);
      } else {
        rememberRun(session());
        setView({ ...empty(), error: error.message });
      }
    }
  }

  async function cancel() {
    if (!visible.pending || visible.job?.status !== "queued") return;
    try {
      await requestJob(`/${visible.pending.id}/cancel`, client, controller.current!.signal,
        {});
    } catch (error) {
      if (controller.current?.signal.aborted) return;
      setView((previous) => ({ ...previous, error: error instanceof Error ? error.message : "Could not cancel this job" }));
    }
  }
  return { ...visible, active: !!visible.pending, submit, cancel };
}
