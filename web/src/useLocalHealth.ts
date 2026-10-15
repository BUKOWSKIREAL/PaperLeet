import { useEffect, useState } from "react";
import { localRequest } from "./localApi";

export function useLocalHealth() {
  const [online, setOnline] = useState(false);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    async function check() {
      try {
        const health = await localRequest("/api/health", { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(5000)]) }) as { mode?: string; judge?: { ready?: boolean } };
        if (!controller.signal.aborted) setOnline(health.mode === "local" && health.judge?.ready === true);
      } catch { if (!controller.signal.aborted) setOnline(false); }
      if (!controller.signal.aborted) timer = setTimeout(check, 15000);
    }
    void check();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [revision]);
  return { online, refresh: () => setRevision((value) => value + 1) };
}
