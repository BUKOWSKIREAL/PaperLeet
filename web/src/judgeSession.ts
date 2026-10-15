export type PendingRun = { id: string; problem: string };
export const PENDING_KEY = "paperleet.offline.judge.pending.v1";
export const CLIENT_KEY = "paperleet.offline.judge.client.v1";
type Session = Pick<Storage, "getItem" | "setItem" | "removeItem">;
export const runId = () => crypto.randomUUID().replaceAll("-", "");
export function clientId(storage: Session): string {
  try {
    const saved = storage.getItem(CLIENT_KEY);
    if (saved && /^[a-f0-9]{32}$/.test(saved)) return saved;
    const id = runId();
    storage.setItem(CLIENT_KEY, id);
    return id;
  } catch { return runId(); }
}
export function rememberRun(storage: Session, value?: PendingRun): void {
  try {
    if (value) storage.setItem(PENDING_KEY, JSON.stringify(value));
    else storage.removeItem(PENDING_KEY);
  } catch { /* The running page still polls; recovery needs session storage. */ }
}
export function recallRun(storage: Session): PendingRun | undefined {
  try {
    const value = JSON.parse(storage.getItem(PENDING_KEY) || "null");
    if (value && /^[a-f0-9]{32}$/.test(value.id) && typeof value.problem === "string" && value.problem)
      return { id: value.id, problem: value.problem };
  } catch { /* Ignore invalid or unavailable session storage. */ }
}
