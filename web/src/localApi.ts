/** Offline builds cannot send code to a hosted origin, even if accidentally published. */
export function isLocalOrigin(origin: string): boolean {
  try {
    const url = new URL(origin);
    return (url.protocol === "http:" || url.protocol === "https:") &&
      ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname) && !url.username && !url.password;
  } catch { return false; }
}
export class LocalRequestError extends Error {
  constructor(message: string, public status: number) { super(message); }
}
export async function localRequest(path: string, init: RequestInit = {}, fetcher: typeof fetch = fetch): Promise<unknown> {
  if (!isLocalOrigin(location.origin) || !/^\/api\/(?:health|workspace(?:\?problem=[^#]*)?|judge\/jobs(?:\/[a-f0-9]{32}(?:\/cancel)?)?)$/.test(path))
    throw new LocalRequestError("Open PaperLeet on localhost to use your computer’s Python service.", 403);
  const response = await fetcher(path, {
    ...init,
    credentials: "omit",
    cache: "no-store",
    redirect: "error",
    headers: { "Content-Type": "application/json", ...init.headers },
  });
  const value = await response.json();
  if (!response.ok) throw new LocalRequestError(
    typeof value?.error === "string" ? value.error : "The local service is unavailable.", response.status,
  );
  return value;
}
