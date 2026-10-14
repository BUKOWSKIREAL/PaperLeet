import { collectionIdFromRoute } from "./paperCollections";
import { decodeProblemHash } from "./reports";

export function collectionHref(id: string): string {
  return `/collections/${encodeURIComponent(id)}/`;
}

/** Keep collection documents addressable while preserving shared hash links. */
export function resolveAppLocation({ pathname, hash }: { pathname: string; hash: string }): {
  route: string;
  canonicalUrl?: string;
} {
  if (hash) {
    const route = decodeProblemHash(hash);
    const collectionId = collectionIdFromRoute(route);
    if (collectionId !== undefined) {
      return { route: `collections/${collectionId}`, canonicalUrl: collectionHref(collectionId) };
    }
    // An old relative hash link opened from a collection still leads to the
    // root workspace, including the empty hash used by earlier home links.
    return { route, ...(pathname !== "/" ? { canonicalUrl: route ? `/#/${encodeURI(route)}` : "/" } : {}) };
  }
  const match = /^\/collections\/([^/]+)\/?$/.exec(pathname);
  if (match) {
    let id: string;
    try { id = decodeURIComponent(match[1]); }
    catch { return { route: "collections/invalid-url" }; }
    const canonicalUrl = collectionHref(id);
    return { route: `collections/${id}`, ...(pathname !== canonicalUrl ? { canonicalUrl } : {}) };
  }
  // A malformed collection URL must not silently display the home library.
  if (pathname.startsWith("/collections/")) return { route: "collections/invalid-url" };
  return { route: "" };
}

export function navigateToProblem(number: number): void {
  if (location.pathname === "/") location.hash = `/${number}`;
  else location.assign(`/#/${number}`);
}
