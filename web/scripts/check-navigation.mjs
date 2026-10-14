import assert from "node:assert/strict";
import fs from "node:fs";
import React from "react";
import { renderToString } from "react-dom/server";
import { createServer } from "vite";

const server = await createServer({ server: { middlewareMode: true }, appType: "custom" });
try {
  const { collectionHref, resolveAppLocation } = await server.ssrLoadModule("/src/navigation.ts");
  const { paperCollections } = await server.ssrLoadModule("/src/paperCollections.ts");
  const { PaperLibrary } = await server.ssrLoadModule("/src/PaperViews.tsx");
  const { default: CollectionDetail } = await server.ssrLoadModule("/src/CollectionDetail.tsx");
  const { default: Brand } = await server.ssrLoadModule("/src/Brand.tsx");
  const catalog = JSON.parse(fs.readFileSync("public/data/catalog.json", "utf8"));
  const origin = "https://paperleet.com";

  assert.equal(server.config.base, "/", "Nested collection reloads need root-relative assets and catalog requests");
  assert.equal(new URL(`${server.config.base}data/catalog.json`, `${origin}/collections/rl/`).pathname,
    "/data/catalog.json", "The catalog must not resolve beneath a collection path");

  assert.equal(resolveAppLocation({ pathname: "/", hash: "" }).route, "");
  for (const route of ["31", "papers/transformer", "account", "community", "community/123", "u/tester1", "submit"]) {
    assert.equal(resolveAppLocation({ pathname: "/", hash: `#/${route}` }).route, route,
      `${route}: existing root hash links must retain their destination`);
    const fromCollection = resolveAppLocation({ pathname: "/collections/rl/", hash: `#/${route}` });
    assert.equal(fromCollection.route, route, `${route}: hash navigation must leave the collection`);
    assert.equal(fromCollection.canonicalUrl, `/#/${route}`,
      `${route}: copied destinations must not retain an unrelated collection pathname`);
  }

  for (const collection of paperCollections) {
    const path = `/collections/${collection.id}/`;
    assert.equal(collectionHref(collection.id), path);
    assert.equal(resolveAppLocation({ pathname: path, hash: "" }).route, `collections/${collection.id}`,
      `${collection.id}: opening a collection URL directly must load its own page`);
    const withoutSlash = resolveAppLocation({ pathname: path.slice(0, -1), hash: "" });
    assert.equal(withoutSlash.route, `collections/${collection.id}`);
    assert.equal(withoutSlash.canonicalUrl, path, "Shared collection URLs should have one canonical form");
    for (const prefix of ["collections", "lists"]) {
      for (const suffix of ["", "/"]) {
        const legacy = resolveAppLocation({ pathname: "/", hash: `#/${prefix}/${collection.id}${suffix}` });
        assert.equal(legacy.route, `collections/${collection.id}`);
        assert.equal(legacy.canonicalUrl, path, "Previously shared links must open the real collection page");
      }
    }
  }
  const unknown = resolveAppLocation({ pathname: "/collections/not-a-collection/", hash: "" });
  assert.equal(unknown.route, "collections/not-a-collection", "Unknown collections must reach the explicit not-found view");
  for (const hash of ["#/%E0%A4%A", "#/collections/%E0%A4%A"])
    assert.doesNotThrow(() => resolveAppLocation({ pathname: "/", hash }), "Malformed legacy URLs must not crash the app");
  assert.doesNotThrow(() => resolveAppLocation({ pathname: "/collections/%E0%A4%A/", hash: "" }),
    "Malformed path encoding must not crash the app");

  const common = { papers: catalog.papers, problems: catalog.problems, saved: {}, progressReady: true };
  const library = renderToString(React.createElement(PaperLibrary, {
    ...common, categories: catalog.categories, onGuide() {},
  }));
  const cards = [...library.matchAll(/<a\b[^>]*class="collection-card [^"]+"[^>]*href="([^"]+)"/g)]
    .map((match) => match[1]);
  assert.deepEqual(cards, paperCollections.map((collection) => collectionHref(collection.id)),
    "Featured cards must navigate to distinct documents instead of homepage hash filters");

  for (const collection of paperCollections) {
    const html = renderToString(React.createElement(CollectionDetail, { ...common, collectionId: collection.id }));
    const hrefs = [...html.matchAll(/<a\b[^>]*href="([^"]+)"/g)].map((match) => match[1]);
    const base = `${origin}${collectionHref(collection.id)}`;
    assert(hrefs.includes("/"), "Every collection needs a working exit to the paper library");
    assert(hrefs.some((href) => /^\/#\/\d+$/.test(href)), "Collection modules must open the problem workspace");
    assert(hrefs.some((href) => href.startsWith("/#/papers/")), "Paper headings must open paper detail pages");
    for (const href of hrefs) {
      const destination = new URL(href, base);
      assert.equal(destination.pathname, "/", "Collection exit links must not inherit the collection pathname");
      assert(!resolveAppLocation(destination).route.startsWith("collections/"),
        "Following a collection exit must leave the collection route");
    }
  }
  const brand = renderToString(React.createElement(Brand));
  assert.match(brand, /href="\/"/, "The logo must return home from a clean collection URL");
  const missing = renderToString(React.createElement(CollectionDetail, { ...common, collectionId: "missing" }));
  assert(missing.includes("collection-not-found") && !missing.includes("collection-module-row"));
  assert(!/href="#/.test(missing), "Not-found navigation must also leave the nested pathname");
  console.log("PASS collection document routes, legacy redirects, deep-link assets, root exits and navigation recovery");
} finally {
  await server.close();
}
