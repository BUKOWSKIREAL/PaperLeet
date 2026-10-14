import assert from "node:assert/strict";
import { createServer } from "vite";

const server = await createServer({
  server: { middlewareMode: true },
  appType: "custom",
});

try {
  const { eligibleUnattemptedProblems, pickRandomUnattemptedProblem } =
    await server.ssrLoadModule("/src/randomProblem.ts");
  const problem = (id) => ({ id, starter: "def solve(x):\n    raise NotImplementedError\n" });
  const report = (id, passed) => ({
    problem: id,
    summary: { T0: [Number(passed), 1] },
    tests: [{ id: "test_contract", layer: "T0", passed }],
  });
  const untouched = problem("untouched");
  const cached = problem("cached-starter");
  const metadata = problem("metadata-only");
  const attemptedIds = ["accepted", "failed", "error", "draft", "empty-draft", "reset-after-run"];
  const scope = [untouched, ...attemptedIds.map(problem), cached, metadata, untouched];
  const saved = {
    accepted: { acceptedReport: report("accepted", true) },
    failed: { report: report("failed", false) },
    error: { report: { problem: "error", error: "Execution timed out", summary: {}, tests: [] } },
    draft: { code: "def solve(x):\n    return x\n" },
    "empty-draft": { code: "" },
    "reset-after-run": { code: cached.starter, report: report("reset-after-run", false) },
    [cached.id]: { code: cached.starter, updated: "2026-10-13T00:00:00Z" },
    [metadata.id]: { updated: "2026-10-13T00:00:00Z" },
    "outside-the-collection": { code: "some draft" },
  };
  const before = structuredClone({ scope, saved });
  assert.deepEqual(eligibleUnattemptedProblems(scope, saved), [untouched, cached, metadata],
    "Skip every run or edited draft, but retain untouched cached starters; duplicate IDs get one chance");
  assert.deepEqual({ scope, saved }, before, "Selection must not modify drafts, reports, or its caller's scope");
  assert.equal(pickRandomUnattemptedProblem(scope, saved, () => 0), untouched);
  assert.equal(pickRandomUnattemptedProblem(scope, saved, () => 1 / 3 - Number.EPSILON), untouched);
  assert.equal(pickRandomUnattemptedProblem(scope, saved, () => 1 / 3), cached);
  assert.equal(pickRandomUnattemptedProblem(scope, saved, () => 2 / 3 - Number.EPSILON), cached);
  assert.equal(pickRandomUnattemptedProblem(scope, saved, () => 2 / 3), metadata);
  assert.equal(pickRandomUnattemptedProblem(scope, saved, () => 1 - Number.EPSILON), metadata,
    "The highest valid random value must select the final eligible module");

  const counts = new Map();
  for (let index = 0; index < 300; index += 1) {
    const selected = pickRandomUnattemptedProblem(scope, saved, () => (index + 0.5) / 300);
    counts.set(selected.id, (counts.get(selected.id) || 0) + 1);
  }
  assert.deepEqual([...counts.values()], [100, 100, 100], "Each unique eligible module gets an equal interval");
  assert.equal(pickRandomUnattemptedProblem([cached], saved, () => 0.99), cached,
    "A filtered scope must never draw from another collection or from saved records");
  let draws = 0;
  const random = () => { draws += 1; return 0.5; };
  assert.equal(pickRandomUnattemptedProblem([], saved, random), undefined);
  assert.equal(pickRandomUnattemptedProblem(attemptedIds.map(problem), saved, random), undefined);
  assert.equal(draws, 0, "Empty or fully attempted scopes should not draw or throw");
  assert.deepEqual(eligibleUnattemptedProblems([untouched, cached], {}), [untouched, cached]);
  console.log("PASS random unattempted modules: attempts, pristine starters, scoped uniform selection, deduplication, and empty states");
} finally {
  await server.close();
}
