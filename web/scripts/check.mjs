import assert from "node:assert/strict";
import fs from "node:fs";
import React from "react";
import { renderToString } from "react-dom/server";
import { createServer } from "vite";

const server = await createServer({
  server: { middlewareMode: true },
  appType: "custom",
});
try {
  const { default: Lab } = await server.ssrLoadModule("/src/Lab.tsx");
  const { ArtifactCharts } = await server.ssrLoadModule("/src/Visuals.tsx");
  const { isPassed, status } = await server.ssrLoadModule("/src/types.ts");
  const { parseReport, parseSaved, decodeProblemHash } =
    await server.ssrLoadModule("/src/reports.ts");
  const { decodeViz } = await server.ssrLoadModule("/src/vizData.ts");
  const { highlight } = await server.ssrLoadModule("/src/highlight.ts");
  const { dpoObjective, tokenGradient, difficultyAdvantage, scheduleData } =
    await server.ssrLoadModule("/src/concepts.ts");
  for (const smoothing of [0, 0.1, 0.5]) {
    for (const margin of [-10000, -2, 0, 2, 10000]) {
      const result = dpoObjective(margin, 0.7, smoothing);
      assert(Number.isFinite(result.loss) && Number.isFinite(result.gradient));
      const step = 1e-3;
      const numerical =
        (dpoObjective(margin + step, 0.7, smoothing).loss -
          dpoObjective(margin - step, 0.7, smoothing).loss) /
        (2 * step);
      assert(
        Math.abs(numerical - result.gradient) < 1e-6,
        "DPO gradient must match loss derivative",
      );
    }
  }
  assert(
    dpoObjective(10000, 1, 0.1).gradient > 0,
    "Smoothing pulls extreme positive margins back",
  );
  const lengths = [3, 6, 12, 24];
  assert.equal(tokenGradient("GRPO", 1, 3, lengths), -1 / 12);
  assert.equal(tokenGradient("GRPO", 1, 24, lengths), -1 / 96);
  for (const length of lengths) {
    assert.equal(tokenGradient("DAPO", 1, length, lengths), -1 / 45);
    assert.equal(tokenGradient("Dr. GRPO", 1, length, lengths), -1 / 96);
  }
  assert.equal(difficultyAdvantage(0.5, false).grpo, 1);
  assert.equal(difficultyAdvantage(0.5, true).grpo, Math.sqrt(7 / 8));
  assert.equal(difficultyAdvantage(0.25).dr, 0.75);
  for (const count of [50, 200, 1000])
    for (const kind of ["linear", "cosine"])
      for (const offset of [0, 0.008, 0.05]) {
        const schedule = scheduleData(kind, count, offset);
        assert.equal(schedule.betas.length, count);
        let cumulative = 1;
        schedule.betas.forEach((beta, index) => {
          assert(beta > 0 && beta < 1);
          cumulative *= 1 - beta;
          assert.equal(schedule.bars[index], cumulative);
        });
        assert(schedule.bars.every(Number.isFinite));
      }
  const catalog = JSON.parse(fs.readFileSync("public/data/catalog.json", "utf8"));
  const maintained = catalog.problems.filter((p) => !p.id.startsWith("community/"));
  for (const problem of maintained) {
    const file = problem.id.replaceAll("/", "-") + ".json";
    assert(fs.existsSync(`public/data/${file}`), `${file}: run paperleet vizdata all`);
    const data = JSON.parse(fs.readFileSync(`public/data/${file}`, "utf8"));
    const html = renderToString(React.createElement(Lab, {problem: {...problem, vizData: data}}));
    assert(!html.includes("NaN"), `${file} renders invalid numeric values`);
    assert(html.includes(data.kind === "toy_experiment" ? "参考实验" : "INTERACTIVE LAB"), `${file} has no visualization`);
    if (data.kind === "toy_experiment") {
      assert(html.includes("我的运行结果"), `${file} must distinguish submitted code from reference data`);
      assert(data.provenance, `${file} must explain toy provenance`);
      for (const chart of data.charts || []) {
        const heading = renderToString(React.createElement("h4", null, String(chart.title || "")));
        assert(html.includes(heading), `${file}: named chart ${chart.title} did not render`);
      }
    }
    console.log(`PASS ${file}`);
  }
  const publishedToy = renderToString(React.createElement(Lab, {
    problem: {id: "community/toy", title: "Reviewed toy", vizData: null},
    report: {problem: "community/toy", tests: [{id: "test_behavior", layer: "T3", passed: true,
      artifacts: {experiment: {kind: "toy_experiment", loss_curve: [[0, 1], [1, .1]], metrics: {error: .1}}}}]},
  }));
  assert(publishedToy.includes("test_behavior") && publishedToy.includes("loss_curve"));
  assert(publishedToy.includes("我的运行结果") && !publishedToy.includes(">参考实验</button>"),
    "Published toys without static previews must show actual judged artifacts, not an empty reference view");

  const packed = Buffer.alloc(8);
  [1000, -2000, 32767, -32768].forEach((value, i) =>
    packed.writeInt16LE(value, i * 2),
  );
  assert.deepEqual(
    decodeViz({
      encoding: "i16-base64",
      shape: [1, 2, 2],
      scale: 1000,
      data: packed.toString("base64"),
    }),
    [
      [
        [1, -2],
        [32.767, -32.768],
      ],
    ],
  );

  const report = {
    problem: "example",
    summary: { T0: [1, 1], T1: [1, 1], T2: [1, 1], T3: [0, 0] },
    tests: [
      {
        id: "test_contract",
        nodeid: "tests.py::test_contract",
        layer: "T0",
        passed: true,
      },
      {
        id: "test_reference",
        nodeid: "tests.py::test_reference",
        layer: "T1",
        passed: true,
      },
      {
        id: "test_property",
        nodeid: "tests.py::test_property",
        layer: "T2",
        passed: true,
      },
    ],
  };
  const problem = { id: "example", tests: { T0: 1, T1: 1, T2: 1, T3: 0 } };
  assert(isPassed(problem, report));
  assert(!isPassed(problem, { ...report, error: "collection failed" }));
  assert(!isPassed(problem, { ...report, summary: { T0: [1, 1] } }));
  assert(!isPassed(problem, { ...report, mutation: {} }));
  assert(!isPassed(problem, { ...report, problem: "another-problem" }));
  assert(!isPassed(problem, { ...report, tests: report.tests.slice(0, 1) }));
  assert(
    !isPassed(problem, {
      ...report,
      summary: { ...report.summary, T2: [9, 9] },
    }),
  );
  const partial = {
    ...report,
    tests: report.tests.slice(0, 1),
    summary: { T0: [1, 1] },
  };
  assert.doesNotThrow(() => parseReport(partial));
  assert(!isPassed(problem, partial));
  const parameterized = {
    ...report,
    tests: [
      ...report.tests,
      {
        id: "test_property",
        nodeid: "tests.py::test_property[other]",
        layer: "T2",
        passed: true,
      },
    ],
    summary: { ...report.summary, T2: [2, 2] },
  };
  assert(isPassed(problem, parameterized));
  assert(
    isPassed(problem, {
      ...parameterized,
      tests: parameterized.tests.map(({ nodeid, ...test }) => test),
    }),
  );
  assert(
    !isPassed(problem, {
      ...parameterized,
      tests: [...report.tests, report.tests[2]],
    }),
  );
  const badRows = [
    { duration_s: "bad" },
    { duration_s: -1 },
    { duration_s: Infinity },
    { duration_s: NaN },
    { why: {} },
    { message: [] },
    { id: null },
    { layer: "T99" },
    { passed: 1 },
    { nodeid: {} },
    { artifacts: [] },
  ];
  for (const bad of badRows)
    assert.throws(() =>
      parseReport({
        ...report,
        tests: [{ ...report.tests[0], ...bad }, ...report.tests.slice(1)],
      }),
    );
  for (const bad of [
    { mutation: { x: null } },
    { mutation: { x: [4] } },
    { impl: {} },
    { created_at: [] },
    { error: {} },
    { summary: { ...report.summary, T0: [0.5, 1] } },
    { summary: { ...report.summary, T0: [2, 1] } },
    { summary: { ...report.summary, T0: [Infinity, Infinity] } },
    { summary: { ...report.summary, T9: [0, 0] } },
  ])
    assert.throws(() => parseReport({ ...report, ...bad }));
  assert.equal(parseReport(report).tests[0].duration_s, 0);
  assert.doesNotThrow(() =>
    parseReport({ ...report, mutation: { example: ["test_contract"] } }),
  );
  const recovered = parseSaved({
    example: { code: "print(1)", report },
    corrupt: {
      code: "draft survives",
      report: { ...report, problem: "corrupt", mutation: { x: null } },
    },
    mismatch: { report },
    nullEntry: null,
    arrayEntry: [],
    badCode: { code: [] },
  });
  assert.deepEqual(Object.keys(recovered).sort(), ["corrupt", "example"]);
  assert.equal(recovered.corrupt.code, "draft survives");
  assert.equal(recovered.corrupt.report, undefined);
  assert.equal(status(problem, {...recovered.example, report: undefined}), "done",
    "Legacy success should survive a reset of the latest report");
  const acceptedThenFailed = parseSaved({example: {acceptedReport: report, report: {
    ...report, error: "Later run failed",
  }}});
  assert.equal(status(problem, acceptedThenFailed.example), "done", "A later failed attempt must not erase completion");
  const wrongAcceptance = parseSaved({other: {acceptedReport: report}});
  assert.equal(wrongAcceptance.other, undefined, "Completion must belong to the same module");
  assert.equal(status(problem, {acceptedReport: partial}), "todo", "A passing subset does not complete a module");
  const { paperProgress } = await server.ssrLoadModule("/src/PaperViews.tsx");
  const paper = {module_ids: ["example", "optional"]};
  const practice = {problems: [{...problem, module: {optional: false}},
    {...problem, id: "optional", module: {optional: true}}], saved: acceptedThenFailed};
  assert.deepEqual(paperProgress(paper, practice), {completed: 1, total: 1, state: "done"},
    "Optional extensions must not invalidate a paper's completed core modules");
  for (const invalid of [null, [], 4, "bad"])
    assert.deepEqual(parseSaved(invalid), {});
  assert.equal(decodeProblemHash("#/%E0%A4%A"), "");
  assert.equal(decodeProblemHash("#/%"), "");
  assert.equal(decodeProblemHash("#/rl%2Fgae"), "rl/gae");

  const output = renderToString(
    React.createElement(ArtifactCharts, {
      data: {
        samples: [[1, 2]],
        reference_samples: [[2, 3]],
        mmd: 0.03,
        threshold: 0.05,
        loss_curve: [
          [0, 5],
          [1, 4],
        ],
        training: [
          { step: 0, reward: 0 },
          { step: 1, reward: 1 },
        ],
      },
    }),
  );
  assert(output.includes("你的实现与参考对照"));
  assert(output.includes("60.0"));
  for (const data of [
    { curve: [{ step: 0, loss: 1 }, null] },
    {
      curve: [
        { step: 0, loss: 1 },
        { step: 1, loss: "bad" },
      ],
    },
    {
      curve: [
        { step: 0, loss: 1 },
        { step: 1, loss: Infinity },
      ],
    },
    { curve: [{}, {}] },
    { samples: [[1, 2], null] },
    { curve: [0, Infinity] },
    { mmd: Infinity, threshold: Infinity },
    { value: undefined },
  ]) {
    const fallback = renderToString(
      React.createElement(ArtifactCharts, { data }),
    );
    assert(!fallback.includes("NaN"));
    assert(
      fallback.includes("<pre>"),
      "Malformed artifact should remain readable as JSON",
    );
  }
  for (const lang of ["python", "bash", "json", "yaml", "latex", "text"])
    assert((await highlight("x = 1", lang)).includes("<pre"));
  console.log(
    "PASS packed trajectories, completion rules, report overlay, MMD threshold, syntax highlighting",
  );
} finally {
  await server.close();
}
