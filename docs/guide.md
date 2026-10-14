# PaperLeet guide

**English** | [简体中文](guide.zh-CN.md) · [Back to README](../README.md)

Detailed reference for practicing, contributing modules, and maintaining an instance. For setup, start with [Quick start](../README.md#quick-start), the [hosting guide](hosting.md), or [Cloudflare Tunnel on your computer](cloudflare-tunnel.md). Run the commands below from the repository root unless shown otherwise.

The CLI is `paperleet`; the earlier `leetpaper` alias remains supported.

## Solve your first problem

1. Choose a paper, such as **6 · DDPM**, then open its **Forward Noising** module.
2. Read the goal and complete only the required functions in the Python editor. Schedules and other supporting functions are already implemented.
3. Click **Run all tests** and wait for the test results to appear automatically.
4. Use each test layer's failure explanations, expected values, and actual values to revise your code, then run it again.

Passing the full test suite automatically marks the problem as complete and unlocks its notes on common pitfalls. Running only a subset of tests does not qualify as passing the full suite. In local mode, drafts and progress are saved in your current browser. In hosted mode, signed-in members sync drafts and judging results to their account. Use the Chinese / English toggle in the upper-right corner to switch the interface, problem statements, and explanations. The paper library uses numeric IDs 1–46. Module IDs 1–60 and internal directory names stay stable for existing drafts and CLI commands. A later failed run does not erase a previously accepted result. Existing drafts are preserved; use the editor reset action after saving your code if you want the new starter with supplied helpers.

Use **Light / Dark** in the top bar to choose an appearance. The first visit follows your system preference; an explicit choice is remembered in this browser. The paper library puts categories, search, filters, and module progress together; paper details contain the technical summary and expandable practice scope.

Drag the workspace dividers to resize the description/code columns and the code/results rows. Sizes are remembered in this browser. Double-click a divider or press Enter while it is focused to restore its default; arrow keys fine-tune the split. Run controls stay visible, and narrow screens stack the description above the editor.

The **Featured collections** carousel groups papers into LLM Basics, RL, Diffusion & Flow, Post-training, JEPA / World Models, and Visual Generation. Each colorful card navigates to a separate page such as `/collections/llm-basics/`, with its own title, cover, progress, and problems grouped by paper in curated order. These URLs support direct visits, refreshes, bookmarks, and browser back navigation. Open a problem directly, continue practice, or filter within the collection. The homepage's compact category chips filter the full paper library. Collections can overlap; existing `#/collections/...` and `#/lists/...` links automatically resolve to the new page addresses. Membership and bilingual card text live in `web/src/paperCollections.ts`.

**Random problem** opens an unattempted module within the current collection and filters. It excludes any saved run, accepted result, or edited draft; an unchanged starter is still eligible. The button waits for practice progress to load and is disabled when no eligible problems remain.

Hosted problem pages show public aggregate statistics: unique participants and solvers, total submissions, accepted and failed submissions, and acceptance rate (accepted / total submissions). Each account counts once per problem for participant and solver totals; repeated full submissions count separately. Partial test runs and judge-service failures are excluded. Statistics include existing server-judged history and persist with the database; local development runs do not contribute. No submissions yet means an unset acceptance rate, not a failed submission.

The website runs code through the judge service and displays results automatically, with no report files to manage manually. The local service listens on `127.0.0.1:8765`, and Vite proxies `/api` requests to it. Code is saved to `workspace/` and executed by Python/PyTorch on your machine. This local setup is intended for personal use. Use the separate hosted configuration for isolated multi-user judging; do not expose the development service directly.

## Community and accounts

Open **Community** to browse discussions, post and reply, like or repost, and follow other researchers. Public profiles show posts, replies, likes, reposts, followers, and following. In institutional mode, writing requires a verified email from an institution approved in `institutions.yaml`; the hosted beta additionally requires an administrator invitation. In test mode, members sign in with an administrator-issued username and access code. Their profiles are labeled **Test account** and do not claim institutional email verification.

Profiles include unique solved and attempted problem totals, difficulty breakdowns, a yearly activity calendar, active days and streaks, and recent accepted problems. The calendar uses the viewer's browser time zone and separates submission counts, distinct problems attempted or passed that day, and first-ever passes. Only server-judged full submissions count; partial tests and service failures do not. Repeated passes never increase the cumulative solved count. Community statistics include posts, replies, likes and reposts received, and follow relationships.

Display names and generated avatars are always public. Verified institution and country visibility are separate switches, both off by default; test accounts have no institution field, and full email addresses stay private. Country is optional and entered by the user, with no IP lookup. Turn off **Allow learning progress to be public** in account settings to restrict personal learning statistics to yourself. Code, drafts, and detailed reports always stay private. Local practice stays in the browser and is not uploaded to profiles. Per-problem statistics remain anonymous aggregate counts.

Institutional email sign-in uses Supabase Auth and requires server configuration. Test account sign-in uses separately provisioned accounts and does not use the email provider. Copy `.env.example` to `.env`, configure it, and explicitly export it in your shell before starting the service. See the [community setup guide](community.md) for OTP templates, SMTP, institution review, and privacy rules. For an online beta, follow the [hosting guide](hosting.md), which uses a separate isolated judge and invitation controls.

## Submit paper

In the **hosted beta**, signed-in members can submit paper proposals and read their own submissions; administrators can read all proposals. Generation, validation, and publication are performed by maintainers through the offline workflow. Hosted judging is limited to the 60 curated modules included in the deployment image.

In **local maintainer mode**, click **Submit paper / 提交论文** in the top navigation and provide a paper URL, an optional official code URL, and what you want to practice. The model reads these sources and generates Chinese and English problem statements, starter code, a reference implementation, layered tests, and buggy variants. Candidates proceed to human review after passing isolated validation. Only the current version approved by a human reviewer receives a numeric problem ID and enters the paper library as a module. Export groups modules that cite the same arXiv paper or source URL under one paper. The generator currently creates one candidate module per submission. It looks for author toys and paper figures, labels teaching adaptations explicitly, and generates T3 tests that record measured curves and metrics through a bounded `artifact` fixture. Human reviewers can inspect those results before approval.

Generation requires `OPENAI_API_KEY` and `PAPERLEET_MODEL` to be configured in the terminal used to start the service. Validation requires Docker to be running and the following image to be built from the repository root:

```bash
docker build -t paperleet-harness:local paperleet/submissions
```

You can still save submissions without a configured model or Docker; the website explains what is needed for generation or validation. Candidate materials and review records are stored in `.paperleet/submissions/`. The local workspace owner performs reviews. Hosted members do not receive generation, validation, or publication controls. See the [harness design (in Chinese)](paper-submission-harness.md) for the full workflow, isolation boundaries, and human review criteria.

## For maintainers: CLI and quality checks

Use the CLI to maintain problems, debug the judge, or check reference solutions:

```bash
paperleet list
paperleet init diffusion/param_conversion
# Edit workspace/diffusion/param_conversion.py
paperleet test diffusion/param_conversion
paperleet test diffusion/param_conversion --layers T0,T2 --report reports/my-result.json
paperleet test rl/gae --impl problems/rl/gae/solution.py
paperleet mutate diffusion/param_conversion
paperleet ci
```

`init` does not overwrite existing code by default. Use `--force` when you intend to reset it.

`paperleet ci` checks that every reference solution passes, every mutant triggers at least one genuine test failure, each mutant changes at most 10 lines, results remain consistent across three runs, and the per-test T3 CPU budget and total runtime meet the limits. The CLI defaults to a 300-second total budget (override with `--budget`); GitHub Actions uses `paperleet ci --budget 600` for hosted-runner variance. Individual T3 limits stay in each package. Results are written to `reports/ci.json`, with full reference reports for each problem also saved under `reports/`. Import and syntax errors do not count as valid mutant detections.

## Module scope

See the [paper catalog](../README.md#paper-library) for the complete list.

The homepage groups papers into seven categories. New modules include CPU toy experiments, source/adaptation notes, and T0–T3 checks. Lab distinguishes reference plots from your submitted code’s T3 results. This is selected core practice; full model training and GPU kernel benchmarking remain outside scope.

Each module declares `module.required_functions` and `module.provided_functions`. The starter leaves only required functions unfinished, and the judge exercises them through the supplied supporting code. All modules include bilingual statements and notes, reference implementations, layered tests, and buggy variants. Module IDs 6 and 7 do not require T3 tests; module 5's length trend is illustrative and is not graded.

## Visualizations and static builds

The repository includes pre-generated data, so you can start it directly. Re-export after changing problems or experiments:

```bash
paperleet vizdata all
paperleet export
cd web
npm run check
npm run build
```

`vizdata` uses the reference implementations, a fixed seed, and two PyTorch CPU threads. Each problem's standalone data file is limited to 500 KB. Concept diagrams that can be computed in the browser update immediately as you move their sliders. `npm run check` verifies chart rendering for all maintained modules, trajectory decoding, report charts, completion status, private workspace isolation, draft conflicts, and accepted-result preservation.

You can deploy `web/dist/` at a domain's root as a static preview for reading papers and viewing visualizations. Configure a fallback to `/index.html` for `/collections/*` so direct visits and refreshes work; the included Caddy configurations already provide it. Account workspaces, editing, synchronization, and judging require the configured `/api` service. Use the [hosting package](hosting.md) for an interactive beta; no public service has been deployed by this repository automatically.

## Repository layout and contributing

```text
paperleet/         CLI, pytest runner, local service, shared mathematical utilities
papers.yaml        Curated papers, contributions, scope, and module ordering
institutions.yaml  Reviewed institutional email domains
problems/          Self-contained core module packages (RL, fine-tuning, generative models, architecture, optimization, representation)
web/               Vite + React + TypeScript
tests/             Judge framework integration tests
workspace/         Personal code, ignored by Git
reports/           Judge and CI reports, ignored by Git
.paperleet/        Local community database and paper submissions, ignored by Git
```

See the [contributing guide (in Chinese)](../CONTRIBUTING.md) for the problem authoring workflow and the [10-problem pilot specification (in Chinese)](../PaperLeet_第一步_10题试点规格.md) for detailed acceptance criteria. GitHub Actions runs framework tests, three rounds of quality checks, data generation, and the frontend build, then uploads reports and static site build artifacts.

See the [pilot validation record (in Chinese)](validation/pilot.md) for measured results and the scope of validation for this implementation.

See the [expanded curriculum validation record](validation/core-curriculum.md) for the 26-paper release, toy provenance, browser checks, and harness validation.

The Oral/Spotlight expansion adds 12 papers and 14 modules. See the [candidate evidence](research/paper-candidates-2026-10-13.md) and [validation record](validation/oral-spotlight-expansion.md) for sources, toy provenance, and checks.

VAR and FD-Loss add four focused modules with CPU experiments. See the [source and validation record](validation/var-and-fd-loss.md).

## Public project metrics

Definitions and refresh instructions for the README’s stars, cumulative created accounts, and weekly commits are in [Project metrics](metrics.md). Account counts include unused beta accounts; only anonymous date/count aggregates are published.
