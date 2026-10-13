# PaperLeet

**English** | [简体中文](README.zh-CN.md)

Read papers, implement the methods, and test your understanding through mathematical properties and actual behavior.

The pilot includes **9 papers and 10 focused core modules**. Start with a paper, read its contribution and selected scope, then implement its core modules in a LeetCode-style split workspace. Supporting code is supplied so each exercise focuses on the paper’s key mechanism. Completing a paper means passing its selected core modules. Paper organization lives in `papers.yaml`; statements, implementations, tests, and visualization data live in `problems/`.

## Run locally

Requires Python 3.10+ and Node.js 20.19+. Run these commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
paperleet export
npm --prefix web ci
make dev
```

Open the URL shown in the terminal (default: **http://127.0.0.1:5173**). On subsequent runs, use **`make dev`** from the repository root to start both the website and the judge service. Press Ctrl-C to stop both processes.

## Solve your first problem

1. Choose a paper, such as **6 · DDPM**, then open its **Forward Noising** module.
2. Read the goal and complete only the required functions in the Python editor. Schedules and other supporting functions are already implemented.
3. Click **Run all tests** and wait for the test results to appear automatically.
4. Use each test layer's failure explanations, expected values, and actual values to revise your code, then run it again.

Passing the full test suite automatically marks the problem as complete and unlocks its notes on common pitfalls. Running only a subset of tests does not qualify as passing the full suite. Drafts and progress are saved in your current browser. Use the Chinese / English toggle in the upper-right corner to switch the interface, problem statements, and explanations. The paper library uses numeric IDs 1–9. Module IDs 1–10 and internal directory names stay stable for existing drafts and CLI commands. A later failed run does not erase a previously accepted result. Existing drafts are preserved; use the editor reset action after saving your code if you want the new starter with supplied helpers.

The website runs code through the judge service and displays results automatically, with no report files to manage manually. The local service listens on `127.0.0.1:8765`, and Vite proxies `/api` requests to it. Code is saved to `workspace/` and executed by Python/PyTorch on your machine. This setup is intended for personal local use; it does not yet provide an isolated code sandbox for multiple users.

## Community and accounts

Open **Community** to browse discussions, post and reply, like or repost, and follow other researchers. Public profiles show posts, replies, likes, reposts, followers, and following. Writing requires a verified email from an institution approved in `institutions.yaml`.

Display names and generated avatars are always public. Institution and country visibility are separate switches, both off by default; full email addresses stay private. Country is optional and entered by the user, with no IP lookup. Practice progress remains browser-local and is not synchronized to public profiles.

Email sign-in uses Supabase Auth and requires server configuration; there is no demo-login bypass. Copy `.env.example` to `.env`, configure it, and explicitly export it in your shell before starting the service. See the [community setup guide](docs/community.md) for OTP templates, SMTP, institution review, and privacy rules. The server remains a local development service; this does not enable safe public multi-user code execution.

## Submit paper

Click **Submit paper / 提交论文** in the top navigation and provide a paper URL, an optional official code URL, and what you want to practice. The model reads these sources and generates Chinese and English problem statements, starter code, a reference implementation, layered tests, and buggy variants. Candidates proceed to human review after passing isolated validation. Only the current version approved by a human reviewer receives a numeric problem ID and enters the paper library as a module. Export groups modules that cite the same arXiv paper or source URL under one paper. The generator currently creates one candidate module per submission.

Generation requires `OPENAI_API_KEY` and `PAPERLEET_MODEL` to be configured in the terminal used to start the service. Validation requires Docker to be running and the following image to be built from the repository root:

```bash
docker build -t paperleet-harness:local paperleet/submissions
```

You can still save submissions without a configured model or Docker; the website explains what is needed for generation or validation. Candidate materials and review records are stored in `.paperleet/submissions/`. The current reviewer is the local workspace owner; multi-user permissions and online deployment require separate integration. See the [harness design (in Chinese)](docs/paper-submission-harness.md) for the full workflow, isolation boundaries, and human review criteria.

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

`paperleet ci` checks that every reference solution passes, every mutant triggers at least one genuine test failure, each mutant changes at most 10 lines, results remain consistent across three runs, and the T3 CPU budget and total runtime meet the limits. Results are written to `reports/ci.json`, with full reference reports for each problem also saved under `reports/`. Import and syntax errors do not count as valid mutant detections.

## Papers and selected core modules

| Paper ID | Paper | Core modules | Module IDs |
| --- | --- | --- | --- |
| 1 | GAE | Backward advantage recurrence and episode boundaries | 1 |
| 2 | PPO | Clipped policy surrogate | 2 |
| 3 | DPO | Preference loss and implicit rewards | 3 |
| 4 | DeepSeekMath | GRPO group-relative advantages and policy objective | 4 |
| 5 | Dr. GRPO | Advantage scaling and fixed-budget normalization | 5 |
| 6 | DDPM | Forward noising; posterior and reverse step | 6, 8 |
| 7 | Progressive Distillation | v-parameterization and conversions | 7 |
| 8 | DDIM | Single-step sampling update | 9 |
| 9 | Flow Matching | Conditional path, target velocity, and regression loss | 10 |

Scope is selected per paper. The Progressive Distillation module covers its supporting v-parameterization technique; distillation training is not included. The cosine schedule accompanying DDPM is supplied related work from Improved DDPM. Full network assembly and large training runs are outside this pilot's required work.

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

`vizdata` uses the reference implementations, a fixed seed, and two PyTorch CPU threads. Each problem's standalone data file is limited to 500 KB. Concept diagrams that can be computed in the browser update immediately as you move their sliders. `npm run check` verifies chart rendering for all 10 problems, trajectory decoding, report charts, and completion status.

You can deploy `web/dist/` to static hosting such as GitHub Pages, but static pages cannot execute Python on their own. Running code online also requires deploying the judge service and forwarding `/api` requests to it. With static files alone, users can read problems, edit code, and download their code. The repository has no automatic publishing configured and no cloud judge service deployed.

## Repository layout and contributing

```text
paperleet/         CLI, pytest runner, local service, shared mathematical utilities
papers.yaml        Curated papers, contributions, scope, and module ordering
institutions.yaml  Reviewed institutional email domains
problems/          Self-contained core module packages (rl / diffusion)
web/               Vite + React + TypeScript
tests/             Judge framework integration tests
workspace/         Personal code, ignored by Git
reports/           Judge and CI reports, ignored by Git
.paperleet/        Local community database and paper submissions, ignored by Git
```

See the [contributing guide (in Chinese)](CONTRIBUTING.md) for the problem authoring workflow and the [10-problem pilot specification (in Chinese)](PaperLeet_第一步_10题试点规格.md) for detailed acceptance criteria. GitHub Actions runs framework tests, three rounds of quality checks, data generation, and the frontend build, then uploads reports and static site build artifacts.

See the [pilot validation record (in Chinese)](docs/validation/pilot.md) for measured results and the scope of validation for this implementation.
