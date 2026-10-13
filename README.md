# PaperLeet

**English** | [简体中文](README.zh-CN.md)

Read papers, implement the methods, and test your understanding through mathematical properties and actual behavior.

The first pilot includes **5 RL problems and 5 diffusion problems**. The website uses a LeetCode-style problem list and split workspace: problem statements, paper explanations, and interactive labs on the left; a Python editor and T0–T3 test reports on the right. All problem statements, function signatures, reference solutions, buggy variants, and visualization data come from `problems/`.

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

1. Choose a problem, such as **7 · x₀ / ε / v Parameterization**.
2. Read the statement on the left and complete the implementation in the Python editor on the right.
3. Click **Run all tests** and wait for the test results to appear automatically.
4. Use each test layer's failure explanations, expected values, and actual values to revise your code, then run it again.

Passing the full test suite automatically marks the problem as complete and unlocks its notes on common pitfalls. Running only a subset of tests does not qualify as passing the full suite. Drafts and progress are saved in your current browser. Use the Chinese / English toggle in the upper-right corner to switch the interface, problem statements, and explanations. The pilot problems have fixed numbers 1–10; internal directory names remain unchanged for compatibility with existing drafts and commands.

The website runs code through the judge service and displays results automatically, with no report files to manage manually. The local service listens on `127.0.0.1:8765`, and Vite proxies `/api` requests to it. Code is saved to `workspace/` and executed by Python/PyTorch on your machine. This setup is intended for personal local use; it does not yet provide an isolated code sandbox for multiple users.

## Submit paper

Click **Submit paper / 提交论文** in the top navigation and provide a paper URL, an optional official code URL, and what you want to practice. The model reads these sources and generates Chinese and English problem statements, starter code, a reference implementation, layered tests, and buggy variants. Candidates proceed to human review after passing isolated validation. Only the current version approved by a human reviewer receives a numeric problem ID and enters the problem bank.

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

## Problems

| ID | Problem | Difficulty | Focus |
| --- | --- | --- | --- |
| 1 | Generalized Advantage Estimation | Medium | Backward recurrence, episode boundaries, and bias–variance tradeoffs |
| 2 | PPO Clipped Surrogate | Medium | Gradients in all four quadrants, value clipping, and trust regions |
| 3 | Direct Preference Optimization | Easy | Label shifting, masking, and numerically stable preference loss |
| 4 | Group Relative Policy Optimization | Medium | Within-group normalization, per-token KL, and aggregation |
| 5 | Dr. GRPO | Medium | Length bias and reward normalization variants |
| 6 | Noise Schedules & Forward Diffusion | Easy | Linear/cosine schedules and forward noising |
| 7 | x₀ / ε / v Parameterization | Easy | Round-trip conversions, tangent vectors, and endpoint stability |
| 8 | DDPM Posterior & Sampling | Medium | Independent Gaussian posterior derivation and analytical oracle sampling |
| 9 | DDIM | Medium | η, timestep skipping, deterministic sampling, and inversion |
| 10 | Flow Matching | Medium | Interpolation, velocity, Euler integration, and training a small model |

Every problem includes `problem.yaml`, paper explanations in Chinese and English, starter code, a reference solution, notes on common pitfalls unlocked after passing, layered tests, and buggy variants. Problems 6 and 7 do not require T3 tests. The length trend in problem 5 is shown for illustration and is not used for grading.

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
problems/          Self-contained problem packages (rl / diffusion)
web/               Vite + React + TypeScript
tests/             Judge framework integration tests
workspace/         Personal code, ignored by Git
reports/           Judge and CI reports, ignored by Git
```

See the [contributing guide (in Chinese)](CONTRIBUTING.md) for the problem authoring workflow and the [10-problem pilot specification (in Chinese)](PaperLeet_第一步_10题试点规格.md) for detailed acceptance criteria. GitHub Actions runs framework tests, three rounds of quality checks, data generation, and the frontend build, then uploads reports and static site build artifacts.

See the [pilot validation record (in Chinese)](docs/validation/pilot.md) for measured results and the scope of validation for this implementation.
