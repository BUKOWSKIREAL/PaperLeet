# Core curriculum and toy experiments

Validated locally on 2026-10-13. The maintained catalog contains **24 papers, 26 modules, and six categories**. Existing paper IDs 1–9 and module IDs 1–10 remain stable.

## Selection and scope

The launch selection follows the proposal's strongest independently testable mechanisms. [TensorTonic Research](https://www.tensortonic.com/research) informed paper-to-module navigation; [TorchLeet](https://github.com/Exorust/TorchLeet) informed topic grouping; [TorchCode](https://github.com/duoan/TorchCode) informed immediate implementation feedback. These references informed organization, not copied problem solutions. The proposal's 136-paper pool remains a roadmap, not the launch inventory.

Each of the 16 added modules requires one to three core functions, with experiment scaffolding supplied separately. The exercise goal is selected core practice, not full benchmark reproduction. In particular, FlashAttention covers the online-softmax recurrence on CPU, QLoRA covers unpacked NF4 and low-rank adaptation, and GKD supplies the specific on-policy distillation formulation.

Toy provenance is explicit in bilingual statements and previews. JiT reconstructs the paper's embedding experiment at reduced scale; Drifting adapts the author's public notebook; DMD includes an analytic bimodal comparison inspired by its figure. Other teaching adaptations are labeled. Optional large sweeps are separate from graded CPU tests. DMD's Gaussian training test still independently exercises the trainable fake score.

## Numerical and infrastructure checks

- Full `paperleet ci`: **26 packages, 2,988 test executions, three repeats, 106/106 mutants caught**. There are 193 distinct reference checks per repeat. The clean run took **134.75 seconds**, with no reference, repeatability, package-quality, or per-test T3 budget failures.
- The total CI budget is now 300 seconds for the expanded catalog. An earlier contended run took 224.74 seconds; even the clean run exceeded the old 120-second budget. Individual T3 limits were retained.
- Framework regression before the artifact extension: **277 passed, 32 skipped**; skips cover opt-in Docker/deployment checks. The submission extension was separately exercised in real Docker, including reference repeats, starter rejection, assertion-killed mutants, published artifact delivery, and candidate-hash rejection after provenance changes.
- Final changed-area regression: **65 passed, 1 skipped** across submission and catalog tests; the skip is the separately exercised opt-in harness Docker case.
- Live hosted Docker validation covered all 26 modules and isolation checks: **63 passed**, with one old Flow Matching test failing only its wall-clock assertion (10.65 seconds against 10 seconds). All its numerical and distribution checks passed. Removed that load-dependent correctness assertion while retaining the 15-second execution timeout and separate CI performance budget. The affected module then passed **120 CI checks over three repeats**, catching all four mutants, and passed a final reference run in the rebuilt Docker worker. No numerical thresholds were relaxed.
- `npm run check` covers every maintained preview, generated toy report rendering without a static preview, chart data, completion semantics, private workspace isolation, and draft conflicts. The production TypeScript/Vite build passes.

## Browser checks

Desktop Chrome reproduced the Continue bug: the original header shrank from 63 px to 54 px, and the brand/nav moved left. After the fix, library and workspace header, brand, and navigation rectangles match. Continue uses the current paper's next required module, or the most recently edited unfinished module.

Checked category filtering (6 RL, 4 post-training, 10 generative, 2 architecture, 1 optimization, 1 representation), paper detail navigation, Chinese/English, dark/light themes, and a 390 px viewport without horizontal page overflow. A browser submission of LoRA passed all six checks and displayed its actual T3 regression curve in **My run results**. This used the separate localhost browser origin; no account was created and no user database was reset.

## Submit Paper boundary

Generated tests can request the backward-compatible `artifact` fixture. The worker serializes measured tensors/NumPy/JSON values with finite-value, shape-size, nesting, and byte limits; the host revalidates JSON without importing PyTorch. Toy provenance is bound to the candidate hash, and reference validation requires a nonempty correctly attributed T3 experiment. The review UI shows its artifacts before human approval.

No paid model generation was performed during this validation. Hosted online generation/review remains an offline maintainer workflow; these changes do not automatically publish model-generated papers or establish their scientific correctness.
