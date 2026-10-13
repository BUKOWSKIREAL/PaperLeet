# Oral / Spotlight curriculum expansion

Implemented on 2026-10-13: **12 papers and 14 modules**, bringing the curated library to **38 papers and 49 modules**. Existing paper/module IDs and draft keys remain unchanged. This is a repository and deployment-package update; it does not provision a public server.

| Paper ID | Paper | Modules | Implemented mechanism |
| --- | --- | --- | --- |
| 27 | OPSD | 36–37 | Dual-context causal alignment; signed pointwise divergence clipping |
| 28 | SDPO | 38–39 | Success/feedback routing; reverse KL; student top-k plus tail mass |
| 29 | REPA | 40 | Patch-wise cosine alignment and combined denoising objective |
| 30 | Rotation Trick | 41 | Frozen rotation/scale surrogate for vector quantization |
| 31 | Differential Transformer | 42 | Two separately normalized attention maps and learnable lambda |
| 32 | Block Diffusion | 43 | Vectorized noisy/clean attention mask and attention computation |
| 33 | DistiLLM-2 | 44 | Teacher-response SKL and student-response SRKL |
| 34 | PKPO | 45 | Expected-maximum estimator, continuous-reward transform, policy loss |
| 35 | SVDQuant | 46 | Smoothing, high-precision low-rank branch, quantized residual branch |
| 36 | MoGe | 47 | Truncated weighted-L1 ROE with shared scale and z-only shift |
| 37 | DAPS | 48 | Linear-Gaussian posterior score, Langevin update, independent renoising |
| 38 | DiffusionNFT | 49 | Positive/negative implicit velocity fields and Eq. (5) objective |

Every new package includes Chinese/English statements and notes, a focused starter, reference implementation, T0–T3 tests, four local mutants, and an experiment that calls the submitted functions. The worker allowlist and explicit Docker COPY list include all 49 curated modules. No user/account/community data is included in the worker image or Git changes.

## Source and scope review

The [35-paper evidence pool](../research/paper-candidates-2026-10-13.md) preserves official Oral/Spotlight program links, original papers, candidate modules, and harness notes. Ten of those candidates are implemented here, alongside OPSD and SDPO from the prior shortlist. Remaining candidates have no live problem entries. Individual `problem.yaml` files pin source versions and anchors; presentation status is not represented as a best-paper award.

Independent review checked the following distinctions:

- OPSD rescoring keeps the same student trajectory despite different teacher/student prompt lengths. Pointwise clipping retains negative vocabulary contributions and clips before vocabulary summation.
- SDPO follows Table 2: an already successful attempt supplies its own solution; otherwise use a successful attempt on the same question, with rich feedback as fallback. Selecting the first successful peer is an explicit deterministic exercise convention. Top-k is selected by the student and the complement remains one probability bin.
- Rotation Trick tests a prescribed surrogate Jacobian, not finite differences of the quantized forward map. Zero and antiparallel vectors are outside the stated exercise domain.
- Differential Transformer excludes head normalization and output projection. Both softmax branches receive the causal mask.
- DistiLLM-2 preserves the distinct skew-mixture directions and Eq. (2)'s `1/(2B)` normalization. The alpha=1 endpoint remains differentiable with zero student gradient.
- PKPO is checked against subset enumeration and nonbinary rewards; an additional exact discrete-policy gradient review agreed within 4.1e-16.
- SVDQuant smooths input and weight in inverse directions; the low-rank branch uses the unquantized smoothed input. Its supplied INT fake quantizer is an explicit teaching choice, not an FP4 GPU kernel.
- MoGe uses truncated weighted L1, not L2 Procrustes. Tests compare the tiny exact solver with an independent two-dimensional line-arrangement oracle.
- DAPS distinguishes the exact Gaussian posterior covariance from the finite-step-size Langevin covariance. DiffusionNFT implements the base Eq. (5) objective and freezes rollout quantities, without claiming the complete production training recipe.

## Toy provenance

PKPO recomputes the paper's Gaussian reward experiment with a bounded Monte Carlo budget. Block Diffusion recreates the Figure 3 attention mask and additionally measures forbidden-token influence. Rotation Trick adapts the paper's quadratic/16-codebook geometry with deterministic directions.

Other new experiments are explicitly marked **teaching adaptations**: small categorical models, synthetic patch features, quantization matrices, robust point alignment, or linear-Gaussian inverse problems. They require no model download or external API. No toy metric is presented as a reproduction of a full-model benchmark. Reference plots and artifacts produced by the learner's test run remain separate in the web UI.

## Browser review

Chrome on the running local app showed 38 paper cards and 49 modules. Searching OPSD returned paper 27 with modules 36 and 37. Its module workspace displayed the bilingual statement, starter, test action, and clipping experiment. English/Chinese switching and light/dark rendering were checked; the initial Chinese/dark preference was restored. No learner draft or result was replaced during this review.


## Validation results

- Final full catalog CI: **49 modules, 5,613 test executions, three consistent runs, 154.89 seconds**, within the 300-second catalog budget. All **368 reference checks** pass per run and all **198 mutants** are detected. The batch adds 110 reference checks and 56 mutants; all new mutants remain within the ten-line change limit.
- Framework regression: **299 passed, 56 skipped**. The subsequent real-catalog export regression and catalog suite: **23 passed**. This regression catches mismatches between public function metadata and supplied helpers before export. Optional live deployment/mail tests remain outside this run.
- Rebuilt `paperleet-worker:local`; all **14 new reference packages passed real Docker judging**, 53.51 seconds total. This validates the final worker contents, report protocol, numerical checks, and T3 artifacts under the pinned container dependencies.
- All 14 unfinished starters fail their required tests; supplying helper functions does not make a blank implementation accepted.
- `npm --prefix web run check`: all **49 experiment render checks**, private-workspace checks, and account UI checks passed. Production TypeScript/Vite build passed.
- Export: **38 paper cards / 49 modules**, complete English/Chinese materials and toy data for all 14 new modules. Numeric paper/module IDs remain distinct and stable.
- `git diff --check` passed. Full machine-readable CI details remain in ignored `reports/ci.json`; the checked-in data contains only curated reference visualizations and curriculum content.
