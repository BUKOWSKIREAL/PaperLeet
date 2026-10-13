# VAR and FD-Loss curriculum expansion

Two paper cards add four core exercises, bringing the library to **40 papers and 53 modules**. Existing paper/module numbers and draft keys remain stable.

| Paper | Module | Selected implementation |
| --- | --- | --- |
| 39 · VAR | 50 · Next-scale prediction | Lower-scale cumulative conditioning and scale-causal attention mask |
| 39 · VAR | 51 · Multiscale residual tokenizer | Quantize residuals from coarse to fine; reconstruct from projected code maps |
| 40 · FD-Loss | 52 · Population moments | Current-batch moments, detached-history EMA, and covariance recovery |
| 40 · FD-Loss | 53 · Differentiable objective | Gaussian Fréchet distance and stop-gradient-normalized representation losses |

Each exercise contains Chinese/English statements and notes, supplied scaffolding, a starter and reference implementation, T0–T3 tests, four local mutants, and a CPU toy that executes the learner's implementation. Both papers are listed under Generative Models. VAR is identified as visual autoregression; FD-Loss is a distributional post-training objective.

## Sources and scope

### VAR

Primary sources are [the paper, §3.2 and Algorithms 1–2](https://arxiv.org/html/2404.02905v2), [the official tokenizer](https://github.com/FoundationVision/VAR/blob/main/models/quant.py), and [the official autoregressive model](https://github.com/FoundationVision/VAR/blob/main/models/var.py). Package metadata records the versioned source used for implementation.

Tokens within one scale are predicted in parallel. Their conditioning features come from the cumulative reconstruction of earlier scales, so the attention mask permits same-scale conditioning positions while blocking future scales. This differs from directly feeding the current target token embeddings. The tests perturb target/future token maps to detect leakage.

The tokenizer downsamples the current residual, selects nearest shared-codebook vectors, lifts and projects them at full resolution, and subtracts that contribution before moving to the next scale. Interpolation and projection helpers are supplied. A fixed channel projection substitutes for the author's learned spatial residual convolution; the exercise states that simplification explicitly. The tokenizer is frozen, and these modules do not train a VQ-VAE or reproduce ImageNet generation quality.

### FD-Loss

Primary sources are [Representation Fréchet Loss for Visual Generation](https://arxiv.org/html/2604.28190v1), §3.2, Eqs. (2)–(6), Algorithm 1, and Appendix B.3, plus [the official implementation](https://github.com/Jiawei-Yang/FD-Loss/tree/5c03b8112fec8b9432631e4ce053c0d918cc24bc).

The selected statistics module implements the EMA route. It blends raw first and second moments, then obtains covariance as `M - mu mu.T`; averaging centered batch covariances would lose between-batch mean variation. History is detached, while current features remain differentiable. EMA moments use a population denominator; the separate official queue implementation's sample-covariance convention is documented but is not mixed into this exercise.

The objective uses the symmetric reference-root sandwich to evaluate the covariance cross term with `eigvalsh`. Reference statistics are frozen. Positive-semidefinite inputs are covered for forward values; gradient guarantees apply to positive-definite covariance inputs, including repeated positive eigenvalues. The square root is not silently regularized to redefine the distance at singular inputs.

The multi-representation objective divides each raw distance by its own stopped-gradient value plus epsilon and sums weighted terms. Gradient checks must hold this denominator fixed: ordinary finite differences of the whole normalized forward expression would test a different derivative. Raw FD gradients are checked separately using an independent two-dimensional closed form.

## Experiments

The new toys use small synthetic features, residual reconstructions, and low-dimensional distributions. They are explicitly **teaching adaptations**. They do not reproduce pretrained image encoders, full autoregressive image generation, large-population distributed training, or the papers' image-quality benchmarks. The FD training visualization reports the raw distance as well as the optimized objective so a near-unit normalized loss is not mistaken for lack of progress.

Reference previews and artifacts from the learner's own T3 run remain separate. The deployment worker explicitly includes the four new packages; no account, post, or learner data is added to version control.

## Validation (2026-10-13)

The four new reference packages contain **34 checks and 16 local mutants**. All unfinished starters fail genuine checks. Independent review covered the equations, official code conventions, independent test oracles, and both language versions.

Measured reference toys:

| Module | Result |
| --- | --- |
| 50 | Current/future target influence `0`; minimum earlier-scale influence `0.5050` |
| 51 | Reconstruction MSE `0.05015` after the coarse scale → `0.0011848` after all scales; encode/decode error `0` |
| 52 | Average batch error `30` vs EMA error `0.01599` under the documented warm start; current-gradient error `0` |
| 53 | Independent current-generator FD `9.2661` → `0.0073302` over 200 steps; final mean error `0.04963`, covariance error `0.14498` |

- Full curriculum CI: **53 modules, 402 reference checks per run, 214/214 mutants caught, 6,123 test executions over three consistent runs**, completed in `169.52s` under the `300s` budget. All T3 limits and mutant quality checks passed.
- Framework suite: `301 passed, 59 skipped` in `86.14s`; optional infrastructure checks are skipped by default.
- The judge image was rebuilt. All four new reference submissions passed in the real Docker sandbox (`4 passed` in `23.81s`).
- All 53 visualization datasets passed frontend checks, along with existing practice/account checks; the production TypeScript/Vite build passed.
- Catalog export confirmed 40 papers, 53 unique modules, complete bilingual content, and agreement with the hosted allowlist. The existing 38 paper cards and 49 module payloads were preserved exactly.
- Browser checks verified VAR/FD-Loss search results, paper-to-module navigation, English/light and Chinese/dark views, mathematical rendering, and the reference experiment curves. Existing draft code and saved results were not replaced.

These checks exercise the maintained packages and worker deployment. They do not claim that an arbitrary new paper can be published without source review and human approval.
