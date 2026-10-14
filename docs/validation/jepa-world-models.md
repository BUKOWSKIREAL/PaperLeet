# JEPA and world-model curriculum expansion

Six paper cards add seven exercises, bringing the maintained library to **46 papers and 60 modules**. Existing paper numbers, module numbers, and draft keys remain stable. The new **JEPA / World Models** study list follows VICReg → I-JEPA → LeJEPA → VISReg → LeWorldModel → Causal-JEPA. It spans Representation Learning and the new World Models category.

| Paper | Module | Selected core |
| --- | --- | --- |
| 41 · VICReg | 54 | Variance, view invariance, and covariance penalties |
| 42 · LeJEPA | 55 | SIGReg characteristic-function statistic and sliced aggregation |
| 43 · VISReg | 56 | Scale, sliced shape, centering, and multi-view invariance |
| 44 · I-JEPA | 57 | Masked teacher targets, prediction loss, and EMA |
| 45 · LeWorldModel | 58 | Joint prediction objective and action-conditioned latent rollout |
| 45 · LeWorldModel | 59 | Terminal goal cost and CEM elite refitting |
| 46 · Causal-JEPA | 60 | Object trajectory masking and history/future prediction losses |

Each package contains bilingual statements and notes, a starter, a reference solution, T0–T3 tests, four local mutants, and a CPU experiment that calls the submitted functions. SIGReg belongs to the LeJEPA paper card. LeJEPA and LeWorldModel are distinct papers.

## Sources and numerical conventions

- **VICReg:** [paper](https://arxiv.org/abs/2105.04906) and [official implementation at `4e12602`](https://github.com/facebookresearch/vicreg/blob/4e12602fd495af83efd1631fbe82523e6db092e0/main_vicreg.py). Use sample variance/covariance with denominator `B-1`; average the two variance penalties but sum the two covariance penalties, with off-diagonal squares divided by feature dimension. Both views receive gradients.
- **LeJEPA / SIGReg:** [paper](https://arxiv.org/abs/2511.08544) and [official implementation at `c293d29`](https://github.com/galilai-group/lejepa/tree/c293d291ca87cd4fddee9d3fffe4e914c7272052). Evaluate real and imaginary empirical characteristic-function errors, the Gaussian window, sample-count scaling, and twice the positive-grid trapezoidal integral. Supplied fixed unit projections make grading deterministic. Exact point collapse may be stationary; the exercise does not claim that every collapsed state automatically escapes.
- **VISReg:** [paper v1, §3.1 and Algorithm 1](https://arxiv.org/html/2606.02572v1) and [author project](https://haiyuwu.github.io/visreg/). The author repository redirects to an unavailable implementation at verification time, so Algorithm 1 is the implementation authority. The exercise uses population standard deviation, a squared scale penalty, stopped-gradient standard deviation in the shape normalization, sorted unit projections, and Gaussian quantiles `i/(N+1)`. It explicitly adds the epsilon from Eq. (2), omitted in the pseudocode, for finite collapsed inputs. Global-view means remain differentiable.
- **I-JEPA:** [paper](https://arxiv.org/abs/2301.08243) and [official training code](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/src/train.py); the package pins the inspected revision. Targets use feature-wise LayerNorm, target-block gathering, context repetition, and stop-gradient. The exercise follows the released code's mean smooth-L1 loss with beta 1, documenting its difference from the paper's L2 description. Teacher EMA is detached.
- **LeWorldModel:** [paper](https://arxiv.org/abs/2603.19312), [author project](https://le-wm.github.io/), and [official implementation at `8edfeb3`](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac). Joint mean-element prediction MSE and supplied SIGReg preserve both encoder gradient paths. The rollout is an explicitly restricted one-context-state model. Planning implements terminal squared L2 cost and population variance of elite candidates for a diagonal Gaussian CEM refit; the sampling loop is supplied.
- **Causal-JEPA:** [paper](https://arxiv.org/abs/2602.11389) and [official implementation at `412337d`](https://github.com/galilai-group/cjepa/tree/412337d0210bf98cee2ca90c3586ab2ea7ca519e). Initial-frame identity anchors stay visible; selected object histories and all future slots become queries. Masked history and future errors receive separate mean reductions, with frozen targets. The exercise excludes initial-frame anchors from the history loss, following the masked-token indicator in paper v2 Eq. (5); the released training loop instead includes time zero for selected history slots. Statements and notes distinguish these conventions. This is an object-masking inductive bias, not a guarantee of causal identification.

## Experiments and scope

The toys are labeled **teaching adaptations**, with source anchors and simplifications in each package. They exercise representation statistics, distribution shape beyond matched moments, masked teacher prediction, action-conditioned dynamics, CEM goal seeking, and reconstruction of a hidden interacting object. They do not claim ImageNet pretraining, pixel-based world-model reproduction, robotic benchmark results, or causal identification. Reference previews remain separate from artifacts produced by the learner's own run.

The deployment image explicitly includes all seven new modules. User accounts, posts, drafts, and results remain outside version control.

## Validation (2026-10-13)

The seven new packages contain **60 reference checks and 28 local mutants**. All seven unfinished starters fail genuine checks without test setup errors. Independent review covered primary sources, numerical conventions, gradient boundaries, bilingual statements, and experiment scope.

Measured reference experiments:

| Module | Result |
| --- | --- |
| 54 · VICReg | Loss `4.7175` → `0.01224`; minimum feature std `1.0777` |
| 55 · SIGReg | Loss `27.9894` → `0.15205`; full-grid oracle error `0`; Gaussian-quantile statistic `0.00473` versus balanced binary statistic `6.5811` |
| 56 · VISReg | Objective `0.84483` → `0.03081`; mean feature std `0.15` → `0.80867` |
| 57 · I-JEPA | Prediction loss `0.46396` → `6.53e-16`; EMA closed-form error `0` |
| 58 · LeWM prediction | Prediction MSE `0.17805` → `4.83e-7`; training-trajectory rollout MSE `2.30e-6` |
| 59 · LeWM planning | Mean-plan terminal cost `8.5` → `0.00025287` |
| 60 · Causal-JEPA | Prediction loss `27.7588` → `0.0012203`; hidden-history leakage `0` |

Catalog export preserves the existing **40 paper cards and 53 module payloads exactly**, with complete bilingual materials for the new exercises. All 60 visualization datasets and frontend checks pass, and the production TypeScript/Vite build succeeds. Browser checks cover list membership, SIGReg search, paper-to-module navigation, equations, reference charts, English/light and Chinese/dark views, and a narrow viewport without horizontal page overflow.

Full curriculum CI passes: **60 modules, 462 reference checks per run, 242/242 mutants caught, and 7,023 test executions across three consistent runs**, in `193.03s` under the `300s` budget. All T3 time limits and mutant quality checks pass.

The judge image was rebuilt, and all seven new reference submissions passed inside the real Docker sandbox (`7 passed` in `32.09s`).

The framework suite passes with `301 passed, 66 skipped` in `81.27s`; optional infrastructure checks are skipped by default. Existing browser draft code and accepted results were not replaced during UI verification.
