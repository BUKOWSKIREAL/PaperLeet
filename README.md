# PaperLeet

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="web/public/brand/paperleet-logo-dark.svg">
  <img src="web/public/brand/paperleet-logo-light.svg" alt="PaperLeet" width="280">
</picture>

**English** | [简体中文](README.zh-CN.md) · [Try the beta](https://paperleet.com) · [Quick start](#quick-start) · [Deployment](#deployment)

## Project activity

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/metrics/stars-dark.svg">
  <img src="docs/metrics/stars.svg" alt="GitHub star history for PaperLeet" width="840">
</picture>

<details>
<summary>Account growth &amp; weekly commits</summary>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/metrics/users-dark.svg">
  <img src="docs/metrics/users.svg" alt="Cumulative created accounts, including unused beta accounts" width="840">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/metrics/commits-dark.svg">
  <img src="docs/metrics/commits.svg" alt="Non-merge commits per UTC week, excluding generated metrics updates" width="840">
</picture>

Account totals include unused beta accounts; they are not active-user counts. Only anonymous dates and counts are published.

</details>

Updated daily. [Metric definitions and refresh details](docs/metrics.md).

## The idea

AI moves fast, and starting out can feel overwhelming. PaperLeet helps people **entering research or preparing for ML and research jobs** learn one core idea at a time: choose a paper, implement its key modules, and check your understanding through tests and small experiments.

Our goal is a **decentralized, open-source research community**. Anyone can contribute papers, discuss ideas, improve exercises, or fork and self-host the project for their own group. The curriculum grows through community contributions and human review.

The beta includes **46 papers and 60 coding exercises**, an English/Chinese editor with automated judging, personal progress and daily activity, and community posts, replies and follows. Online practice currently uses administrator-issued test accounts.

## Quick start

Requires Python 3.10+ and Node.js 20.19+.

```bash
git clone https://github.com/Hongyang-Du/PaperLeet.git
cd PaperLeet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
npm --prefix web ci
make dev
```

Open **http://127.0.0.1:5173**, choose a paper, complete a module, and click **Run all tests**. Use `make dev` to start again; Ctrl-C stops both services. Local practice runs Python on your computer.

## Deployment

Use the Docker deployment for a shared, invitation-only beta:

| Where | Setup |
| --- | --- |
| Linux server and domain | [Server deployment](docs/hosting.md): HTTPS, isolated judging, persistent storage and backups. |
| Your own computer and domain | [Cloudflare Tunnel](docs/cloudflare-tunnel.md): Docker hosts the beta; keep the computer awake and online. |

Start with [test accounts](docs/test-accounts.md) for username/access-code sign-in without email registration. Accounts, posts and practice data stay in the deployment's private database volume; keep backups outside GitHub. The guides cover configuration and startup for each option.

## Paper library

Explore LLM Basics, RL, Post-training, Diffusion & Flow, JEPA / World Models, and Visual Generation. Each paper contains focused core exercises, supplied helper code, and tests; selected modules include CPU toy experiments.

<details>
<summary>Browse all 46 papers and 60 core modules</summary>

| Paper ID | Paper | Core scope | Module IDs |
| --- | --- | --- | --- |
| 1 | GAE | Implement the backward GAE recurrence and episode boundary handling. | 1 |
| 2 | PPO | Implement the clipped surrogate and its gradient behavior; value loss, normalization, and diagnostics are supplied. | 2 |
| 3 | DPO | Implement the preference loss and implicit rewards; token alignment and sequence log-probability calculation are supplied. | 3 |
| 4 | DeepSeekMath · GRPO | Focus on the paper's GRPO module: group-relative advantages, clipped objectives, and response-level aggregation. | 4 |
| 5 | Dr. GRPO | Implement two corrections: remove group standard-deviation scaling and normalize by a fixed generation budget. | 5 |
| 6 | DDPM | Implement closed-form forward noising, the Gaussian posterior, and an epsilon-parameterized reverse step; schedules and sampling loops are supplied. | 6, 8 |
| 7 | Progressive Distillation | Implement v-parameterization conversions, then derive and validate the two-teacher-step to one-student-step distillation target. | 7, 26 |
| 8 | DDIM | Implement the DDIM update and clean endpoint; timestep selection, loops, and inversion exploration code are supplied. | 9 |
| 9 | Flow Matching | Implement the conditional path, target velocity, and CFM regression loss; time sampling, Euler integration, and training helpers are supplied. | 10 |
| 10 | GSPO | Implement sequence ratios and the GSPO loss; test masking, clipping regions, and gradients. | 11 |
| 11 | GKD · On-policy Distillation | Implement student rollouts and full-vocabulary divergence in a tiny autoregressive model. OPD here means on-policy distillation. | 12 |
| 12 | DAPO | Implement group filtering and the token-level objective; sampling infrastructure and distributed training are outside this module. | 13 |
| 13 | LoRA | Implement the low-rank update and linear forward pass; verify frozen parameters and a small regression experiment. | 14 |
| 14 | QLoRA | Implement blockwise NF4 quantization, dequantization, and adapter forward; double quantization and paged optimizers are outside this exercise. | 15 |
| 15 | MeanFlow | Implement the average-flow JVP target and gradient boundaries, then run a small generative experiment. | 16 |
| 16 | JiT | Implement data-to-velocity conversion and the denoising objective with a paper-derived embedded-manifold experiment. | 17 |
| 17 | Drifting Models | Implement the drifting field and stopped-gradient loss, and verify generator behavior on a small two-dimensional distribution. | 18 |
| 18 | DMD | Implement the score-difference target and paired regression term; inspect mode coverage using an analytic bimodal distribution. | 19 |
| 19 | DMD2 | Implement two-timescale updates and auxiliary objectives; verify parameter-update boundaries in a small experiment. | 20 |
| 20 | Consistency Models | Implement the boundary parameterization and adjacent-time consistency-distillation loss. | 21 |
| 21 | Transformer | Implement scaled dot-product attention, multi-head attention, LayerNorm and Add & Norm, sinusoidal positions, and position-wise FFN as five independent modules. | 22, 27–30 |
| 22 | AdamW | Implement moment estimates, bias correction, and a decoupled weight-decay step. | 23 |
| 23 | FlashAttention | Implement blockwise recurrence and rescaling. This CPU algorithm exercise does not benchmark GPU kernels. | 24 |
| 24 | CLIP | Implement feature normalization, temperature scaling, and bidirectional contrastive loss with a tiny paired-retrieval experiment. | 25 |
| 25 | GLM-5 | Implement head-wise Muon Split and double-sided importance corrections for asynchronous RL. | 31–32 |
| 26 | Kimi K3 | Implement bounded KDA recurrence, Block Attention Residuals, and SiTU-GLU; recreate the report’s gate and activation curves. | 33–35 |
| 27 | OPSD | Dual-context causal alignment and pointwise divergence clipping. | 36, 37 |
| 28 | SDPO | Same-question feedback routing, reverse KL, and top-k tail-mass coarsening. | 38, 39 |
| 29 | REPA | Patch-wise representation alignment and gradient boundaries. | 40 |
| 30 | Rotation Trick | Rotation-based gradient transformation with quantized forward values. | 41 |
| 31 | Differential Transformer | Differential attention on projected tensors and lambda parameterization. | 42 |
| 32 | Block Diffusion | Single-forward attention masks for concatenated noisy and clean tokens. | 43 |
| 33 | DistiLLM-2 | Skew KL on teacher trajectories and skew reverse KL on student trajectories. | 44 |
| 34 | PKPO | Continuous-reward transformation, policy loss, and the paper’s one-dimensional Gaussian toy. | 45 |
| 35 | SVDQuant | Smoothing, SVD low-rank decomposition, and a 4-bit residual forward pass. | 46 |
| 36 | MoGe | Robust optimal estimation with truncated weighted L1 residuals. | 47 |
| 37 | DAPS | Linear-Gaussian posterior score, Langevin updates, and independent renoising. | 48 |
| 38 | DiffusionNFT | Positive/negative beta implicit velocities, the two-branch objective, and tiny training. | 49 |
| 39 | VAR | Next-scale conditioning and attention masks; multiscale residual quantization and reconstruction. | 50–51 |
| 40 | FD-Loss | EMA population moments, differentiable Fréchet distance, and stopped-gradient loss normalization. | 52–53 |
| 41 | VICReg | Variance, view invariance, and off-diagonal covariance penalties. | 54 |
| 42 | LeJEPA · SIGReg | Characteristic-function Gaussianity statistic and sliced aggregation. | 55 |
| 43 | VISReg | Scale, sliced shape, centering, and multi-view invariance. | 56 |
| 44 | I-JEPA | Masked teacher targets, prediction loss, and EMA updates. | 57 |
| 45 | LeWorldModel | Joint prediction objective and latent rollout; terminal cost and CEM elite refitting. | 58–59 |
| 46 | Causal-JEPA | Object masking with identity anchors and masked history/future prediction. | 60 |

</details>

## Contributing

Bring a paper, improve an explanation or test, or help with the UI, accounts, community and deployment. Issues and PRs are welcome. New exercises require human review before entering the library.

[Contribution guide](CONTRIBUTING.md) · [User and maintainer guide](docs/guide.md) · [Paper submission harness](docs/paper-submission-harness.md)

Thanks to [TorchCode](https://github.com/duoan/TorchCode) for inspiration on presenting hands-on ML practice.
