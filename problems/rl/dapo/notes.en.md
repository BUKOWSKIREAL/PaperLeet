# Implementation notes — DAPO Dynamic Filtering and Token Objective

`dynamic_group_mask` receives binary outcome rewards `[prompts,group]` with group size at least 2. Return a boolean `[prompts]` vector retaining groups with at least one success and one failure. Both `0/1` and `-1/1` reward encodings are supported. Selecting mixed groups is the core filtering primitive; a production sampler would replenish the batch after filtering.

`dapo_loss` accepts `[B,L]` current/old scores and binary masks plus `[B]` group-normalized advantages. These responses have already been retained by the caller. Freeze old scores and advantages, use lower/upper clip widths independently, and normalize by the **total valid token count**, not by each response's length. Defaults are `.2/.28`. There is no reference-policy KL term. An empty mask returns differentiable zero; mask extreme padded scores before exponentiation.

The toy keeps only the mixed group among all-fail, mixed and all-success examples, then scans ratios from .5 to 1.5. A ratio 1.25 for positive advantage should still have nonzero gradient under clip_high=.28. The tests also distinguish unequal response-length weighting.

## Deliberately wrong variants

- Use the lower width for both sides.
- Normalize within each response then average.
- Retain all-success groups.
- Allow gradients into the behavior policy.

Each mutant differs from the reference in a small, intentional change. T2 includes independent formula or gradient expectations, so equality with the reference is not the only acceptance evidence. T3 produces artifacts from the supplied implementation.

Source: [2503.14476v1](https://arxiv.org/html/2503.14476v1#S3); [Official framework linked by the paper (redirects to verl-project/verl)](https://github.com/volcengine/verl). Toy provenance: LeetPaper synthetic binary-reward groups and ratio sweep; no AIME reproduction.
