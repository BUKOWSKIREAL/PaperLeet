# 18. Drifting: Pairwise Field and Frozen Targets

Implement the distance-kernel drifting field and its stop-gradient target; train a generator with one-pass inference.

## Contract

- `drift_field(generated: Tensor, positive: Tensor, temperature: float = .2) -> Tensor`
- `drifting_loss(generated: Tensor, positive: Tensor, temperature: float = .2) -> Tensor`

The field uses exp(-Euclidean distance/temperature), masks generated self-pairs, normalizes by sqrt(row_mass*column_mass), and combines positive and negative weighted centers with their opposite group masses. This selects the official toy variant, not a generic attraction-only field. The whole g+V target is frozen; removing this detach can preserve the scalar loss while making the gradient zero. One-step means generator inference, not one training update.

## Runnable experiment

**Provenance: Official toy adaptation.** Adapts the authors’ Swiss-roll/checkerboard notebook. Reduced from noise32→3×256, batch2048, 2000 steps to noise8→2×48, generated96/positive128, 260 steps and temperature .2 for CPU grading. No pretrained features, CFG, or image benchmarks. Paper Figure 3 is a separate bimodal experiment.

The T3 test runs your submitted implementation, saves points/curves/metrics, and uses fixed seed 42. `toy.py` contains the provided experiment. It does not download models or datasets. These exercises reproduce selected mechanisms, not full image benchmark results.

## Sources

- https://arxiv.org/abs/2602.04770
- https://github.com/lambertae/lambertae.github.io/blob/a7fa4baae52abbab2aaeb57136d557caee79ff39/projects/drifting/notebooks/drifting_model_demo.ipynb
- https://github.com/lambertae/drifting/tree/8861066430db11d2fb0858d042c7845b99df49f1

Anchors: arXiv v2 §3 drifting field and fixed-point loss; §5.1 Figure 3 bimodal toy; Official notebook compute_drift and drifting_loss.

Numerical adaptation: compute the doubly normalized kernel with logsumexp rather than the notebook’s exp/product clamp. This preserves the mathematical normalized affinity when raw exponentials underflow and intentionally removes its absolute 1e-12 product floor. Empty sample sets and nonpositive/nonfinite temperature are rejected.
