# Review: Drifting: Pairwise Field and Frozen Targets

The field uses exp(-Euclidean distance/temperature), masks generated self-pairs, normalizes by sqrt(row_mass*column_mass), and combines positive and negative weighted centers with their opposite group masses. This selects the official toy variant, not a generic attraction-only field. The whole g+V target is frozen; removing this detach can preserve the scalar loss while making the gradient zero. One-step means generator inference, not one training update.

Adapts the authors’ Swiss-roll/checkerboard notebook. Reduced from noise32→3×256, batch2048, 2000 steps to noise8→2×48, generated96/positive128, 260 steps and temperature .2 for CPU grading. No pretrained features, CFG, or image benchmarks. Paper Figure 3 is a separate bimodal experiment.

The deliberately incorrect implementations isolate row only kernel, attraction only, unfrozen target, self pairs unmasked. T1 checks formula values, T2 checks invariants/gradient routing, and T3 invokes the learner implementation in the toy.

Numerical adaptation: compute the doubly normalized kernel with logsumexp rather than the notebook’s exp/product clamp. This preserves the mathematical normalized affinity when raw exponentials underflow and intentionally removes its absolute 1e-12 product floor. Empty sample sets and nonpositive/nonfinite temperature are rejected.
