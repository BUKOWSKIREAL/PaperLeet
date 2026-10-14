# 16. MeanFlow: Average Velocity and JVP Targets

Implement the total-derivative average-velocity target and its frozen-target loss, then sample in one call.

## Contract

- `meanflow_target(model, z: Tensor, r: Tensor, t: Tensor, velocity: Tensor) -> tuple[Tensor, Tensor]`
- `meanflow_loss(model, clean: Tensor, noise: Tensor, r: Tensor, t: Tensor) -> Tensor`
- `one_step(model, noise: Tensor) -> Tensor`

Use paper argument order `(z,r,t)`: JVP tangents `(v,0,1)` hold r fixed. Noise increases with t. The target is `stopgrad(v-(t-r) du/dt)`. We use elementwise mean MSE, omitting the optional adaptive weighting in the official image implementation. At r=t only the target becomes conditional flow matching; the untrained network need not predict v.

## Runnable experiment

**Provenance: LeetPaper teaching adaptation.** An analytic point-data family learns its center using the learner JVP target. This is a mechanism demonstration, not a low-dimensional experiment claimed by the paper.

The T3 test runs your submitted implementation, saves points/curves/metrics, and uses fixed seed 42. `toy.py` contains the provided experiment. It does not download models or datasets. These exercises reproduce selected mechanisms, not full image benchmark results.

## Sources

- https://arxiv.org/abs/2505.13447
- https://github.com/Gsunshine/py-meanflow/blob/1f6d72d94247c8fdeb89489acca1a8007a6baf6c/meanflow/models/meanflow.py

Anchors: arXiv v1 §3 MeanFlow identity and stop-gradient training objective; Official forward_with_loss and sample.
