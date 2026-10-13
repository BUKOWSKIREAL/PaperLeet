# 17. JiT: Clean Prediction in High Dimensions

Implement clean-data prediction with velocity-space supervision and compare x/epsilon/v outputs on embedded data.

## Contract

- `make_noisy(clean: Tensor, noise: Tensor, t: Tensor) -> Tensor`
- `prediction_to_velocity(prediction: Tensor, z: Tensor, t: Tensor, kind: str = 'x', eps: float = .01) -> Tensor`
- `velocity_loss(prediction: Tensor, clean: Tensor, noise: Tensor, t: Tensor, kind: str = 'x', eps: float = .01) -> Tensor`

For t away from the clamped endpoints, v=x-epsilon, x prediction maps via (x_hat-z)/(1-t), and epsilon prediction via (z-epsilon_hat)/t. All networks use the same velocity loss. Clamping follows official x-prediction code; exact cross-parameterization equivalence is only asserted outside the clamped boundary region. Broadcast t from [B] over any trailing dimensions.

## Runnable experiment

**Provenance: Paper experiment reimplementation, reduced.** Reimplements the fixed orthogonal 2D→D embedding idea in Figure 2 with synthetic spiral data. Default D=16, width=32, depth=3, 160 steps; the original uses D=2/8/16/512 and width=256, depth=5. Run `python problems/diffusion/jit/toy.py --impl problems/diffusion/jit/solution.py --full --steps 5000` for that dimension/architecture sweep; the step count and spiral distribution are our choices. Short-run comparisons are observations, not a guarantee of the paper ranking. No original toy source was found.

The T3 test runs your submitted implementation, saves points/curves/metrics, and uses fixed seed 42. `toy.py` contains the provided experiment. It does not download models or datasets. These exercises reproduce selected mechanisms, not full image benchmark results.

## Sources

- https://arxiv.org/abs/2511.13720
- https://github.com/LTH14/JiT/blob/cbc743a2ada5e9762697da2c83f8c4f8379e8c17/denoiser.py

Anchors: arXiv v1 §3.2 prediction and loss spaces; §3.3 Figure 2 high-dimensional manifold toy; Official Denoiser.forward.

In the experiment scaffold, depth counts all Linear layers including the output; `--full` therefore has five Linear layers with ReLU between them.
