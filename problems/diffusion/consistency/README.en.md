# 21. Consistency Models: Boundary and Teacher Targets

Implement a boundary-preserving consistency function, stop-gradient pair loss, and EMA target update.

## Contract

- `consistency_function(model, noisy: Tensor, sigma: Tensor, sigma_min: float = .002, sigma_data: float = .5) -> Tensor`
- `consistency_loss(prediction: Tensor, target_prediction: Tensor) -> Tensor`
- `update_ema(online, target, decay: float = .95) -> None`
- `teacher_euler(noisy: Tensor, denoised: Tensor, sigma: Tensor, next_sigma: Tensor) -> Tensor` (provided)

At sigma_min the function must be exactly the identity, regardless of network output. The model receives c_in*x and the original sigma; this exercise supplies raw sigma conditioning instead of the official embedding. The skip/output coefficients follow the official boundary parameterization. The lower-noise teacher/EMA branch is frozen. The supplied Euler helper uses decreasing positive noise scales. This is consistency distillation with a teacher, not the separate teacher-free consistency-training algorithm.

## Runnable experiment

**Provenance: PaperLeet teaching adaptation.** A point-data teacher makes its probability-flow Euler step exact. Small MLP distillation exercises the boundary and target routing without pretrained models. Half the training pairs use a positive adjacent noise level sigma/2, so the EMA network affects the target; the other half anchor the exact sigma_min boundary.

The T3 test runs your submitted implementation, saves points/curves/metrics, and uses fixed seed 42. `toy.py` contains the provided experiment. It does not download models or datasets. These exercises reproduce selected mechanisms, not full image benchmark results.

## Sources

- https://arxiv.org/abs/2303.01469
- https://github.com/openai/consistency_models/blob/e32b69ee436d518377db86fb2127a3972d0d8716/cm/karras_diffusion.py

Anchors: arXiv v1 §3 consistency function boundary parameterization; §4 consistency distillation and EMA target; Official get_scalings_for_boundary_condition.
