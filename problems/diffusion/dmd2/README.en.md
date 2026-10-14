# 20. DMD2: Two-Time-Scale and Adversarial Updates

Implement the critic-to-generator update ratio and non-saturating GAN contribution on top of distribution matching.

## Contract

- `generator_due(completed_critic_updates: int, critic_per_generator: int = 5) -> bool`
- `adversarial_losses(real_logits: Tensor, fake_logits: Tensor) -> tuple[Tensor, Tensor]`
- `generator_loss(generated: Tensor, gradient: Tensor, fake_logits: Tensor, gan_weight: float = .1) -> Tensor`

The critic/fake diffusion model updates more often than the generator (5:1 here). Counts are completed critic updates, so update 0 does not trigger a generator step. Logistic discriminator and generator losses have opposite signs on fake logits. Caller detaches fake data during critic training and freezes critic parameters, while retaining input gradients, during generator training. Unlike DMD, the generator objective has no paired teacher regression. This exercise covers the stabilization/real-data contribution, not the paper’s multistep backward simulation.

## Runnable experiment

**Provenance: LeetPaper teaching adaptation.** An unpaired shifted-Gaussian toy jointly trains a fake score and discriminator for five updates per student update. It demonstrates scheduling, GAN gradient direction, and removal of paired regression. Uses scalar Gaussian score parameters rather than pretrained diffusion features.

The T3 test runs your submitted implementation, saves points/curves/metrics, and uses fixed seed 42. `toy.py` contains the provided experiment. It does not download models or datasets. These exercises reproduce selected mechanisms, not full image benchmark results.

## Sources

- https://arxiv.org/abs/2405.14867
- https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/main/edm/edm_guidance.py

Anchors: arXiv v1 §3 two-time-scale updates and GAN loss; Official EDM compute_generator_clean_cls_loss and compute_guidance_clean_cls_loss.
