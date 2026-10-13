import torch
from torch import Tensor


def _time(t: Tensor, x: Tensor) -> Tensor:
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def score_difference(real_score: Tensor, fake_score: Tensor, weight: Tensor) -> Tensor:
    """KL(q||p) descent gradient is weight*(score_q-score_p), treated as a target."""
    return (_time(weight, real_score) * (fake_score - real_score)).detach()


def dmd_loss(generated: Tensor, gradient: Tensor, paired_target: Tensor, regression_weight: float = .1) -> Tensor:
    """Surrogate gives gradient/N plus the paired regression gradient."""
    target = (generated - gradient).detach()
    matching = .5 * (generated - target).square().mean()
    regression = (generated - paired_target.detach()).square().mean()
    return matching + regression_weight * regression


def fake_score_loss(model, generated: Tensor, noise: Tensor, sigma: Tensor) -> Tensor:
    """VE denoising score matching on detached student samples; sigma must be positive."""
    scale = _time(sigma, generated)
    clean = generated.detach()
    noisy = clean + scale * noise
    return (scale * model(noisy, sigma) + noise).square().mean()
