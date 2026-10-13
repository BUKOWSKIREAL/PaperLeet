"""Implement the selected core functions; the experiment scaffold is provided."""
import torch
from torch import Tensor


def _time(t: Tensor, x: Tensor) -> Tensor:
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def score_difference(real_score: Tensor, fake_score: Tensor, weight: Tensor) -> Tensor:
    """KL(q||p) descent gradient is weight*(score_q-score_p), treated as a target."""
    # TODO: implement this paper core.
    raise NotImplementedError


def dmd_loss(generated: Tensor, gradient: Tensor, paired_target: Tensor, regression_weight: float = .1) -> Tensor:
    """Surrogate gives gradient/N plus the paired regression gradient."""
    # TODO: implement this paper core.
    raise NotImplementedError


def fake_score_loss(model, generated: Tensor, noise: Tensor, sigma: Tensor) -> Tensor:
    """VE denoising score matching on detached student samples; sigma must be positive."""
    # TODO: implement this paper core.
    raise NotImplementedError
