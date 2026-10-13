"""DAPS Eq. (7), specialized to a supplied linear measurement operator."""
import math
import torch
from torch import Tensor


def _positive(value, name, allow_zero=False):
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} has an invalid value")


def _matching(x, *others):
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("finite floating tensors required")
    for other in others:
        if other.shape != x.shape or other.dtype != x.dtype or other.device != x.device or not torch.isfinite(other).all():
            raise ValueError("tensor shape/dtype/device must match")


def posterior_score(x: Tensor, x0_hat: Tensor, operator: Tensor, measurement: Tensor,
                    r: float, beta: float) -> Tensor:
    """Score of N(x0_hat, r^2 I) times N(measurement; A x, beta^2 I)."""
    _positive(r, "r"); _positive(beta, "beta")
    _matching(x, x0_hat)
    if x.ndim != 2 or operator.ndim != 2 or operator.shape[1] != x.shape[1]:
        raise ValueError("x is (B,D); operator is (M,D)")
    if measurement.shape != (x.shape[0], operator.shape[0]):
        raise ValueError("measurement must be (B,M)")
    for value in (operator, measurement):
        if value.dtype != x.dtype or value.device != x.device or not torch.isfinite(value).all():
            raise ValueError("finite matching dtype/device required")
    prior = (x0_hat - x) / r ** 2
    likelihood = (measurement - x @ operator.T) @ operator / beta ** 2
    return prior + likelihood


def langevin_step(x: Tensor, score: Tensor, eta: float, noise: Tensor) -> Tensor:
    """Euler Langevin step; caller supplies fresh standard-normal noise explicitly."""
    _positive(eta, "eta", allow_zero=True)
    _matching(x, score, noise)
    return x + eta * score + math.sqrt(2 * eta) * noise


def renoise(x0: Tensor, sigma_next: float, noise: Tensor) -> Tensor:
    """Decoupled transition N(x0, sigma_next^2 I), with fresh independent noise."""
    _positive(sigma_next, "sigma_next", allow_zero=True)
    _matching(x0, noise)
    return x0 + sigma_next * noise
