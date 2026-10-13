# Bug: wrong ddim sign.
import torch
from torch import Tensor


def _time(t: Tensor, x: Tensor) -> Tensor:
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def ddim_step(noisy: Tensor, clean_prediction: Tensor, alpha_t: Tensor, sigma_t: Tensor, alpha_s: Tensor, sigma_s: Tensor) -> Tensor:
    """Deterministic DDIM from a noisier t to s, with alpha/sigma amplitudes [B]."""
    ratio = _time(sigma_s / sigma_t, noisy)
    return ratio * noisy + (_time(alpha_s, noisy) + ratio * _time(alpha_t, noisy)) * clean_prediction


def distilled_target(noisy: Tensor, teacher_endpoint: Tensor, alpha_t: Tensor, sigma_t: Tensor, alpha_s: Tensor, sigma_s: Tensor) -> Tensor:
    """Solve for the clean prediction whose single DDIM step hits the two-step teacher endpoint."""
    ratio = _time(sigma_s / sigma_t, noisy)
    coefficient = _time(alpha_s, noisy) - ratio * _time(alpha_t, noisy)
    return ((teacher_endpoint - ratio * noisy) / coefficient).detach()


def distillation_loss(prediction: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    """Weighted squared clean-prediction error; target carries no teacher gradient."""
    return (_time(weight, prediction) * (prediction - target.detach()).square()).mean()


@torch.no_grad()
def teacher_pair(teacher, noisy: Tensor, t: Tensor, midpoint: Tensor, end: Tensor) -> Tensor:
    """Provided cosine schedule and exactly two teacher calls; t > midpoint > end >= 0."""
    a=lambda time: torch.cos(time*torch.pi/2)
    s=lambda time: torch.sin(time*torch.pi/2)
    middle=ddim_step(noisy,teacher(noisy,t),a(t),s(t),a(midpoint),s(midpoint))
    return ddim_step(middle,teacher(middle,midpoint),a(midpoint),s(midpoint),a(end),s(end))
