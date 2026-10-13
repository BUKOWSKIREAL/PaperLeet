# Bug: wrong time direction.
import torch
from torch import Tensor


def _time(t: Tensor, x: Tensor) -> Tensor:
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def make_noisy(clean: Tensor, noise: Tensor, t: Tensor) -> Tensor:
    """z_t=t*x+(1-t)*epsilon, t=0 noise and t=1 clean."""
    time = _time(t, clean)
    return (1 - time) * clean + time * noise


def prediction_to_velocity(prediction: Tensor, z: Tensor, t: Tensor, kind: str = 'x', eps: float = .01) -> Tensor:
    """Map x/eps/v predictions to velocity; clamp each denominator at eps."""
    time = _time(t, z)
    if kind == 'x':
        return (prediction - z) / (1 - time).clamp_min(eps)
    if kind == 'eps':
        return (z - prediction) / time.clamp_min(eps)
    if kind == 'v':
        return prediction
    raise ValueError('kind must be x, eps or v')


def velocity_loss(prediction: Tensor, clean: Tensor, noise: Tensor, t: Tensor, kind: str = 'x', eps: float = .01) -> Tensor:
    """All prediction types use velocity-space MSE, matching the x-code clamped target."""
    z = make_noisy(clean, noise, t)
    target = (clean - z) / (1 - _time(t, z)).clamp_min(eps)
    return (prediction_to_velocity(prediction, z, t, kind, eps) - target).square().mean()
