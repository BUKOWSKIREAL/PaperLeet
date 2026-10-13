# Bug: wrong jvp time.
import torch
from torch import Tensor


def _time(t: Tensor, x: Tensor) -> Tensor:
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def meanflow_target(model, z: Tensor, r: Tensor, t: Tensor, velocity: Tensor) -> tuple[Tensor, Tensor]:
    """Model arguments are (z, r, t); r <= t, t=0 data, t=1 noise."""
    prediction, derivative = torch.func.jvp(model, (z, r, t), (velocity, torch.ones_like(r), torch.zeros_like(t)))
    target = (velocity - _time(t - r, z) * derivative).detach()
    return prediction, target


def meanflow_loss(model, clean: Tensor, noise: Tensor, r: Tensor, t: Tensor) -> Tensor:
    """Unweighted MSE core (adaptive loss weighting is outside this module)."""
    z = (1 - _time(t, clean)) * clean + _time(t, clean) * noise
    prediction, target = meanflow_target(model, z, r, t, noise - clean)
    return (prediction - target).square().mean()


@torch.no_grad()
def one_step(model, noise: Tensor) -> Tensor:
    """One model call: z_0 = z_1 - u(z_1, 0, 1)."""
    r = noise.new_zeros(len(noise)); t = noise.new_ones(len(noise))
    return noise - model(noise, r, t)
