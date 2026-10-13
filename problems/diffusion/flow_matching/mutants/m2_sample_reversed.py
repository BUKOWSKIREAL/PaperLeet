# Bug: 采样方向从数据到噪声
"""Conditional flow matching: t=0 is noise and t=1 is data."""
import torch
from torch import Tensor
from typing import Callable


def sample_t(batch: int, generator: torch.Generator | None, dist: str = "uniform", m: float = 0.0, s: float = 1.0) -> Tensor:
    """[B] strictly in (0,1); logit_normal uses sigmoid(N(m,s²))."""
    if dist == "uniform":
        t = torch.rand(batch, generator=generator)
    elif dist == "logit_normal":
        t = (m + s * torch.randn(batch, generator=generator)).sigmoid()
    else:
        raise ValueError("dist must be uniform or logit_normal")
    return t.clamp(torch.finfo(t.dtype).eps, 1 - torch.finfo(t.dtype).eps)


def interpolate(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """Broadcast t [B] to x0/x1 [B,...]."""
    time = t.reshape(-1, *([1] * (x0.ndim - 1)))
    return (1 - time) * x0 + time * x1


def cfm_target(x0: Tensor, x1: Tensor) -> Tensor:
    """Conditional velocity [B,...] along the linear interpolation."""
    return x1 - x0


def cfm_loss(model: Callable[[Tensor, Tensor], Tensor], x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
    """Scalar elementwise mean squared velocity error; x0 is noise."""
    return (model(interpolate(x0, x1, t), t) - cfm_target(x0, x1)).square().mean()


@torch.no_grad()
def euler_sample(model: Callable[[Tensor, Tensor], Tensor], x0: Tensor, num_steps: int, return_traj: bool = False) -> Tensor | tuple[Tensor, Tensor]:
    """Evaluate t=k/n only, k=0..n-1; optional (samples,[n+1,B,D]) trajectory."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    x = x0.clone()
    trajectory = [x.clone()] if return_traj else None
    for k in range(num_steps):
        t = torch.full((len(x),), 1 - k / num_steps, dtype=x.dtype, device=x.device)
        x = x - model(x, t) / num_steps
        if return_traj:
            trajectory.append(x.clone())
    return (x, torch.stack(trajectory)) if return_traj else x
