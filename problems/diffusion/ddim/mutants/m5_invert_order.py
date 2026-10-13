# Bug: 反演错误地降序遍历
"""DDIM with explicit trailing subsequences and a clean final boundary."""
import torch
from torch import Tensor
from typing import Callable


def ddim_step(x_t: Tensor, eps_pred: Tensor, ab_t: Tensor, ab_prev: Tensor,
              eta: float, noise: Tensor) -> Tensor:
    """Coefficients broadcast to x_t; final ab_prev=1."""
    x0 = (x_t - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
    sigma = eta * ((1 - ab_prev) / (1 - ab_t)).sqrt() * (1 - ab_t / ab_prev).clamp(min=0).sqrt()
    direction = (1 - ab_prev - sigma.square()).clamp(min=0).sqrt()
    return ab_prev.sqrt() * x0 + direction * eps_pred + sigma * noise


def ddim_timesteps(T: int, num_steps: int) -> Tensor:
    """[num_steps] long: T-1-k*(T//num_steps), strict descending."""
    if not 1 <= num_steps <= T:
        raise ValueError("num_steps must lie in [1,T]")
    return T - 1 - torch.arange(num_steps, dtype=torch.long) * (T // num_steps)


@torch.no_grad()
def ddim_sample(eps_model: Callable[[Tensor, Tensor], Tensor], x_T: Tensor, alpha_bars: Tensor, num_steps: int, eta: float = 0.0, generator: torch.Generator | None = None, return_traj: bool = False) -> Tensor | tuple[Tensor, Tensor]:
    """Return samples or (samples, trajectory including x_T and clean endpoint)."""
    steps = ddim_timesteps(len(alpha_bars), num_steps).tolist()
    ab = alpha_bars.to(x_T)
    x = x_T.clone()
    trajectory = [x.clone()] if return_traj else None
    for i, step in enumerate(steps):
        previous = ab[steps[i + 1]] if i + 1 < len(steps) else torch.ones_like(ab[0])
        t = torch.full((len(x),), step, dtype=torch.long, device=x.device)
        noise = torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device) if eta else torch.zeros_like(x)
        x = ddim_step(x, eps_model(x, t), ab[step], previous, eta, noise)
        if return_traj:
            trajectory.append(x.clone())
    return (x, torch.stack(trajectory)) if return_traj else x


@torch.no_grad()
def ddim_invert(eps_model: Callable[[Tensor, Tensor], Tensor], x0: Tensor, alpha_bars: Tensor, num_steps: int) -> Tensor:
    """First-order deterministic inversion; evaluate eps at the current time."""
    steps = ddim_timesteps(len(alpha_bars), num_steps).tolist()
    ab = alpha_bars.to(x0)
    x = x0.clone()
    current_ab = torch.ones_like(ab[0])
    current_step = 0
    for step in steps:
        t = torch.full((len(x),), current_step, dtype=torch.long, device=x.device)
        eps = eps_model(x, t)
        clean = (x - (1 - current_ab).sqrt() * eps) / current_ab.sqrt()
        x = ab[step].sqrt() * clean + (1 - ab[step]).sqrt() * eps
        current_ab, current_step = ab[step], step
    return x
