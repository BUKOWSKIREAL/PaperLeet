# Bug: 反向扩散错误地按正序运行
"""Ancestral DDPM sampling with a caller-supplied epsilon predictor."""
import torch
from torch import Tensor
from typing import Callable


def _extract(values, t, x):
    return values.to(x)[t].reshape(-1, *([1] * (x.ndim - 1)))


def q_posterior(x0: Tensor, x_t: Tensor, t: Tensor, betas: Tensor, alpha_bars: Tensor,
                alpha_bars_prev: Tensor) -> tuple[Tensor, Tensor]:
    """Return posterior mean [B,...] and broadcastable variance [B,1,...]."""
    beta = _extract(betas, t, x_t)
    ab, prev = _extract(alpha_bars, t, x_t), _extract(alpha_bars_prev, t, x_t)
    c0 = prev.sqrt() * beta / (1 - ab)
    ct = (1 - beta).sqrt() * (1 - prev) / (1 - ab)
    mean = c0 * x0 + ct * x_t
    var = beta * (1 - prev) / (1 - ab)
    return mean, var


def p_sample_step(eps_pred: Tensor, x_t: Tensor, t: Tensor, betas: Tensor,
                  alpha_bars: Tensor, alpha_bars_prev: Tensor, noise: Tensor,
                  var_type: str = "posterior") -> Tensor:
    """Use posterior or beta variance; samples with index t=0 receive no noise."""
    ab = _extract(alpha_bars, t, x_t)
    x0 = (x_t - (1 - ab).sqrt() * eps_pred) / ab.sqrt()
    mean, var = q_posterior(x0, x_t, t, betas, alpha_bars, alpha_bars_prev)
    if var_type == "beta":
        var = _extract(betas, t, x_t)
    elif var_type != "posterior":
        raise ValueError("var_type must be posterior or beta")
    nonzero = (t != 0).to(x_t).reshape(-1, *([1] * (x_t.ndim - 1)))
    return mean + nonzero * var.clamp(min=0).sqrt() * noise


@torch.no_grad()
def ddpm_sample(eps_model: Callable[[Tensor, Tensor], Tensor], shape: tuple[int, ...], betas: Tensor, generator: torch.Generator, var_type: str = "posterior", return_traj: bool = False) -> Tensor | tuple[Tensor, Tensor]:
    """Start at N(0,I), traverse T-1..0; optionally return (samples, [T+1,B,...])."""
    ab = (1 - betas).cumprod(0)
    prev = torch.cat((torch.ones_like(ab[:1]), ab[:-1]))
    x = torch.randn(shape, dtype=betas.dtype, device=betas.device, generator=generator)
    trajectory = [x.clone()] if return_traj else None
    for step in range(len(betas)):
        t = torch.full((shape[0],), step, dtype=torch.long, device=x.device)
        noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        x = p_sample_step(eps_model(x, t), x, t, betas, ab, prev, noise, var_type)
        if return_traj:
            trajectory.append(x.clone())
    return (x, torch.stack(trajectory)) if return_traj else x
