"""按 README 中的形状和时间约定实现以下公开函数。"""
import torch
from torch import Tensor
from typing import Callable

def ddim_step(x_t: Tensor, eps_pred: Tensor, ab_t: Tensor, ab_prev: Tensor, eta: float, noise: Tensor) -> Tensor:
    """Coefficients broadcast to x_t; final ab_prev=1.
Sample tensors [B,...], schedule tensors [T], indices [B] long; model(x,t) returns epsilon of x.shape."""
    raise NotImplementedError

def ddim_timesteps(T: int, num_steps: int) -> Tensor:
    """[num_steps] long: T-1-k*(T//num_steps), strict descending."""
    raise NotImplementedError

def ddim_sample(eps_model: Callable[[Tensor, Tensor], Tensor], x_T: Tensor, alpha_bars: Tensor, num_steps: int, eta: float=0.0, generator: torch.Generator | None=None, return_traj: bool=False) -> Tensor | tuple[Tensor, Tensor]:
    """Return samples or (samples, trajectory including x_T and clean endpoint).
Sample tensors [B,...], schedule tensors [T], indices [B] long; model(x,t) returns epsilon of x.shape."""
    raise NotImplementedError

def ddim_invert(eps_model: Callable[[Tensor, Tensor], Tensor], x0: Tensor, alpha_bars: Tensor, num_steps: int) -> Tensor:
    """First-order deterministic inversion; evaluate eps at the current time.
Sample tensors [B,...], schedule tensors [T], indices [B] long; model(x,t) returns epsilon of x.shape."""
    raise NotImplementedError
