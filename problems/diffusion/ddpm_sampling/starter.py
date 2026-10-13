"""按 README 中的形状和时间约定实现以下公开函数。"""
import torch
from torch import Tensor
from typing import Callable

def q_posterior(x0: Tensor, x_t: Tensor, t: Tensor, betas: Tensor, alpha_bars: Tensor, alpha_bars_prev: Tensor) -> tuple[Tensor, Tensor]:
    """Return posterior mean [B,...] and broadcastable variance [B,1,...].
Sample tensors [B,...], schedule tensors [T], indices [B] long; model(x,t) returns epsilon of x.shape."""
    raise NotImplementedError

def p_sample_step(eps_pred: Tensor, x_t: Tensor, t: Tensor, betas: Tensor, alpha_bars: Tensor, alpha_bars_prev: Tensor, noise: Tensor, var_type: str='posterior') -> Tensor:
    """Use posterior or beta variance; samples with index t=0 receive no noise.
Sample tensors [B,...], schedule tensors [T], indices [B] long; model(x,t) returns epsilon of x.shape."""
    raise NotImplementedError

def ddpm_sample(eps_model: Callable[[Tensor, Tensor], Tensor], shape: tuple[int, ...], betas: Tensor, generator: torch.Generator, var_type: str='posterior', return_traj: bool=False) -> Tensor | tuple[Tensor, Tensor]:
    """Start at N(0,I), traverse T-1..0; optionally return (samples, [T+1,B,...]).
Sample tensors [B,...], schedule tensors [T], indices [B] long; model(x,t) returns epsilon of x.shape."""
    raise NotImplementedError
