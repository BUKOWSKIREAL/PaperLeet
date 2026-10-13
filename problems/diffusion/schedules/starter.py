"""按 README 中的形状和时间约定实现以下公开函数。"""
import torch
from torch import Tensor
from typing import Callable

def linear_betas(T: int, beta_start: float=0.0001, beta_end: float=0.02) -> Tensor:
    """Return [T] float64 linearly spaced variances."""
    raise NotImplementedError

def cosine_betas(T: int, s: float=0.008, max_beta: float=0.999) -> Tensor:
    """Discretize cosine cumulative alpha and clip beta."""
    raise NotImplementedError

def make_schedule(betas: Tensor) -> dict[str, Tensor]:
    """All values [T] float64; alpha_bars_prev[0] = 1."""
    raise NotImplementedError

def q_sample(x0: Tensor, t: Tensor, noise: Tensor, alpha_bars: Tensor) -> Tensor:
    """x0/noise [B,...], t [B] long; return x0's shape and dtype."""
    raise NotImplementedError
