"""按 README 中的形状和时间约定实现以下公开函数。"""
import torch
from torch import Tensor
from typing import Callable

def eps_to_x0(x_t: Tensor, eps: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    """Recover clean data; coefficients broadcast to x_t, denominator >= 1e-8.
Tensor inputs [B,...]; alpha/sigma broadcast to inputs, alpha²+sigma²=1. Return same shape and dtype."""
    raise NotImplementedError

def x0_to_eps(x_t: Tensor, x0: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    """Recover noise, retaining finite values at the clean endpoint.
Tensor inputs [B,...]; alpha/sigma broadcast to inputs, alpha²+sigma²=1. Return same shape and dtype."""
    raise NotImplementedError

def make_v(x0: Tensor, eps: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    """Angular velocity for x_t = alpha*x0 + sigma*eps.
Tensor inputs [B,...]; alpha/sigma broadcast to inputs, alpha²+sigma²=1. Return same shape and dtype."""
    raise NotImplementedError

def v_to_x0(x_t: Tensor, v: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    """Inverse orthogonal rotation, stable at both endpoints.
Tensor inputs [B,...]; alpha/sigma broadcast to inputs, alpha²+sigma²=1. Return same shape and dtype."""
    raise NotImplementedError

def v_to_eps(x_t: Tensor, v: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    """Inverse orthogonal rotation, stable at both endpoints.
Tensor inputs [B,...]; alpha/sigma broadcast to inputs, alpha²+sigma²=1. Return same shape and dtype."""
    raise NotImplementedError

def convert(pred: Tensor, src: str, dst: str, x_t: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    """Convert a prediction between x0, eps and v; preserve shape/dtype.
Tensor inputs [B,...]; alpha/sigma broadcast to inputs, alpha²+sigma²=1. Return same shape and dtype."""
    raise NotImplementedError
