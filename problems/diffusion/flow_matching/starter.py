"""按 README 中的形状和时间约定实现以下公开函数。"""
import torch
from torch import Tensor
from typing import Callable

def sample_t(batch: int, generator: torch.Generator | None, dist: str='uniform', m: float=0.0, s: float=1.0) -> Tensor:
    """[B] strictly in (0,1); logit_normal uses sigmoid(N(m,s²))."""
    raise NotImplementedError

def interpolate(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """Broadcast t [B] to x0/x1 [B,...]."""
    raise NotImplementedError

def cfm_target(x0: Tensor, x1: Tensor) -> Tensor:
    """Conditional velocity [B,...] along the linear interpolation."""
    raise NotImplementedError

def cfm_loss(model: Callable[[Tensor, Tensor], Tensor], x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
    """Scalar elementwise mean squared velocity error; x0 is noise."""
    raise NotImplementedError

def euler_sample(model: Callable[[Tensor, Tensor], Tensor], x0: Tensor, num_steps: int, return_traj: bool=False) -> Tensor | tuple[Tensor, Tensor]:
    """Evaluate t=k/n only, k=0..n-1; optional (samples,[n+1,B,D]) trajectory."""
    raise NotImplementedError
