import torch
from torch import Tensor


def adamw_step(parameter: Tensor, gradient: Tensor, first_moment: Tensor,
               second_moment: Tensor, step: int, lr: float = 1e-3,
               betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
               weight_decay: float = 0.01) -> tuple[Tensor, Tensor, Tensor]:
    """Required: return new parameter, first moment and second moment.
    Same-shaped floating tensors; moments start at zero and step starts at 1.
    Apply bias correction, eps outside sqrt, and decoupled decay lr*weight_decay.
    Do not change inputs or add weight decay to gradient/moment estimates.
    Keep tensor autograd paths; the caller decides when to disable gradients.
    No AMSGrad, maximize option, or schedule multiplier in this exercise."""
    raise NotImplementedError()
