import torch
from torch import Tensor


def adamw_step(parameter: Tensor, gradient: Tensor, first_moment: Tensor,
               second_moment: Tensor, step: int, lr: float = 1e-3,
               betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
               weight_decay: float = 0.01) -> tuple[Tensor, Tensor, Tensor]:
    """One functional AdamW update. step is one-based; inputs are not modified."""
    beta1, beta2 = betas
    moment = beta1 * first_moment + (1 - beta1) * gradient
    variance = beta2 * second_moment + (1 - beta2) * gradient.square()
    corrected_moment = moment / (1 - beta1 ** step)
    corrected_variance = variance / (1 - beta2 ** step)
    updated = parameter * (1 - lr * weight_decay) - lr * corrected_moment / (corrected_variance.sqrt() + eps)
    return updated, moment, variance
