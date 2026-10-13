# Bug: Detaching the capped up branch blocks its gradient.
import math
import torch
from torch import Tensor


def softcap(x: Tensor, beta: float) -> Tensor:
    """Smoothly cap a floating tensor using a finite positive scalar beta."""
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    return beta * torch.tanh(x / beta)


def situ_glu(gate: Tensor, up: Tensor, beta1: float = 4.0,
             beta2: float = 25.0) -> Tensor:
    """Apply SiTU-GLU to two already projected, equally shaped branches."""
    return softcap(gate, beta1) * torch.sigmoid(gate) * softcap(up, beta2).detach()
