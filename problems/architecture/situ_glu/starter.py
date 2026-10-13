import math
import torch
from torch import Tensor


def softcap(x: Tensor, beta: float) -> Tensor:
    """Smoothly cap a floating tensor using a finite positive scalar beta."""
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    raise NotImplementedError("Implement softcap")


def situ_glu(gate: Tensor, up: Tensor, beta1: float = 4.0,
             beta2: float = 25.0) -> Tensor:
    """Apply SiTU-GLU to two already projected, equally shaped branches."""
    raise NotImplementedError("Implement situ_glu")
