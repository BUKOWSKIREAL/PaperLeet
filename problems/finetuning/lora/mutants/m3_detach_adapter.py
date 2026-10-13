# Intentional bug: Detach the low-rank update.
import torch
from torch import Tensor


def lora_delta(a: Tensor, b: Tensor, alpha: float = 1.0) -> Tensor:
    """A [r,in], B [out,r] -> delta W [out,in], scaled by alpha/r."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 1 or a.shape[0] != b.shape[1]:
        raise ValueError('A and B must be rank-compatible matrices with positive rank')
    return (b @ a) * (alpha / a.shape[0])


def lora_linear(x: Tensor, weight: Tensor, a: Tensor, b: Tensor,
                alpha: float = 1.0, bias: Tensor | None = None) -> Tensor:
    """Frozen base and bias; gradients still flow to x and both adapters."""
    base = torch.nn.functional.linear(x, weight.detach(), None if bias is None else bias.detach())
    return base + torch.nn.functional.linear(x, lora_delta(a, b, alpha).detach())
