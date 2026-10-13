import torch
from torch import Tensor


def positionwise_ffn(x: Tensor, w1: Tensor, b1: Tensor,
                     w2: Tensor, b2: Tensor) -> Tensor:
    """x [..., D], w1 [H, D], b1 [H], w2 [D, H], b2 [D]; no dropout."""
    hidden = torch.relu(x @ w1.transpose(-1, -2) + b1)
    return hidden @ w2.transpose(-1, -2) + b2
