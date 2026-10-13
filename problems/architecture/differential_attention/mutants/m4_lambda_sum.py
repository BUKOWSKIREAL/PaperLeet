# BUG: Adds exponentials in the lambda parameterization.
import math
import torch
from torch import Tensor


def differential_lambda(q1: Tensor, k1: Tensor, q2: Tensor, k2: Tensor,
                        lambda_init: float = 0.2) -> Tensor:
    """The paper's scalar exponential-difference parameterization."""
    if q1.ndim != 1 or q1.numel() == 0 or any(x.shape != q1.shape for x in (k1,q2,k2)):
        raise ValueError("lambda parameters must have equal nonempty vector shapes")
    if not math.isfinite(lambda_init):
        raise ValueError("lambda_init must be finite")
    return (q1*k1).sum().exp() + (q2*k2).sum().exp() + lambda_init


def differential_attention(q1: Tensor, k1: Tensor, q2: Tensor, k2: Tensor,
                           value: Tensor, coefficient, causal: bool = True) -> Tensor:
    """Projected single/multiple heads, before head normalization and output projection."""
    if q1.ndim != 4 or q1.numel() == 0 or any(x.shape != q1.shape for x in (k1,q2,k2)):
        raise ValueError("Q/K must have equal nonempty [B,H,T,D] shapes")
    if value.shape != (*q1.shape[:-1], 2*q1.shape[-1]):
        raise ValueError("V must have shape [B,H,T,2D]")
    if isinstance(coefficient, Tensor) and coefficient.ndim != 0:
        raise ValueError("coefficient must be scalar")
    if not math.isfinite(float(coefficient.detach() if isinstance(coefficient, Tensor) else coefficient)):
        raise ValueError("coefficient must be finite")
    scores1 = (q1 @ k1.transpose(-1,-2)) / math.sqrt(q1.shape[-1])
    scores2 = (q2 @ k2.transpose(-1,-2)) / math.sqrt(q1.shape[-1])
    if causal:
        future = torch.ones(q1.shape[-2],q1.shape[-2],dtype=torch.bool,device=q1.device).triu(1)
        scores1 = scores1.masked_fill(future,float('-inf'))
        scores2 = scores2.masked_fill(future,float('-inf'))
    weights = scores1.softmax(dim=-1) - coefficient * scores2.softmax(dim=-1)
    return weights @ value
