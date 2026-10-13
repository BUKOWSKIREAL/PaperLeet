# Bug: Mix sequence positions into the feature statistics.
import torch
from torch import Tensor


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    """Normalize the last feature dimension using population variance, then affine."""
    centered = x - x.mean(dim=tuple(range(max(0, x.ndim - 2), x.ndim)), keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    return centered * torch.rsqrt(variance + eps) * weight + bias


def post_norm_residual(x: Tensor, sublayer_output: Tensor, weight: Tensor,
                       bias: Tensor, eps: float = 1e-5) -> Tensor:
    """Original Transformer Add & Norm; sublayer output is supplied, dropout disabled."""
    return layer_norm(x + sublayer_output, weight, bias, eps)
