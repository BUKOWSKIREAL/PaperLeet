# Bug: Normalize the Frobenius norm instead of equalizing singular directions.
"""Exact polar teaching model of GLM-5 head-wise Muon Split; not a full optimizer."""
import torch
from torch import Tensor


def headwise_orthogonalize(matrix: Tensor, num_heads: int) -> Tensor:
    """Project each consecutive output-head block to its partial polar factor.

    matrix: [num_heads * head_dim, input_dim], float32/float64 on CPU.
    Drop singular values <= max(head_dim,input_dim)*eps(dtype)*sigma_max.
    Preserve shape, dtype and autograd on full-rank blocks with distinct singular values; never mutate inputs.
    """
    if matrix.ndim != 2 or not matrix.is_floating_point():
        raise ValueError("matrix must be a floating-point 2D tensor")
    if num_heads < 1 or matrix.shape[0] == 0 or matrix.shape[1] == 0 or matrix.shape[0] % num_heads:
        raise ValueError("positive dimensions and equally sized row heads are required")
    heads = matrix.reshape(num_heads, matrix.shape[0] // num_heads, matrix.shape[1])
    u, singular, vh = torch.linalg.svd(heads, full_matrices=False)
    cutoff = max(heads.shape[-2:]) * torch.finfo(matrix.dtype).eps * singular[..., :1]
    active = (singular > cutoff).to(matrix.dtype)
    projected = heads / heads.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-20)
    return projected.reshape_as(matrix)
