# BUG: Divide weights instead of compensating the input scale.
import torch
from torch import Tensor


def fake_quantize(x: Tensor, bits: int = 4) -> Tensor:
    """Provided per-tensor symmetric INT fake quantizer, round ties to even."""
    if type(bits) is not int or not 2 <= bits <= 8:
        raise ValueError("bits must be an integer from 2 to 8")
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax() / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return torch.round(x / scale).clamp(-qmax, qmax) * scale


def _check_smoothing(x, weight, scale):
    if x.ndim != 2 or weight.ndim != 2 or scale.ndim != 1:
        raise ValueError("expected x[B,M], weight[M,N], scale[M]")
    if x.shape[1] != weight.shape[0] or scale.shape[0] != weight.shape[0]:
        raise ValueError("input channel dimensions must match")
    if min(*x.shape, *weight.shape) < 1 or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("nonempty dimensions and finite positive scale required")


def _check_rank(weight, rank):
    if weight.ndim != 2 or type(rank) is not int or not 0 <= rank <= min(weight.shape):
        raise ValueError("rank must be an integer from 0 to min(M,N)")


def smooth_inputs(x: Tensor, weight: Tensor, scale: Tensor) -> tuple[Tensor, Tensor]:
    """Preserve XW by dividing X channels and multiplying W rows."""
    _check_smoothing(x, weight, scale)
    return x / scale, weight / scale[:, None]


def split_weight(weight: Tensor, rank: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return truncated-SVD factors L1[M,r], L2[r,N], and unquantized residual."""
    _check_rank(weight, rank)
    u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
    left = u[:, :rank] * singular[:rank]
    right = vh[:rank, :]
    residual = weight - left @ right
    return left, right, residual


def svdquant_linear(x: Tensor, weight: Tensor, scale: Tensor, rank: int,
                    bits: int = 4) -> Tensor:
    """High-precision low-rank branch plus fake-quantized activations/residual."""
    xs, ws = smooth_inputs(x, weight, scale)
    left, right, residual = split_weight(ws, rank)
    return (xs @ left) @ right + fake_quantize(xs, bits) @ fake_quantize(residual, bits)
