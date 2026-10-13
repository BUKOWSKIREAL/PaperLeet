"""Tiny exact ROE solver; enumerate breakpoints instead of production sorting kernels."""
import math
import torch
from torch import Tensor


def _check_vectors(x, y, weight, trunc):
    if x.ndim != 1 or x.numel() == 0 or y.shape != x.shape or weight.shape != x.shape:
        raise ValueError("expected nonempty equal-shaped vectors")
    if not all(t.is_floating_point() and torch.isfinite(t).all() for t in (x, y, weight)):
        raise ValueError("inputs must be finite floating tensors")
    if any(t.dtype != x.dtype or t.device != x.device for t in (y, weight)) or (weight < 0).any():
        raise ValueError("matching dtype/device and nonnegative weights required")
    if trunc is not None and (not math.isfinite(trunc) or trunc <= 0):
        raise ValueError("trunc must be None or finite positive")


@torch.no_grad()
def optimal_scale_1d(x: Tensor, y: Tensor, weight: Tensor,
                     trunc: float | None = None) -> tuple[Tensor, Tensor]:
    """Minimize sum min(weight * abs(scale*x-y), trunc); return scale and loss."""
    _check_vectors(x, y, weight, trunc)
    raise NotImplementedError("Implement the paper core")


def _check_points(prediction, target, mask, trunc):
    if prediction.ndim != 2 or prediction.shape[1] != 3 or target.shape != prediction.shape:
        raise ValueError("point maps must be (N, 3)")
    if mask.dtype != torch.bool or mask.shape != prediction.shape[:1] or not mask.any():
        raise ValueError("mask must select at least one point")
    if mask.device != prediction.device:
        raise ValueError("mask device must match")
    p, q = prediction[mask], target[mask]
    _check_vectors(p.flatten(), q.flatten(), torch.ones_like(p).flatten(), trunc)
    if (q[:, 2] <= 0).any():
        raise ValueError("valid target depths must be positive")
    return p, q


@torch.no_grad()
def roe_align(prediction: Tensor, target: Tensor, mask: Tensor,
              trunc: float | None = None) -> tuple[Tensor, Tensor]:
    """Exact inverse-depth-weighted L1 alignment with a shared scale and z-only shift."""
    p, q = _check_points(prediction, target, mask, trunc)
    raise NotImplementedError("Implement the paper core")
