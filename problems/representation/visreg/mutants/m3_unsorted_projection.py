# BUG: Sample order is wrongly used as quantile order.
"""VISReg Algorithm 1 with explicit projections and the epsilon from Eq. (2)."""
import math
import torch
from torch import Tensor


def _features(x, ndim):
    if not isinstance(x, Tensor) or x.ndim != ndim or min(x.shape) < 1 or x.dtype not in (torch.float32, torch.float64) or not torch.isfinite(x).all():
        raise ValueError('expected finite, nonempty float32/float64 features')


def _check_terms(z, directions, eps):
    _features(z, 2); _features(directions, 2)
    if z.shape[0] < 2 or directions.shape[0] != z.shape[1] or directions.dtype != z.dtype or directions.device != z.device:
        raise ValueError('need N>=2, directions [D,K] and matching dtype/device')
    if (directions.square().sum(0) == 0).any() or not math.isfinite(eps) or eps <= 0:
        raise ValueError('nonzero directions and positive finite epsilon required')


def visreg_terms(z: Tensor, directions: Tensor, eps: float = 1e-4) -> tuple[Tensor, Tensor, Tensor]:
    """Return (scale, shape, center), all scalars; supplied directions are fixed."""
    _check_terms(z, directions, eps)
    mu = z.mean(0)
    centered = z - mu
    std = centered.std(0, correction=0)
    scale = (std - 1).square().mean()
    normalized = centered / (std.detach() + eps)
    unit = directions.detach() / directions.detach().norm(dim=0, keepdim=True)
    projected = (normalized @ unit)
    ranks = torch.arange(1, z.shape[0] + 1, dtype=z.dtype, device=z.device) / (z.shape[0] + 1)
    quantiles = math.sqrt(2.) * torch.erfinv(2 * ranks - 1)
    shape = (projected - quantiles[:, None]).square().mean()
    center = mu.square().mean()
    return scale, shape, center


def view_invariance(views: Tensor, num_global: int = 2) -> Tensor:
    """Mean squared error of all [V,N,D] views from the first G views' mean."""
    _features(views, 3)
    if not isinstance(num_global, int) or isinstance(num_global, bool) or not 1 <= num_global <= views.shape[0]:
        raise ValueError('num_global must be an integer in [1,V]')
    global_mean = views[:num_global].mean(0)
    return (views - global_mean).square().mean()
