# Bug: 先标准化投影使目标的均值与尺度约束失效
"""LeJEPA SIGReg with supplied fixed directions and positive quadrature nodes."""
import torch
from torch import Tensor


def _check_projected(projected, t):
    if projected.ndim != 2 or min(projected.shape) < 1 or projected.dtype not in (torch.float32, torch.float64):
        raise ValueError("projected must be nonempty float32/float64 (N,M)")
    if t.ndim != 1 or t.numel() < 2 or t.dtype != projected.dtype or t.device != projected.device:
        raise ValueError("t must be a matching vector with at least two nodes")
    if not torch.isfinite(projected).all() or not torch.isfinite(t).all():
        raise ValueError("inputs must be finite")
    if t[0] != 0 or not (t[1:] > t[:-1]).all():
        raise ValueError("t must start at zero and increase strictly")


def _check_features(features, directions, t):
    _check_projected(features, t)
    if directions.ndim != 2 or directions.shape[0] != features.shape[1] or directions.shape[1] < 1:
        raise ValueError("directions must have shape (D,M), M>=1")
    if directions.dtype != features.dtype or directions.device != features.device or not torch.isfinite(directions).all():
        raise ValueError("directions must match feature dtype/device and be finite")
    tolerance = 100 * torch.finfo(features.dtype).eps
    if not torch.allclose(directions.norm(dim=0), torch.ones_like(directions[0]), atol=tolerance, rtol=tolerance):
        raise ValueError("each direction must have unit Euclidean norm")


def epps_pulley(projected: Tensor, t: Tensor) -> Tensor:
    """Return one N-scaled Epps-Pulley statistic per projection; t is fixed."""
    _check_projected(projected, t)
    nodes = t.detach()
    projected = (projected - projected.mean(0)) / (projected.std(0, correction=0) + 1e-4)
    phase = projected.unsqueeze(-1) * nodes
    phi = torch.exp(-nodes.square() / 2)
    error = (phase.cos().mean(0) - phi).square() + phase.sin().mean(0).square()
    return 2 * projected.shape[0] * torch.trapezoid(error * phi, nodes, dim=-1)


def sigreg_loss(features: Tensor, directions: Tensor, t: Tensor) -> Tensor:
    """Mean over supplied unit directions; no centering, whitening or feature detach."""
    _check_features(features, directions, t)
    projected = features @ directions.detach()
    return epps_pulley(projected, t).mean()
