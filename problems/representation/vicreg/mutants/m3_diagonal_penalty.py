# Bug: 把协方差对角线也加入去冗余惩罚
"""VICReg's three losses, using the official single-process normalization."""
import math
import torch
from torch import Tensor


def _check_pair(x, y, eps):
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or x.dtype not in (torch.float32, torch.float64):
        raise ValueError("x must be float32/float64 (B,D), B>=2, D>=1")
    if y.shape != x.shape or y.dtype != x.dtype or y.device != x.device:
        raise ValueError("x and y must have the same shape, dtype and device")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("views must be finite")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


def _check_weights(sim_coeff, std_coeff, cov_coeff):
    if any(not math.isfinite(v) or v < 0 for v in (sim_coeff, std_coeff, cov_coeff)):
        raise ValueError("coefficients must be finite and nonnegative")


def vicreg_terms(x: Tensor, y: Tensor, eps: float = 1e-4) -> tuple[Tensor, Tensor, Tensor]:
    """Return scalar invariance, variance and covariance terms, in that order."""
    _check_pair(x, y, eps)
    invariance = (x - y).square().mean()
    xc, yc = x - x.mean(0), y - y.mean(0)
    std_x = (xc.square().sum(0) / (x.shape[0] - 1) + eps).sqrt()
    std_y = (yc.square().sum(0) / (y.shape[0] - 1) + eps).sqrt()
    variance = ((1 - std_x).relu().mean() + (1 - std_y).relu().mean()) / 2
    cov_x, cov_y = xc.T @ xc / (x.shape[0] - 1), yc.T @ yc / (y.shape[0] - 1)
    off_diagonal = torch.ones((x.shape[1], x.shape[1]), dtype=torch.bool, device=x.device)
    covariance = (cov_x[off_diagonal].square().sum() + cov_y[off_diagonal].square().sum()) / x.shape[1]
    return invariance, variance, covariance


def vicreg_loss(x: Tensor, y: Tensor, sim_coeff: float = 25., std_coeff: float = 25.,
                cov_coeff: float = 1., eps: float = 1e-4) -> Tensor:
    """Keep gradients in both views and combine the three scalar terms."""
    _check_weights(sim_coeff, std_coeff, cov_coeff)
    invariance, variance, covariance = vicreg_terms(x, y, eps)
    return sim_coeff * invariance + std_coeff * variance + cov_coeff * covariance
