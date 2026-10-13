"""Differentiable Gaussian FD and stopped-denominator representation normalization."""
import math
import torch
from torch import Tensor


def _check_gaussian(mean, covariance):
    if mean.ndim != 1 or mean.numel() < 1 or mean.dtype not in (torch.float32, torch.float64):
        raise ValueError("mean must be a nonempty float32/float64 vector")
    if covariance.shape != (mean.numel(), mean.numel()) or covariance.dtype != mean.dtype or covariance.device != mean.device:
        raise ValueError("covariance must be matching (D,D)")
    if not torch.isfinite(mean).all() or not torch.isfinite(covariance).all():
        raise ValueError("Gaussian statistics must be finite")
    tolerance = 50 * torch.finfo(mean.dtype).eps * max(1., covariance.detach().abs().max().item())
    if not torch.allclose(covariance, covariance.T, atol=tolerance, rtol=0):
        raise ValueError("covariance must be symmetric")
    if torch.linalg.eigvalsh(covariance.detach()).min() < -tolerance:
        raise ValueError("covariance must be positive semidefinite")


def _check_pair(mean, covariance, reference_mean, reference_covariance):
    _check_gaussian(mean, covariance)
    _check_gaussian(reference_mean, reference_covariance)
    if reference_mean.shape != mean.shape or reference_mean.dtype != mean.dtype or reference_mean.device != mean.device:
        raise ValueError("reference and generated statistics must match")


def _check_normalization(distances, weights, eps):
    if distances.ndim != 1 or distances.numel() < 1 or distances.dtype not in (torch.float32, torch.float64):
        raise ValueError("distances must be a nonempty float32/float64 vector")
    if weights.shape != distances.shape or weights.dtype != distances.dtype or weights.device != distances.device:
        raise ValueError("weights must match distances")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not torch.isfinite(distances).all() or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("distances and nonnegative weights must be finite")
    if (distances <= -eps).any():
        raise ValueError("normalization denominator must be positive")


def frechet_distance(mean: Tensor, covariance: Tensor, reference_mean: Tensor,
                     reference_covariance: Tensor) -> Tensor:
    """PSD covariances support forward values; gradients require BOTH covariances SPD."""
    _check_pair(mean, covariance, reference_mean, reference_covariance)
    real_mean, real_cov = reference_mean.detach(), reference_covariance.detach()
    values, vectors = torch.linalg.eigh(real_cov)
    real_sqrt = (vectors * values.clamp_min(0).sqrt()) @ vectors.T
    sandwich = real_sqrt @ covariance @ real_sqrt
    sandwich = (sandwich + sandwich.T) / 2
    eigenvalues = torch.linalg.eigvalsh(sandwich).clamp_min(0)
    cross_trace = eigenvalues.sqrt().sum()
    return (mean - real_mean).square().sum() + covariance.trace() + real_cov.trace() - 2 * cross_trace


def normalized_fd_loss(distances: Tensor, weights: Tensor, eps: float = .01) -> Tensor:
    """Sum representation losses with a stopped denominator and fixed weights."""
    _check_normalization(distances, weights, eps)
    return (weights.detach() * distances / (distances.detach() + eps)).sum()
