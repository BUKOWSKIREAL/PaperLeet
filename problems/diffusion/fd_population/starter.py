"""FD-loss population moments, with gradients restricted to the current batch."""
import math
import torch
from torch import Tensor


def _check_features(features):
    if features.ndim != 2 or min(features.shape) < 1 or features.dtype not in (torch.float32, torch.float64):
        raise ValueError("features must be a nonempty float32/float64 (B,D) tensor")
    if not torch.isfinite(features).all():
        raise ValueError("features must be finite")


def _check_history(features, old_mean, old_second, beta):
    _check_features(features)
    d = features.shape[1]
    if old_mean.shape != (d,) or old_second.shape != (d, d):
        raise ValueError("history must have shapes (D,) and (D,D)")
    for value in (old_mean, old_second):
        if value.dtype != features.dtype or value.device != features.device or not torch.isfinite(value).all():
            raise ValueError("finite history must match feature dtype/device")
    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be finite and in [0,1]")


def batch_moments(features: Tensor) -> tuple[Tensor, Tensor]:
    'Return mean and RAW second moment E[xx^T], divided by B.'
    _check_features(features)
    raise NotImplementedError("Implement batch_moments")


def ema_population_stats(features: Tensor, old_mean: Tensor, old_second: Tensor,
                         beta: float) -> tuple[Tensor, Tensor, Tensor]:
    'Return differentiable mean, raw second moment and population covariance.'
    _check_history(features, old_mean, old_second, beta)
    raise NotImplementedError("Implement ema_population_stats")
