# BUG: Average weights that are already sample-normalized.
import math
import torch
from torch import Tensor


def _check(rewards, k):
    if rewards.ndim < 1 or rewards.shape[-1] == 0 or not rewards.is_floating_point():
        raise ValueError("rewards must be floating [..., N] with N > 0")
    if type(k) is not int or not 1 <= k <= rewards.shape[-1]:
        raise ValueError("k must be an integer in [1, N]")


def _comb(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def maxk_estimate(rewards: Tensor, k: int) -> Tensor:
    """U-statistic for expected max reward among k samples, along the last axis."""
    _check(rewards, k)
    values = rewards.sort(-1).values
    n = rewards.shape[-1]
    weights = rewards.new_tensor([_comb(i, k - 1) / _comb(n, k) for i in range(n)])
    return (values * weights).sum(-1)


def pkpo_weights(rewards: Tensor, k: int) -> Tensor:
    """Eq.33 sloo-minus-one coefficients, in original sample order, detached."""
    _check(rewards, k)
    raw = rewards.detach()
    n = raw.shape[-1]
    if k == 1:
        return raw / n
    values, order = raw.sort(-1)
    coefficients = raw.new_tensor([_comb(j, k - 2) / _comb(n, k) for j in range(n)])
    cumulative = coefficients.cumsum(-1) - coefficients
    weighted = values * coefficients
    previous = weighted.cumsum(-1) - weighted
    transformed = values * cumulative - previous
    return torch.zeros_like(raw).scatter(-1, order, transformed)


def pkpo_loss(log_probs: Tensor, rewards: Tensor, k: int) -> Tensor:
    """Negative policy gradient surrogate; weights already include sample normalization."""
    if log_probs.shape != rewards.shape:
        raise ValueError("log_probs and rewards must have matching shapes")
    weights = pkpo_weights(rewards, k)
    return -(log_probs * weights).mean()
