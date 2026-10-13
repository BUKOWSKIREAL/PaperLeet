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
    raise NotImplementedError("Implement maxk_estimate")


def pkpo_weights(rewards: Tensor, k: int) -> Tensor:
    """Eq.33 sloo-minus-one coefficients, in original sample order, detached."""
    _check(rewards, k)
    raise NotImplementedError("Implement pkpo_weights")


def pkpo_loss(log_probs: Tensor, rewards: Tensor, k: int) -> Tensor:
    """Negative policy gradient surrogate; weights already include sample normalization."""
    if log_probs.shape != rewards.shape:
        raise ValueError("log_probs and rewards must have matching shapes")
    raise NotImplementedError("Implement pkpo_loss")
