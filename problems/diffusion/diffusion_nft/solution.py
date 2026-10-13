"""DiffusionNFT Eq. (5), with a frozen rollout policy and bounded rewards."""
import math
import torch
from torch import Tensor


def _check_velocity(current, old, beta):
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if current.ndim < 2 or current.numel() == 0 or not current.is_floating_point():
        raise ValueError("velocity needs a nonempty batch and feature dimensions")
    if old.shape != current.shape or old.dtype != current.dtype or old.device != current.device:
        raise ValueError("old and current velocity must match")
    if not torch.isfinite(current).all() or not torch.isfinite(old).all():
        raise ValueError("velocities must be finite")


def implicit_velocities(current: Tensor, old: Tensor, beta: float) -> tuple[Tensor, Tensor]:
    """Return the implicit positive and negative policies; only current receives gradients."""
    _check_velocity(current, old, beta)
    frozen = old.detach()
    positive = (1 - beta) * frozen + beta * current
    negative = (1 + beta) * frozen - beta * current
    return positive, negative


def nft_loss(current: Tensor, old: Tensor, target: Tensor, rewards: Tensor,
             beta: float) -> Tensor:
    """Mean over examples of reward-weighted squared L2 norms, summed over features."""
    _check_velocity(current, old, beta)
    if target.shape != current.shape or target.dtype != current.dtype or target.device != current.device:
        raise ValueError("target velocity must match current")
    if rewards.shape != current.shape[:1] or rewards.dtype != current.dtype or rewards.device != current.device:
        raise ValueError("rewards must be a matching (B,) tensor")
    if not torch.isfinite(target).all() or not torch.isfinite(rewards).all() or ((rewards < 0) | (rewards > 1)).any():
        raise ValueError("finite targets and rewards in [0,1] required")
    positive, negative = implicit_velocities(current, old, beta)
    v, r = target.detach(), rewards.detach()
    positive_error = (positive - v).flatten(1).square().sum(-1)
    negative_error = (negative - v).flatten(1).square().sum(-1)
    return (r * positive_error + (1 - r) * negative_error).mean()
