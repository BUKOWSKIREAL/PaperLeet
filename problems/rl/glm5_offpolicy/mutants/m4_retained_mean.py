# Bug: Renormalize by retained tokens and change the original valid-token objective.
"""GLM-5 Eq. (3-5) teaching loss with explicit detached score-function weights."""
import math
import torch
from torch import Tensor


def masked_is_weights(logp: Tensor, rollout_logp: Tensor, mask: Tensor,
                      clip_low: float = .2, clip_high: float = .2) -> Tensor:
    """Detached token weights r inside STRICT (1-low,1+high), zero elsewhere.

    Same-shaped [B,L] log-probabilities and binary mask. Compare the computed
    ratio in its native dtype; torch.where safely discards detached infinities.
    """
    if logp.ndim != 2 or rollout_logp.shape != logp.shape or mask.shape != logp.shape:
        raise ValueError("log probabilities and mask must have the same [B,L] shape")
    if not 0 < clip_low < 1 or not math.isfinite(clip_high) or clip_high <= 0:
        raise ValueError("require 0 < clip_low < 1 and finite clip_high > 0")
    valid = mask.bool()
    delta = torch.where(valid, logp.detach() - rollout_logp.detach(), torch.zeros_like(logp))
    ratio = delta.exp()
    retained = valid & (ratio > 1 - clip_low) & (ratio < 1 + clip_high)
    return torch.where(retained, ratio, torch.zeros_like(ratio))


def offpolicy_loss(logp: Tensor, rollout_logp: Tensor, advantages: Tensor, mask: Tensor,
                   clip_low: float = .2, clip_high: float = .2) -> Tensor:
    """Negative weighted REINFORCE, normalized over ORIGINAL valid tokens.

    advantages: [B], frozen. logp/rollout_logp/mask: [B,L]. Current logp is
    differentiated only in its direct score-function factor, never in weights.
    An empty mask yields a differentiable zero. Inputs are not mutated.
    """
    weights = masked_is_weights(logp, rollout_logp, mask, clip_low, clip_high)
    if advantages.ndim != 1 or advantages.shape[0] != logp.shape[0]:
        raise ValueError("advantages must have shape [B]")
    valid = mask.bool()
    scores = torch.where(valid, logp, torch.zeros_like(logp))
    return -(weights * advantages.detach()[:, None] * scores).sum() / (weights != 0).sum().clamp_min(1)
