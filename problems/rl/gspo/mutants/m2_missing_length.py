# Intentional bug: Omit response-length normalization.
import torch
from torch import Tensor


def sequence_ratio(logp: Tensor, old_logp: Tensor, mask: Tensor) -> Tensor:
    """Length-normalized sequence importance weights, [B,L] -> [B]."""
    delta = torch.where(mask.bool(), logp - old_logp.detach(), torch.zeros_like(logp))
    lengths = mask.sum(-1).clamp_min(1)
    return (delta.sum(-1)).exp()


def gspo_loss(logp: Tensor, old_logp: Tensor, advantages: Tensor,
              mask: Tensor, clip_eps: float = 0.2) -> Tensor:
    """Sequence clipping; empty responses contribute zero to the batch mean."""
    ratio = sequence_ratio(logp, old_logp, mask)
    advantage = advantages.detach()
    objective = torch.minimum(ratio * advantage,
                              ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage)
    return -(objective * mask.bool().any(-1)).mean()
