import torch
from torch import Tensor


def dynamic_group_mask(rewards: Tensor) -> Tensor:
    """[prompts,group] binary rewards (0/1 or -1/1) -> informative prompts."""
    if rewards.ndim != 2 or rewards.shape[-1] < 2:
        raise ValueError('rewards must have shape [prompts, group >= 2]')
    return rewards.max(-1).values > rewards.min(-1).values


def dapo_loss(logp: Tensor, old_logp: Tensor, advantages: Tensor, mask: Tensor,
              clip_low: float = 0.2, clip_high: float = 0.28) -> Tensor:
    """Asymmetric clipping, averaged over all retained response tokens."""
    delta = torch.where(mask.bool(), logp - old_logp.detach(), torch.zeros_like(logp))
    ratio = delta.exp()
    advantage = advantages.detach().unsqueeze(-1)
    objective = torch.minimum(ratio * advantage,
                              ratio.clamp(1 - clip_low, 1 + clip_high) * advantage)
    return -(objective * mask).sum() / mask.sum().clamp_min(1)
