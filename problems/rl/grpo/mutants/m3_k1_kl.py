# Bug: k1 逐 token 可为负
import torch
from torch import Tensor


def masked_logprobs(logp: Tensor, old_logp: Tensor, ref_logp: Tensor, mask: Tensor):
    """Provided helper: neutralize padding before exponential calculations."""
    active = mask.bool()
    return tuple(torch.where(active, x, torch.zeros_like(x)) for x in (logp, old_logp, ref_logp))


def token_statistics(ratio: Tensor, kl: Tensor, mask: Tensor, clip_eps: float) -> dict:
    """Provided helper: detached diagnostics averaged over valid tokens."""
    with torch.no_grad():
        denom = mask.sum().clamp_min(1)
        return dict(kl_mean=(kl * mask).sum() / denom,
                    clipfrac=(((ratio - 1).abs() > clip_eps) * mask).sum() / denom,
                    ratio_mean=(ratio * mask).sum() / denom)


def grpo_advantages(rewards: Tensor, group_size: int, scale_std: bool = True, eps: float = 1e-4) -> Tensor:
    """Adjacent groups [B*G], sample std (correction=1), G >= 2."""
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError('group_size must be >= 2 and divide the reward count')
    grouped = rewards.reshape(-1, group_size)
    centered = grouped - grouped.mean(-1, keepdim=True)
    scale = grouped.std(-1, keepdim=True, unbiased=True) + eps if scale_std else 1.0
    return (centered / scale).reshape_as(rewards)


def token_objective(logp: Tensor, old_logp: Tensor, ref_logp: Tensor, advantages: Tensor,
                    clip_eps: float = 0.2, beta: float = 0.04):
    """Shared ratio, clipping and k3 computation used by the Dr. GRPO module."""
    ratio = (logp - old_logp.detach()).exp()
    diff = ref_logp.detach() - logp
    kl = diff.exp() - diff - 1
    advantage = advantages.detach().unsqueeze(-1)
    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage) - beta * kl
    return objective, ratio, kl


def grpo_loss(logp: Tensor, old_logp: Tensor, ref_logp: Tensor, advantages: Tensor,
              mask: Tensor, clip_eps: float = 0.2, beta: float = 0.04) -> tuple[Tensor, dict]:
    """[N,L] token log-probs; mean tokens within each response, then responses."""
    logp, old_logp, ref_logp = masked_logprobs(logp, old_logp, ref_logp, mask)
    ratio = (logp - old_logp.detach()).exp()
    diff = ref_logp.detach() - logp
    kl = -diff
    advantage = advantages.detach().unsqueeze(-1)
    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage) - beta * kl
    lengths = mask.sum(-1).clamp_min(1)
    loss = -((objective * mask).sum(-1) / lengths).mean()
    return loss, token_statistics(ratio, kl, mask, clip_eps)
