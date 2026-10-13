# Bug: 仍然按序列长度归一
import torch
from torch import Tensor
from problems.rl.grpo.solution import token_objective, grpo_loss as grpo_reference_loss, grpo_advantages as grpo_reference_advantages


def drgrpo_advantages(rewards: Tensor, group_size: int) -> Tensor:
    """[B*G] group-centered rewards, without standard deviation scaling."""
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError('group_size must be >= 2 and divide the reward count')
    grouped = rewards.reshape(-1, group_size)
    return (grouped - grouped.mean(-1, keepdim=True)).reshape_as(rewards)


def drgrpo_loss(logp: Tensor, old_logp: Tensor, ref_logp: Tensor, advantages: Tensor,
                mask: Tensor, max_len: int, clip_eps: float = 0.2, beta: float = 0.04) -> tuple[Tensor, dict]:
    """[N,L] tokens normalized by N * max_len, an external generation limit."""
    if max_len <= 0:
        raise ValueError('max_len must be positive')
    active = mask.bool()
    logp, old_logp, ref_logp = [torch.where(active, x, torch.zeros_like(x)) for x in (logp, old_logp, ref_logp)]
    objective, ratio, kl = token_objective(logp, old_logp, ref_logp, advantages, clip_eps, beta)
    loss = -((objective * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
    with torch.no_grad():
        denom = mask.sum().clamp_min(1)
        stats = dict(kl_mean=(kl * mask).sum() / denom,
                     clipfrac=(((ratio - 1).abs() > clip_eps) * mask).sum() / denom,
                     ratio_mean=(ratio * mask).sum() / denom)
    return loss, stats
