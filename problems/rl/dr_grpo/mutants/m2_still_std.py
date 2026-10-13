# Bug: 仍然除以标准差
import torch
from torch import Tensor
from problems.rl.grpo.solution import (token_objective, masked_logprobs, token_statistics,
    grpo_loss as grpo_reference_loss, grpo_advantages as grpo_reference_advantages)


def drgrpo_advantages(rewards: Tensor, group_size: int) -> Tensor:
    """[B*G] group-centered rewards, without standard deviation scaling."""
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError('group_size must be >= 2 and divide the reward count')
    grouped = rewards.reshape(-1, group_size)
    return ((grouped - grouped.mean(-1, keepdim=True)) / (grouped.std(-1, keepdim=True) + 1e-4)).reshape_as(rewards)


def drgrpo_loss(logp: Tensor, old_logp: Tensor, ref_logp: Tensor, advantages: Tensor,
                mask: Tensor, max_len: int, clip_eps: float = 0.2, beta: float = 0.04) -> tuple[Tensor, dict]:
    """[N,L] tokens normalized by N * max_len, an external generation limit."""
    if max_len <= 0:
        raise ValueError('max_len must be positive')
    logp, old_logp, ref_logp = masked_logprobs(logp, old_logp, ref_logp, mask)
    objective, ratio, kl = token_objective(logp, old_logp, ref_logp, advantages, clip_eps, beta)
    loss = -(objective * mask).sum() / (logp.shape[0] * max_len)
    return loss, token_statistics(ratio, kl, mask, clip_eps)
