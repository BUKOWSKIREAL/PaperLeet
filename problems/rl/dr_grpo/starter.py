import torch
from torch import Tensor
from problems.rl.grpo.solution import (token_objective, masked_logprobs, token_statistics,
    grpo_loss as grpo_reference_loss, grpo_advantages as grpo_reference_advantages)


# Required: implement the core functions below.
def drgrpo_advantages(rewards: Tensor, group_size: int) -> Tensor:
    """rewards [B*G] groups are adjacent, G>=2 and divides the length.
    Return group-centered [B*G] rewards without std scaling."""
    raise NotImplementedError()

def drgrpo_loss(logp: Tensor, old_logp: Tensor, ref_logp: Tensor, advantages: Tensor, mask: Tensor, max_len: int, clip_eps: float=0.2, beta: float=0.04) -> tuple[Tensor, dict]:
    """logp/old_logp/ref_logp/mask: [N,L]; advantages [N].
    Normalize masked token objective by N times externally fixed max_len.
    Return scalar loss and detached valid-token means kl_mean, clipfrac,
    ratio_mean. Shared token_objective returns (objective, ratio, k3) [N,L],
    with objective already including the negative KL penalty."""
    raise NotImplementedError()
