import torch
from torch import Tensor

def grpo_advantages(rewards: Tensor, group_size: int, scale_std: bool=True, eps: float=0.0001) -> Tensor:
    """rewards [B*G] contains adjacent groups of G answers to each prompt.
    Use sample std (correction=1) plus eps when scale_std=True.
    Return [B*G]; G must be >=2 and divide the reward count."""
    raise NotImplementedError()

def grpo_loss(logp: Tensor, old_logp: Tensor, ref_logp: Tensor, advantages: Tensor, mask: Tensor, clip_eps: float=0.2, beta: float=0.04) -> tuple[Tensor, dict]:
    """logp/old_logp/ref_logp and binary mask are [N,L]; advantages [N].
    Freeze old/ref snapshots. Average effective tokens within each response,
    then average responses. Empty responses contribute zero. Return scalar
    loss and detached kl_mean, clipfrac, ratio_mean, averaged over valid tokens.
    clipfrac counts abs(ratio-1)>clip_eps."""
    raise NotImplementedError()
