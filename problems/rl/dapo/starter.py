import torch
from torch import Tensor

def dynamic_group_mask(rewards: Tensor) -> Tensor:
    """[prompts,group] binary rewards (0/1 or -1/1) -> informative prompts."""
    raise NotImplementedError('Implement dynamic_group_mask')

def dapo_loss(logp: Tensor, old_logp: Tensor, advantages: Tensor, mask: Tensor, clip_low: float=0.2, clip_high: float=0.28) -> Tensor:
    """Asymmetric clipping, averaged over all retained response tokens."""
    raise NotImplementedError('Implement dapo_loss')
