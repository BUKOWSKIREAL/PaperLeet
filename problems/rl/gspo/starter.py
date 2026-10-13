import torch
from torch import Tensor

def sequence_ratio(logp: Tensor, old_logp: Tensor, mask: Tensor) -> Tensor:
    """Length-normalized sequence importance weights, [B,L] -> [B]."""
    raise NotImplementedError('Implement sequence_ratio')

def gspo_loss(logp: Tensor, old_logp: Tensor, advantages: Tensor, mask: Tensor, clip_eps: float=0.2) -> Tensor:
    """Sequence clipping; empty responses contribute zero to the batch mean."""
    raise NotImplementedError('Implement gspo_loss')
