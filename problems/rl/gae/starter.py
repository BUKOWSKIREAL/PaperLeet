import torch
from torch import Tensor

def compute_gae(rewards: Tensor, values: Tensor, next_values: Tensor, terminated: Tensor, truncated: Tensor, gamma: float=0.99, lam: float=0.95) -> tuple[Tensor, Tensor]:
    """Compute advantages and returns, both [T,N], preserving rewards dtype/device.
    rewards/values/next_values: floating [T,N]. next_values uses the true
    pre-reset final state. terminated/truncated: bool [T,N]. Termination
    removes bootstrap; both endings stop propagation. returns = advantages + values."""
    raise NotImplementedError()
