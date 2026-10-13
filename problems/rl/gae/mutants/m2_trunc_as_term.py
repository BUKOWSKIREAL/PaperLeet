# Bug: 把时间截断当作终止
import torch
from torch import Tensor


def compute_gae(rewards: Tensor, values: Tensor, next_values: Tensor, terminated: Tensor,
                truncated: Tensor, gamma: float = 0.99, lam: float = 0.95) -> tuple[Tensor, Tensor]:
    """[T,N] tensors. Truncation bootstraps but stops the advantage recursion."""
    delta = rewards + gamma * (~(terminated | truncated)).to(rewards.dtype) * next_values - values
    done = terminated | truncated
    advantages = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        carry = delta[t] + gamma * lam * (~done[t]).to(rewards.dtype) * carry
        advantages[t] = carry
    returns = advantages + values
    return advantages, returns
