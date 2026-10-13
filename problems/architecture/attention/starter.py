import math
import torch
from torch import Tensor


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor,
                                 mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Required: q [B,H,Q,D], k [B,H,K,D], v [B,H,K,Dv].
    Return output [B,H,Q,Dv], weights [B,H,Q,K]. Boolean mask broadcasts
    to weights; True means allowed. Fully masked queries return exact zeros
    and finite zero gradients. Preserve dtype/device; no dropout."""
    raise NotImplementedError()


def causal_mask(query_length: int, key_length: int, device=None) -> Tensor:
    """Provided: upper-left aligned [Q,K] mask; key j is visible iff j <= i."""
    return torch.arange(key_length, device=device)[None, :] <= torch.arange(query_length, device=device)[:, None]
