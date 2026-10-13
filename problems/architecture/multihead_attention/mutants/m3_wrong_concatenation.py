# Bug: Flatten heads before restoring the query axis.
import math
import torch
from torch import Tensor


def attention_kernel(q: Tensor, k: Tensor, v: Tensor,
                     mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Provided: [B,H,Q,D] attention; True means visible, empty rows are zero."""
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if mask is None:
        weights = scores.softmax(-1)
    else:
        allowed = torch.broadcast_to(mask, scores.shape)
        scores = scores.masked_fill(~allowed, -torch.inf)
        scores = torch.where(allowed.any(-1, keepdim=True), scores, torch.zeros_like(scores))
        weights = scores.softmax(-1).masked_fill(~allowed, 0.)
    return weights @ v, weights


def multi_head_attention(query: Tensor, key: Tensor, value: Tensor,
                         w_q: Tensor, w_k: Tensor, w_v: Tensor, w_o: Tensor,
                         num_heads: int, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Paper-layout projections [D,D], inputs [B,Q/K,D]; return [B,Q,D], [B,H,Q,K]."""
    batch, queries, width = query.shape
    keys, head_width = key.shape[1], width // num_heads
    q = (query @ w_q).reshape(batch, queries, num_heads, head_width).transpose(1, 2)
    k = (key @ w_k).reshape(batch, keys, num_heads, head_width).transpose(1, 2)
    v = (value @ w_v).reshape(batch, keys, num_heads, head_width).transpose(1, 2)
    heads, weights = attention_kernel(q, k, v, mask)
    concatenated = heads.reshape(batch, queries, width)
    return concatenated @ w_o, weights
