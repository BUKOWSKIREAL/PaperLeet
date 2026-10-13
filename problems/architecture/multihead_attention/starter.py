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
    """Required: project, split heads, attend, concatenate, and output-project.
    query [B,Q,D], key/value [B,K,D], all projection matrices [D,D].
    Matrix convention is x @ w (paper layout, not nn.Linear.weight layout).
    D is divisible by num_heads. Boolean mask broadcasts to [B,H,Q,K],
    True means visible. Return output [B,Q,D] and weights [B,H,Q,K].
    Preserve autograd, dtype and device; no bias/dropout; fully masked rows zero.
    The attention kernel above is supplied and may be used unchanged.
    """
    raise NotImplementedError()
