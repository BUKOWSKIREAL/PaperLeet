import math
import torch
from torch import Tensor


def online_softmax_update(scores: Tensor, values: Tensor, row_max: Tensor,
                          normalizer: Tensor, numerator: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Required: merge finite scores [Q,K] and values [K,Dv] into online state.
    row_max [Q] starts at -inf, normalizer [Q] and numerator [Q,Dv] at zero.
    Rescale old mass AND weighted values whenever the row maximum changes.
    Return new maximum, unnormalized sum, unnormalized weighted sum.
    Preserve autograd and inputs. All blocks have at least one key."""
    raise NotImplementedError()


def block_attention(q: Tensor, k: Tensor, v: Tensor, block_size: int = 16) -> tuple[Tensor, Tensor]:
    """Provided: tile traversal, no dropout/masks; returns output and per-row log-sum-exp."""
    outputs, logsumexp = [], []
    for query_start in range(0, q.shape[0], block_size):
        queries = q[query_start:query_start + block_size]
        row_max = q.new_full((len(queries),), -torch.inf)
        normalizer = q.new_zeros(len(queries))
        numerator = q.new_zeros((len(queries), v.shape[-1]))
        for key_start in range(0, k.shape[0], block_size):
            scores = (queries @ k[key_start:key_start + block_size].T) / math.sqrt(q.shape[-1])
            values = v[key_start:key_start + block_size]
            row_max, normalizer, numerator = online_softmax_update(scores, values, row_max, normalizer, numerator)
        outputs.append(numerator / normalizer[:, None])
        logsumexp.append(row_max + normalizer.log())
    return torch.cat(outputs), torch.cat(logsumexp)
