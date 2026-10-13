# Bug: 旧 weighted sum 未随新最大值缩放
import math
import torch
from torch import Tensor


def online_softmax_update(scores: Tensor, values: Tensor, row_max: Tensor,
                          normalizer: Tensor, numerator: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Merge finite score block [Q,K] into state [Q], [Q], [Q,Dv]."""
    new_max = torch.maximum(row_max, scores.max(-1).values)
    rescale = torch.exp(row_max - new_max)
    probabilities = torch.exp(scores - new_max[:, None])
    new_normalizer = rescale * normalizer + probabilities.sum(-1)
    new_numerator = numerator + probabilities @ values
    return new_max, new_normalizer, new_numerator


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
