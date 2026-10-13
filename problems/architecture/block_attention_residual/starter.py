import torch
from torch import Tensor


def assemble_sources(embedding: Tensor, completed_blocks: list[Tensor],
                     partial_block: Tensor | None = None) -> Tensor:
    """Required: stack [embedding, *completed_blocks, optional partial_block].
    Each tensor is [B,T,D]. Return [N,B,T,D], embedding first. A missing partial
    block is omitted; a supplied all-zero partial block is a real source.
    Do not mutate the caller's list or tensors. Completed blocks exclude embedding.
    """
    raise NotImplementedError()


def depth_attention(sources: Tensor, query: Tensor, norm_weight: Tensor,
                    eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Required: keys-only RMSNorm, depth softmax, original-values aggregation.
    sources [N,B,T,D], learned query [D], RMSNorm weight [D], eps > 0.
    Keys = sources / sqrt(mean(sources**2, -1) + eps) * norm_weight.
    No sqrt(D) score scaling. Return mixed [B,T,D], weights [N,B,T].
    Preserve dtype/device, gradients, inputs; support N=1 and noncontiguous tensors.
    The full backbone, residual accumulation and layer projections are outside this task.
    """
    raise NotImplementedError()
