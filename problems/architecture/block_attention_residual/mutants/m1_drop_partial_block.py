# Bug: Omits the current partial block from the available depth sources.
import torch
from torch import Tensor


def assemble_sources(embedding: Tensor, completed_blocks: list[Tensor],
                     partial_block: Tensor | None = None) -> Tensor:
    """Stack embedding, completed blocks, optional current sum on leading depth axis."""
    sources = [embedding, *completed_blocks]
    if partial_block is not None:
        pass
    return torch.stack(sources, dim=0)


def depth_attention(sources: Tensor, query: Tensor, norm_weight: Tensor,
                    eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Sources [N,B,T,D]; query/scale [D]; return mixed [B,T,D], weights [N,B,T]."""
    keys = sources * torch.rsqrt(sources.square().mean(dim=-1, keepdim=True) + eps) * norm_weight
    logits = torch.einsum('nbtd,d->nbt', keys, query)
    weights = logits.softmax(dim=0)
    mixed = torch.einsum('nbt,nbtd->btd', weights, sources)
    return mixed, weights
