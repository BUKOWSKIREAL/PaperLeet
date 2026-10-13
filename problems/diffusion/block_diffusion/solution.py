import math
import torch
from torch import Tensor


def block_diffusion_mask(length: int, block_size: int, device=None) -> Tensor:
    """Allowed edges for concatenated [noisy tokens, clean tokens]."""
    if type(length) is not int or type(block_size) is not int or length<=0 or block_size<=0:
        raise ValueError("length and block_size must be positive integers")
    blocks=torch.arange(length,device=device)//block_size
    same=blocks[:,None]==blocks[None,:]
    earlier=blocks[:,None]>blocks[None,:]
    previous_or_same=blocks[:,None]>=blocks[None,:]
    return torch.cat((torch.cat((same,earlier),dim=1),
                      torch.cat((torch.zeros_like(same),previous_or_same),dim=1)),dim=0)


def block_diffusion_attention(query: Tensor, key: Tensor, value: Tensor, block_size: int) -> Tensor:
    """Dense CPU attention using the vectorized block-diffusion training mask."""
    if query.ndim!=4 or query.numel()==0 or key.shape!=query.shape or query.shape[-2]%2:
        raise ValueError("Q/K must have equal nonempty [B,H,2L,D] shapes")
    if value.ndim!=4 or value.shape[:-1]!=query.shape[:-1] or value.shape[-1]==0:
        raise ValueError("V must have shape [B,H,2L,Dv] with Dv>0")
    allowed=block_diffusion_mask(query.shape[-2]//2,block_size,query.device)
    scores=(query @ key.transpose(-1,-2))/math.sqrt(query.shape[-1])
    return scores.masked_fill(~allowed,float('-inf')).softmax(dim=-1) @ value
