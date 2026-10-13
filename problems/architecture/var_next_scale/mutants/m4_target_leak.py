# BUG: Builds each input from its own target scale.
import torch
import torch.nn.functional as F
from torch import Tensor


def _validate_codes(token_maps, codebook, projections):
    if not token_maps or codebook.ndim != 2 or min(codebook.shape) == 0:
        raise ValueError("nonempty token maps and [V,C] codebook required")
    if any(t.ndim != 3 for t in token_maps):
        raise ValueError("token maps must have three dimensions")
    scales = [t.shape[-1] for t in token_maps]
    if scales[0] != 1 or any(a >= b for a, b in zip(scales, scales[1:])):
        raise ValueError("square scales must start at 1 and strictly increase")
    batch = token_maps[0].shape[0]
    for t, n in zip(token_maps, scales):
        if t.ndim != 3 or t.shape != (batch, n, n) or batch == 0 or t.dtype != torch.long:
            raise ValueError("tokens must be nonempty int64 [B,n,n]")
        if t.device != codebook.device or (t < 0).any() or (t >= len(codebook)).any():
            raise ValueError("token indices must be on the codebook device and in range")
    channels = codebook.shape[1]
    if projections.shape != (len(scales), channels, channels):
        raise ValueError("projections must have shape [K,C,C]")
    if projections.dtype != codebook.dtype or projections.device != codebook.device:
        raise ValueError("projection dtype/device must match codebook")
    return scales


def lift_codes(tokens: Tensor, codebook: Tensor, projection: Tensor, full_size: int) -> Tensor:
    """Provided: code lookup, bicubic upsampling, then a fixed channel projection."""
    codes = F.embedding(tokens, codebook).permute(0, 3, 1, 2)
    if codes.shape[-1] != full_size:
        codes = F.interpolate(codes, size=(full_size, full_size), mode="bicubic", align_corners=False)
    return torch.einsum("oc,bchw->bohw", projection, codes)


def scale_causal_mask(scales: list[int], device=None) -> Tensor:
    """Return [L,L] bool visibility: keys at the same or an earlier scale."""
    if not scales or any(type(n) is not int or n <= 0 for n in scales):
        raise ValueError("scales must contain positive integers")
    if scales[0] != 1 or any(a >= b for a, b in zip(scales, scales[1:])):
        raise ValueError("scales must start at 1 and strictly increase")
    levels = torch.cat([torch.full((n*n,), k, device=device) for k, n in enumerate(scales)])
    return levels[:, None] >= levels[None, :]


@torch.no_grad()
def next_scale_inputs(token_maps: list[Tensor], codebook: Tensor, projections: Tensor) -> Tensor:
    """Frozen tokenizer features for scales 2..K; start embedding is provided separately."""
    scales = _validate_codes(token_maps, codebook, projections)
    batch, channels, full = token_maps[0].shape[0], codebook.shape[1], scales[-1]
    cumulative = codebook.new_zeros(batch, channels, full, full)
    chunks = []
    for k in range(len(scales) - 1):
        cumulative = cumulative + lift_codes(token_maps[k+1], codebook, projections[k+1], full)
        next_features = F.interpolate(cumulative, size=(scales[k+1], scales[k+1]), mode="area")
        chunks.append(next_features.flatten(2).transpose(1, 2))
    return torch.cat(chunks, dim=1) if chunks else codebook.new_empty(batch, 0, channels)
