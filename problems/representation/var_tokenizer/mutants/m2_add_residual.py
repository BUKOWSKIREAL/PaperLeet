# BUG: Adds the selected reconstruction back to the residual.
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


def resize_residual(residual: Tensor, size: int) -> Tensor:
    """Provided: area reduction, preserving the full-resolution case."""
    return F.interpolate(residual, size=(size, size), mode="area") if size != residual.shape[-1] else residual


@torch.no_grad()
def multiscale_quantize(features: Tensor, codebook: Tensor, scales: list[int], projections: Tensor) -> tuple[list[Tensor], list[Tensor]]:
    """Frozen residual VQ: return tokens and full-resolution cumulative reconstructions."""
    if features.ndim != 4 or features.numel() == 0 or features.shape[-2] != features.shape[-1]:
        raise ValueError("features must be nonempty square [B,C,H,H]")
    if not scales or any(type(n) is not int or n <= 0 for n in scales):
        raise ValueError("scales must contain positive integers")
    batch, channels, full, _ = features.shape
    if codebook.ndim != 2 or channels != codebook.shape[-1] or scales[-1] != full:
        raise ValueError("codebook channels and final scale must match features")
    if features.dtype != codebook.dtype or features.device != codebook.device:
        raise ValueError("feature dtype/device must match codebook")
    _validate_codes([torch.zeros(batch, n, n, dtype=torch.long, device=features.device) for n in scales], codebook, projections)
    residual = features.clone()
    cumulative = torch.zeros_like(features)
    tokens, reconstructions = [], []
    for k, n in enumerate(scales):
        reduced = resize_residual(residual, n).permute(0, 2, 3, 1)
        distances = (reduced[..., None, :] - codebook).square().sum(-1)
        indices = distances.argmin(-1)
        contribution = lift_codes(indices, codebook, projections[k], full)
        residual = residual + contribution
        cumulative = cumulative + contribution
        tokens.append(indices)
        reconstructions.append(cumulative.clone())
    return tokens, reconstructions


@torch.no_grad()
def decode_multiscale(token_maps: list[Tensor], codebook: Tensor, projections: Tensor) -> Tensor:
    """Sum all projected code maps at the largest resolution."""
    scales = _validate_codes(token_maps, codebook, projections)
    cumulative = codebook.new_zeros(token_maps[0].shape[0], codebook.shape[1], scales[-1], scales[-1])
    for k, token_map in enumerate(token_maps):
        cumulative = cumulative + lift_codes(token_map, codebook, projections[k], scales[-1])
    return cumulative
