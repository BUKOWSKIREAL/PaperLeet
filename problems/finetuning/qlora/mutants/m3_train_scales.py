# Intentional bug: Allow gradients into frozen scales.
import torch
from torch import Tensor


# NF4 levels from bitsandbytes 0.43.3 get_4bit_type('nf4'). Zero is index 7.
NF4 = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
       -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
       0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
       0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
       0.7229568362236023, 1.0)


def nf4_quantize(weight: Tensor, block_size: int = 64) -> tuple[Tensor, Tensor]:
    """Unpacked uint8 code indices plus per-block absmax; last block may be short."""
    if block_size < 1 or weight.numel() == 0:
        raise ValueError('positive block_size and a nonempty weight are required')
    flat = weight.detach().reshape(-1)
    levels = weight.new_tensor(NF4)
    codes, scales = [], []
    for block in flat.split(block_size):
        scale = block.abs().max()
        normalized = block / torch.where(scale > 0, scale, torch.ones_like(scale))
        codes.append((normalized[:, None] - levels).abs().argmin(-1).to(torch.uint8))
        scales.append(scale)
    return torch.cat(codes).reshape_as(weight), torch.stack(scales)


def nf4_dequantize(codes: Tensor, scales: Tensor, block_size: int = 64) -> Tensor:
    """Frozen block scales; output shape matches codes, dtype/device follow scales."""
    if block_size < 1 or codes.numel() == 0 or scales.numel() != (codes.numel() + block_size - 1) // block_size:
        raise ValueError('one scale is required for each nonempty block')
    levels = scales.new_tensor(NF4)
    block_scales = scales.reshape(-1).repeat_interleave(block_size)[:codes.numel()]
    return (levels[codes.long().reshape(-1)] * block_scales).reshape_as(codes)


def qlora_linear(x: Tensor, codes: Tensor, scales: Tensor, a: Tensor, b: Tensor,
                 alpha: float = 1.0, block_size: int = 64, bias: Tensor | None = None) -> Tensor:
    """Dequantized frozen base plus a differentiable alpha/r low-rank adapter."""
    weight = nf4_dequantize(codes, scales, block_size).to(dtype=x.dtype)
    base = torch.nn.functional.linear(x, weight, None if bias is None else bias.detach())
    return base + ((x @ a.T) @ b.T) * (alpha / a.shape[0])
