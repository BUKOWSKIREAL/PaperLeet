import torch
from torch import Tensor
NF4 = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0)

def nf4_quantize(weight: Tensor, block_size: int=64) -> tuple[Tensor, Tensor]:
    """Unpacked uint8 code indices plus per-block absmax; last block may be short."""
    raise NotImplementedError('Implement nf4_quantize')

def nf4_dequantize(codes: Tensor, scales: Tensor, block_size: int=64) -> Tensor:
    """Frozen block scales; output shape matches codes, dtype/device follow scales."""
    raise NotImplementedError('Implement nf4_dequantize')

def qlora_linear(x: Tensor, codes: Tensor, scales: Tensor, a: Tensor, b: Tensor, alpha: float=1.0, block_size: int=64, bias: Tensor | None=None) -> Tensor:
    """Dequantized frozen base plus a differentiable alpha/r low-rank adapter."""
    raise NotImplementedError('Implement qlora_linear')
