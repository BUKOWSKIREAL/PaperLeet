# Bug: 使用 i/d 而不是 2i/d
import math
import torch
from torch import Tensor


def sinusoidal_encoding(length: int, d_model: int, start: int = 0, *,
                        dtype=torch.float32, device=None) -> Tensor:
    """Return [length, d_model]; positions start..start+length-1, odd widths allowed."""
    if length < 0 or d_model <= 0 or start < 0:
        raise ValueError("length/start must be nonnegative and d_model must be positive")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    positions = torch.arange(start, start + length, dtype=dtype, device=device)
    frequencies = torch.exp(torch.arange(0, d_model, 2, dtype=dtype, device=device) * (-math.log(10000.0) / (2 * d_model)))
    angles = positions[:, None] * frequencies[None, :]
    result = torch.empty(length, d_model, dtype=dtype, device=device)
    result[:, 0::2] = angles.sin()
    result[:, 1::2] = angles[:, :d_model // 2].cos()
    return result
