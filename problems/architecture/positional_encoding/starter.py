import math
import torch
from torch import Tensor


def sinusoidal_encoding(length: int, d_model: int, start: int = 0, *,
                        dtype=torch.float32, device=None) -> Tensor:
    """Return [length, d_model]; positions start..start+length-1, odd widths allowed."""
    raise NotImplementedError("Implement sinusoidal_encoding")
