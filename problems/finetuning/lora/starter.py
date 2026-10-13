import torch
from torch import Tensor

def lora_delta(a: Tensor, b: Tensor, alpha: float=1.0) -> Tensor:
    """A [r,in], B [out,r] -> delta W [out,in], scaled by alpha/r."""
    raise NotImplementedError('Implement lora_delta')

def lora_linear(x: Tensor, weight: Tensor, a: Tensor, b: Tensor, alpha: float=1.0, bias: Tensor | None=None) -> Tensor:
    """Frozen base and bias; gradients still flow to x and both adapters."""
    raise NotImplementedError('Implement lora_linear')
