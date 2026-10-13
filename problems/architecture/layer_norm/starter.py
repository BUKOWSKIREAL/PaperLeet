import torch
from torch import Tensor


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    """Required: x [..., D], affine weight/bias [D], population variance over D.
    Epsilon belongs inside the square root. Preserve dtype, device and gradients.
    Finite float32/float64 inputs, D >= 1, eps > 0; no input mutation.
    Implement with tensor reductions rather than calling a layer_norm API."""
    raise NotImplementedError()


def post_norm_residual(x: Tensor, sublayer_output: Tensor, weight: Tensor,
                       bias: Tensor, eps: float = 1e-5) -> Tensor:
    """Required: original Transformer LayerNorm(x + sublayer_output).
    x and sublayer_output have identical shapes; dropout is disabled.
    Reuse your layer_norm; the sublayer output has already been computed."""
    raise NotImplementedError()
