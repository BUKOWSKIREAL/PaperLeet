import math
import torch
from torch import Tensor


def repa_alignment(projected: Tensor, teacher: Tensor, eps: float = 1e-8) -> Tensor:
    """Negative patch-wise cosine; teacher features are fixed targets."""
    if projected.shape != teacher.shape or projected.ndim != 3 or projected.numel() == 0:
        raise ValueError("features must have the same nonempty [batch, patches, channels] shape")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    raise NotImplementedError("Normalize each patch, stop teacher gradients, and reduce")


def repa_objective(prediction: Tensor, velocity_target: Tensor, projected: Tensor,
                   teacher: Tensor, weight: float = 0.5, eps: float = 1e-8) -> Tensor:
    """Mean velocity MSE plus weighted representation alignment."""
    if prediction.shape != velocity_target.shape or prediction.numel() == 0:
        raise ValueError("velocity tensors must have equal nonempty shapes")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    raise NotImplementedError("Combine velocity MSE and repa_alignment")
