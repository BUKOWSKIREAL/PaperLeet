"""Implement the selected core functions; the experiment scaffold is provided."""
import torch
from torch import Tensor


def _time(t: Tensor, x: Tensor) -> Tensor:
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def drift_field(generated: Tensor, positive: Tensor, temperature: float = .2) -> Tensor:
    """Official toy's doubly normalized distance kernel; generated self-pairs masked."""
    # TODO: implement this paper core.
    raise NotImplementedError


def drifting_loss(generated: Tensor, positive: Tensor, temperature: float = .2) -> Tensor:
    """MSE(g, stopgrad(g + V)); gradients flow only through the left-hand g."""
    # TODO: implement this paper core.
    raise NotImplementedError
