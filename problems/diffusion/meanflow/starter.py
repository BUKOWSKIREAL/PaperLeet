"""Implement the selected core functions; the experiment scaffold is provided."""
import torch
from torch import Tensor


def _time(t: Tensor, x: Tensor) -> Tensor:
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def meanflow_target(model, z: Tensor, r: Tensor, t: Tensor, velocity: Tensor) -> tuple[Tensor, Tensor]:
    """Model arguments are (z, r, t); r <= t, t=0 data, t=1 noise."""
    # TODO: implement this paper core.
    raise NotImplementedError


def meanflow_loss(model, clean: Tensor, noise: Tensor, r: Tensor, t: Tensor) -> Tensor:
    """Unweighted MSE core (adaptive loss weighting is outside this module)."""
    # TODO: implement this paper core.
    raise NotImplementedError


@torch.no_grad()
def one_step(model, noise: Tensor) -> Tensor:
    """One model call: z_0 = z_1 - u(z_1, 0, 1)."""
    # TODO: implement this paper core.
    raise NotImplementedError
