# Bug: skip input scaling.
import torch
from torch import Tensor


def _time(t: Tensor, x: Tensor) -> Tensor:
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def consistency_function(model, noisy: Tensor, sigma: Tensor, sigma_min: float = .002, sigma_data: float = .5) -> Tensor:
    """Boundary-preserving EDM-style function with c_in-preconditioned network input."""
    s = _time(sigma, noisy)
    c_skip = sigma_data**2 / ((s - sigma_min).square() + sigma_data**2)
    c_out = (s - sigma_min) * sigma_data / (s.square() + sigma_data**2).sqrt()
    c_in = 1 / (s.square() + sigma_data**2).sqrt()
    return c_skip * noisy + c_out * model(noisy, sigma)


def consistency_loss(prediction: Tensor, target_prediction: Tensor) -> Tensor:
    """Selected L2 distance; target is an EMA prediction at the adjacent lower noise."""
    return (prediction - target_prediction.detach()).square().mean()


@torch.no_grad()
def update_ema(online, target, decay: float = .95) -> None:
    """In-place parameter EMA: target <- decay*target + (1-decay)*online."""
    if not 0 <= decay <= 1:
        raise ValueError('decay must be in [0,1]')
    for source, destination in zip(online.parameters(), target.parameters(), strict=True):
        destination.mul_(decay).add_(source, alpha=1 - decay)


@torch.no_grad()
def teacher_euler(noisy: Tensor, denoised: Tensor, sigma: Tensor, next_sigma: Tensor) -> Tensor:
    """Provided VE probability-flow Euler step from sigma to a lower next_sigma."""
    return noisy + _time(next_sigma - sigma, noisy) * (noisy - denoised) / _time(sigma, noisy)
