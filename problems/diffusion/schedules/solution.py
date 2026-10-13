"""Index i corresponds to paper time i+1; schedules are always float64."""
import math
import torch
from torch import Tensor


def linear_betas(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> Tensor:
    """Return [T] float64 linearly spaced variances."""
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)


def cosine_betas(T: int, s: float = 0.008, max_beta: float = 0.999) -> Tensor:
    """Discretize cosine cumulative alpha and clip beta."""
    time = torch.linspace(0, 1, T + 1, dtype=torch.float64)
    f = torch.cos((time + s) / (1 + s) * math.pi / 2).square()
    return (1 - f[1:] / f[:-1]).clamp(max=max_beta)


def make_schedule(betas: Tensor) -> dict[str, Tensor]:
    """All values [T] float64; alpha_bars_prev[0] = 1."""
    alphas = 1 - betas.double()
    ab = alphas.cumprod(0)
    prev = torch.cat((torch.ones_like(ab[:1]), ab[:-1]))
    return {"alphas": alphas, "alpha_bars": ab, "alpha_bars_prev": prev,
            "sqrt_alpha_bars": ab.sqrt(), "sqrt_one_minus_alpha_bars": (1 - ab).sqrt(),
            "log_snr": ab.log() - torch.log1p(-ab)}


def q_sample(x0: Tensor, t: Tensor, noise: Tensor, alpha_bars: Tensor) -> Tensor:
    """x0/noise [B,...], t [B] long; return x0's shape and dtype."""
    ab = alpha_bars.to(x0)[t].reshape(-1, *([1] * (x0.ndim - 1)))
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise
