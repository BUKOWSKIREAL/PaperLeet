"""Conversions use alpha=sqrt(alpha_bar), never alpha_bar itself."""
from torch import Tensor


def eps_to_x0(x_t: Tensor, eps: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    """Recover clean data; coefficients broadcast to x_t, denominator >= 1e-8."""
    return (x_t - sigma * eps) / alpha.clamp(min=1e-8)


def x0_to_eps(x_t: Tensor, x0: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    """Recover noise, retaining finite values at the clean endpoint."""
    return (x_t - alpha * x0) / sigma.clamp(min=1e-8)


def make_v(x0: Tensor, eps: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    """Angular velocity for x_t = alpha*x0 + sigma*eps."""
    return alpha * eps - sigma * x0


def v_to_x0(x_t: Tensor, v: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    """Inverse orthogonal rotation, stable at both endpoints."""
    return alpha * x_t - sigma * v


def v_to_eps(x_t: Tensor, v: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    """Inverse orthogonal rotation, stable at both endpoints."""
    return sigma * x_t + alpha * v


def convert(pred: Tensor, src: str, dst: str, x_t: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
    """Convert a prediction between x0, eps and v; preserve shape/dtype."""
    if src not in {"x0", "eps", "v"} or dst not in {"x0", "eps", "v"}:
        raise ValueError("src and dst must be x0, eps or v")
    if src == dst:
        return pred
    if src == "v":
        return v_to_x0(x_t, pred, alpha, sigma) if dst == "x0" else v_to_eps(x_t, pred, alpha, sigma)
    x0 = pred if src == "x0" else eps_to_x0(x_t, pred, alpha, sigma)
    eps = pred if src == "eps" else x0_to_eps(x_t, pred, alpha, sigma)
    return {"x0": x0, "eps": eps, "v": make_v(x0, eps, alpha, sigma)}[dst]
