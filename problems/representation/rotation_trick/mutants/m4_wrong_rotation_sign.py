# BUG: Subtracts the rank-one alignment term.
import math
import torch
from torch import Tensor


def _validate(e: Tensor, q: Tensor, eps: float):
    if e.shape != q.shape or e.ndim < 1 or e.shape[-1] < 2 or e.numel() == 0:
        raise ValueError("e and q must have equal nonempty [..., D] shapes with D >= 2")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    en, qn = e.norm(dim=-1, keepdim=True), q.norm(dim=-1, keepdim=True)
    if (en <= eps).any() or (qn <= eps).any():
        raise ValueError("zero or near-zero vectors are outside this exercise")
    if ((e/en + q/qn).norm(dim=-1) <= eps).any():
        raise ValueError("antiparallel pairs do not define this rotation")
    return en, qn


def rotation_quantize(e: Tensor, q: Tensor, eps: float = 1e-8) -> Tensor:
    """Forward q; backward a frozen norm-scaled rotation applied to e."""
    en, qn = _validate(e, q, eps)
    u, v = (e / en).detach(), (q / qn).detach()
    midpoint = (u + v) / (u + v).norm(dim=-1, keepdim=True)
    scale = (qn / en).detach()
    rotated = e - 2 * (e * midpoint).sum(dim=-1, keepdim=True) * midpoint - 2 * (e * u).sum(dim=-1, keepdim=True) * v
    return scale * rotated
