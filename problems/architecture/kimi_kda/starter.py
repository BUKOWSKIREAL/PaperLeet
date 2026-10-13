import torch
from torch import Tensor


def bounded_decay(logits: Tensor, log_scale: Tensor, lower_bound: float = -5.) -> Tensor:
    """Required: K3 Eq.5 retention exp(lower_bound * sigmoid(exp(A) * z)).
    logits [B,T,H,K], log_scale [H], lower_bound < 0; preserve autograd/dtype/device.
    """
    raise NotImplementedError()


def recurrent_kda(q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor,
                  initial_state: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Required: decay state, correct its key prediction, then read the updated state.
    q/k are prepared vectors [B,T,H,K], v [B,T,H,V], alpha [B,T,H,K], beta [B,T,H].
    initial_state [B,H,K,V], default zeros. Return output [B,T,H,V], final [B,H,K,V].
    alpha and beta are retention/write factors, not logits; endpoints 0/1 are supported.
    No projections, q scaling, q/k normalization, output gate or RMSNorm here.
    Support T=0, K!=V, noncontiguous inputs, gradients and continuation from initial_state.
    Preserve all input tensors; the tiny experiments and projection scaffolding are supplied.
    """
    raise NotImplementedError()
