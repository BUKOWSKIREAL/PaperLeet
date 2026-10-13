# Bug: Retains Kimi Linear unbounded softplus instead of K3 bounded gate.
import torch
from torch import Tensor


def bounded_decay(logits: Tensor, log_scale: Tensor, lower_bound: float = -5.) -> Tensor:
    """[B,T,H,K] logits, [H] scale; Eq.5 retention, with lower_bound < 0."""
    scaled = logits * log_scale.exp()[None, None, :, None]
    return (-torch.nn.functional.softplus(scaled)).exp()


def recurrent_kda(q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor,
                  initial_state: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Prepared q/k [B,T,H,K], v [B,T,H,V], alpha [B,T,H,K], beta [B,T,H].
    State [B,H,K,V]; read after update. No extra scaling/normalization/projections.
    """
    batch, length, heads, keys = q.shape
    values = v.shape[-1]
    state = q.new_zeros(batch, heads, keys, values) if initial_state is None else initial_state
    outputs = []
    for t in range(length):
        decayed = alpha[:, t, :, :, None] * state
        error = v[:, t] - torch.einsum('bhk,bhkv->bhv', k[:, t], decayed)
        state = decayed + beta[:, t, :, None, None] * k[:, t, :, :, None] * error[:, :, None, :]
        outputs.append(torch.einsum('bhk,bhkv->bhv', q[:, t], state))
    output = torch.stack(outputs, dim=1) if outputs else q.new_empty(batch, 0, heads, values)
    return output, state
