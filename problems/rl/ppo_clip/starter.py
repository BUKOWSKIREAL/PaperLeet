import torch
from torch import Tensor

def ppo_loss(new_logp: Tensor, old_logp: Tensor, advantages: Tensor, new_values: Tensor, old_values: Tensor, returns: Tensor, entropy: Tensor | None=None, clip_eps: float=0.2, vf_clip: float | None=0.2, vf_coef: float=0.5, ent_coef: float=0.0, norm_adv: bool=True) -> dict[str, Tensor]:
    """Inputs are floating [B]; entropy is optional [B]. Minimize loss.
    Freeze old_logp, old_values, advantages and returns. Advantage normalization
    uses sample std + 1e-8 (B=1 std=0). vf_clip=None disables value clipping.
    Return scalar loss, pg_loss, v_loss, entropy, approx_kl, clipfrac; the last
    two diagnostics must be detached. entropy=None contributes zero."""
    raise NotImplementedError()
