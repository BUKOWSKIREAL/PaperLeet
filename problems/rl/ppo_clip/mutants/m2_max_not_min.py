# Bug: 乐观界代替悲观界
import torch
from torch import Tensor


def ppo_loss(new_logp: Tensor, old_logp: Tensor, advantages: Tensor, new_values: Tensor,
             old_values: Tensor, returns: Tensor, entropy: Tensor | None = None,
             clip_eps: float = 0.2, vf_clip: float | None = 0.2, vf_coef: float = 0.5,
             ent_coef: float = 0.0, norm_adv: bool = True) -> dict[str, Tensor]:
    """All inputs [B]; minimize returned loss. Diagnostics are detached scalars."""
    old_logp = old_logp.detach()
    old_values = old_values.detach()
    advantages, returns = advantages.detach(), returns.detach()
    if norm_adv:
        std = advantages.std(unbiased=True) if advantages.numel() > 1 else advantages.new_zeros(())
        advantages = (advantages - advantages.mean()) / (std + 1e-8)
    logratio = new_logp - old_logp
    ratio = logratio.exp()
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps)
    pg_loss = -torch.maximum(ratio * advantages, clipped * advantages).mean()
    value_error = (new_values - returns).square()
    if vf_clip is not None:
        v_clipped = old_values + (new_values - old_values).clamp(-vf_clip, vf_clip)
        value_error = torch.maximum(value_error, (v_clipped - returns).square())
    v_loss = 0.5 * value_error.mean()
    ent = new_logp.new_zeros(()) if entropy is None else entropy.mean()
    with torch.no_grad():
        approx_kl = (ratio - 1 - logratio).mean()
        clipfrac = ((ratio - 1).abs() > clip_eps).to(new_logp.dtype).mean()
    return dict(loss=pg_loss + vf_coef * v_loss - ent_coef * ent, pg_loss=pg_loss,
                v_loss=v_loss, entropy=ent, approx_kl=approx_kl, clipfrac=clipfrac)
