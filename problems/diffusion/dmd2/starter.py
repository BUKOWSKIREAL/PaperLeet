"""Implement the selected core functions; the experiment scaffold is provided."""
import torch
from torch import Tensor


def _time(t: Tensor, x: Tensor) -> Tensor:
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def generator_due(completed_critic_updates: int, critic_per_generator: int = 5) -> bool:
    """Schedule after a positive number of completed critic updates, starting at 1."""
    # TODO: implement this paper core.
    raise NotImplementedError


def adversarial_losses(real_logits: Tensor, fake_logits: Tensor) -> tuple[Tensor, Tensor]:
    """Return logistic discriminator loss and non-saturating generator loss."""
    # TODO: implement this paper core.
    raise NotImplementedError


def generator_loss(generated: Tensor, gradient: Tensor, fake_logits: Tensor, gan_weight: float = .1) -> Tensor:
    """Frozen DM surrogate plus GAN; there is deliberately no paired regression term."""
    # TODO: implement this paper core.
    raise NotImplementedError
