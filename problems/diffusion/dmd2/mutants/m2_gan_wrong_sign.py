# Bug: gan wrong sign.
import torch
from torch import Tensor


def _time(t: Tensor, x: Tensor) -> Tensor:
    return t.reshape(-1, *([1] * (x.ndim - 1)))


def generator_due(completed_critic_updates: int, critic_per_generator: int = 5) -> bool:
    """Schedule after a positive number of completed critic updates, starting at 1."""
    if critic_per_generator < 1 or completed_critic_updates < 0:
        raise ValueError('invalid update counts')
    return completed_critic_updates > 0 and completed_critic_updates % critic_per_generator == 0


def adversarial_losses(real_logits: Tensor, fake_logits: Tensor) -> tuple[Tensor, Tensor]:
    """Return logistic discriminator loss and non-saturating generator loss."""
    discriminator = torch.nn.functional.softplus(-real_logits).mean() + torch.nn.functional.softplus(fake_logits).mean()
    generator = torch.nn.functional.softplus(fake_logits).mean()
    return discriminator, generator


def generator_loss(generated: Tensor, gradient: Tensor, fake_logits: Tensor, gan_weight: float = .1) -> Tensor:
    """Frozen DM surrogate plus GAN; there is deliberately no paired regression term."""
    target = (generated - gradient).detach()
    matching = .5 * (generated - target).square().mean()
    return matching + gan_weight * torch.nn.functional.softplus(-fake_logits).mean()
