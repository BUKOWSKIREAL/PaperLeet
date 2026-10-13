"""Small analytic distributions used by the diffusion judges."""
import math
import torch
from torch import Tensor


class GaussianMixture:
    def __init__(self, means: Tensor, std: float, weights: Tensor | None = None):
        self.means = torch.as_tensor(means)
        if not self.means.is_floating_point():
            self.means = self.means.float()
        self.std = float(std)
        if self.means.ndim != 2 or self.std < 0:
            raise ValueError("means must be [K,D] and std nonnegative")
        self.weights = (torch.ones(len(self.means), dtype=self.means.dtype, device=self.means.device)
                        if weights is None else torch.as_tensor(weights).to(self.means))
        if (self.weights < 0).any() or self.weights.sum() <= 0:
            raise ValueError("weights must be nonnegative with positive sum")
        self.weights = self.weights / self.weights.sum()

    def sample(self, n: int, generator=None) -> Tensor:
        k = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        return self.means[k] + self.std * torch.randn(n, self.means.shape[1], dtype=self.means.dtype,
                                                     device=self.means.device, generator=generator)

    def log_prob(self, x: Tensor) -> Tensor:
        if self.std == 0:
            raise ValueError("point masses have no finite Lebesgue density")
        delta = x[:, None, :] - self.means.to(x)[None, :, :]
        logits = -delta.square().sum(-1) / (2 * self.std**2) + self.weights.to(x).log()
        return logits.logsumexp(-1) - x.shape[-1] * math.log(self.std * math.sqrt(2 * math.pi))

    def score(self, x: Tensor) -> Tensor:
        if self.std == 0:
            raise ValueError("diffuse a point mass before evaluating its score")
        delta = x[:, None, :] - self.means.to(x)[None, :, :]
        logits = -delta.square().sum(-1) / (2 * self.std**2) + self.weights.to(x).log()
        return -(logits.softmax(-1)[..., None] * delta).sum(1) / self.std**2

    def diffused(self, alpha_bar: float | Tensor):
        ab = float(alpha_bar)
        return GaussianMixture(math.sqrt(ab) * self.means,
                               math.sqrt(ab * self.std**2 + 1 - ab), self.weights)


def eight_gaussians(n: int, radius: float = 2.0, std: float = 0.15, generator=None) -> Tensor:
    angles = torch.arange(8) * (2 * math.pi / 8)
    centers = radius * torch.stack((angles.cos(), angles.sin()), -1)
    return GaussianMixture(centers, std).sample(n, generator)
