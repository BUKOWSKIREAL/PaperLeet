"""Exact conditional expectations; no learned network is hidden in these helpers."""
import torch
from .toydata import GaussianMixture


def gmm_eps_oracle(gmm: GaussianMixture, alpha_bars):
    def predict(x, t):
        ab = alpha_bars.to(x)[t].reshape(-1, 1, 1)
        var = ab * gmm.std**2 + 1 - ab
        delta = x[:, None, :] - ab.sqrt() * gmm.means.to(x)[None, :, :]
        logits = -delta.square().sum(-1) / (2 * var[..., 0]) + gmm.weights.to(x).log()
        score = -(logits.softmax(-1)[..., None] * delta).sum(1) / var[..., 0]
        return -(1 - ab[..., 0]).sqrt() * score
    return predict


def point_velocity_oracle(mean):
    def velocity(x, t):
        return (torch.as_tensor(mean).to(x) - x) / (1 - t[:, None])
    return velocity


def gaussian_velocity_oracle(mean, std):
    def velocity(x, t):
        t = t[:, None]
        mu = torch.as_tensor(mean).to(x)
        coef = (t * std**2 - (1 - t)) / ((1 - t).square() + t.square() * std**2)
        return mu + coef * (x - t * mu)
    return velocity
