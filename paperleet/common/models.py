"""A deterministic, CPU-sized time-conditioned network."""
import math
import torch
from torch import nn


class TimeMLP(nn.Module):
    def __init__(self, dim=2, hidden=128, depth=3, t_dim=32):
        super().__init__()
        if t_dim % 2:
            raise ValueError("t_dim must be even")
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(100), t_dim // 2)))
        with torch.random.fork_rng():
            torch.manual_seed(42)
            layers = [nn.Linear(dim + t_dim, hidden), nn.SiLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden, hidden), nn.SiLU()]
            layers += [nn.Linear(hidden, dim)]
            self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        phase = t[:, None] * self.frequencies.to(x)
        return self.net(torch.cat((x, phase.sin(), phase.cos()), -1))
