"""Small exact RL environments, with explicit generators for reproducibility."""
import torch
from torch import Tensor, nn


class RandomWalkMDP:
    """Absorbing random walk: interior states 1..n, terminal states 0,n+1.

    Action 0 moves left, action 1 right. Entering the right terminal earns 1.
    The policy has one Bernoulli parameter shared by every interior state.
    """
    def __init__(self, n_states: int = 7):
        self.n_states = n_states

    def exact_values(self, gamma: float = 0.99, p_right: float = 0.5):
        n = self.n_states
        transition = torch.zeros(n, n, dtype=torch.float64)
        for s in range(n):
            if s > 0:
                transition[s, s - 1] = 1 - p_right
            if s < n - 1:
                transition[s, s + 1] = p_right
        reward = torch.zeros(n, dtype=torch.float64)
        reward[-1] = p_right
        values = torch.zeros(n + 2, dtype=torch.float64)
        values[1:-1] = torch.linalg.solve(torch.eye(n, dtype=torch.float64) - gamma * transition, reward)
        q = torch.zeros(n + 2, 2, dtype=torch.float64)
        q[1:-1, 0] = gamma * values[:-2]
        q[1:-1, 1] = gamma * values[2:]
        q[n, 1] += 1
        return values, q

    def sample_episodes(self, n: int = 4000, gamma: float = 0.99, generator=None, p_right: float = 0.5):
        state = torch.full((n,), (self.n_states + 1) // 2, dtype=torch.long)
        live = torch.ones(n, dtype=torch.bool)
        records = {key: [] for key in ('states', 'actions', 'next_states', 'rewards', 'terminated', 'mask')}
        for _ in range(1000):
            action = (torch.rand(n, generator=generator) < p_right).long()
            next_state = torch.where(live, state + 2 * action - 1, state)
            term = (next_state == 0) | (next_state == self.n_states + 1)
            reward = ((next_state == self.n_states + 1) & live).double()
            for key, value in zip(records, (state.clone(), action, next_state, reward, term, live.clone())):
                records[key].append(value)
            live = live & ~term
            state = next_state
            if not live.any():
                return {key: torch.stack(value) for key, value in records.items()}
        raise RuntimeError('Random walk exceeded 1000 steps; increase rollout bound')


class Bandit:
    """A deterministic k-armed bandit with rewards evenly spaced in [0, 1]."""
    def __init__(self, k: int = 5):
        self.k = k
        self.means = torch.linspace(0, 1, k)

    def reward(self, actions: Tensor) -> Tensor:
        return self.means[actions]


class TablePolicyLM(nn.Module):
    """Position-specific categorical logits; final vocabulary item is EOS.

    Sampling includes the first EOS in the mask. Tokens after it are padding.
    Target a,b,c correspond to vocabulary IDs 0,1,2.
    """
    def __init__(self, max_len: int = 8, vocab: int = 5):
        super().__init__()
        self.max_len, self.vocab, self.eos = max_len, vocab, vocab - 1
        self.logits = nn.Parameter(torch.zeros(max_len, vocab))

    @torch.no_grad()
    def sample(self, batch: int, generator=None):
        tokens = torch.multinomial(self.logits.softmax(-1), batch, replacement=True, generator=generator).T
        is_eos = tokens == self.eos
        mask = (is_eos.cumsum(1) - is_eos.long() == 0).float()
        return tokens, mask

    def log_prob(self, tokens: Tensor) -> Tensor:
        return self.logits.log_softmax(-1).unsqueeze(0).expand(tokens.shape[0], -1, -1).gather(2, tokens.unsqueeze(-1)).squeeze(-1)

    def rewards(self, tokens: Tensor, mask: Tensor) -> Tensor:
        target = torch.arange(3, device=tokens.device)
        return ((tokens[:, :3] == target).float() * mask[:, :3]).mean(-1)

    def expected_reward(self) -> Tensor:
        probs = self.logits.softmax(-1)
        survival = torch.cat((probs.new_ones(1), (1 - probs[:2, self.eos]).cumprod(0)))
        return (survival * probs[torch.arange(3), torch.arange(3)]).mean()
