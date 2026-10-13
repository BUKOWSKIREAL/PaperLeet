import torch
from paperleet.common.rl_envs import TablePolicyLM


def run(advantage_fn, loss_fn, seed=42, dr=False, steps=150):
    gen = torch.Generator().manual_seed(seed)
    model = TablePolicyLM(max_len=8, vocab=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=.12)
    curve = [dict(step=0, reward=float(model.expected_reward().detach()), length=4.16, wrong_length=4.16)]
    for step in range(steps):
        tokens, mask = model.sample(64, gen)
        rewards = model.rewards(tokens, mask)
        advantage = advantage_fn(rewards, 8)
        old = model.log_prob(tokens).detach()
        reference = torch.full_like(old, -torch.log(torch.tensor(5.)).item())
        kwargs = dict(max_len=8) if dr else {}
        loss, _ = loss_fn(model.log_prob(tokens), old, reference, advantage, mask, beta=.04, **kwargs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wrong = rewards < 1
        wrong_length = float(mask.sum(-1)[wrong].mean()) if wrong.any() else None
        curve.append(dict(step=step + 1, reward=float(model.expected_reward().detach()),
                          length=float(mask.sum(-1).mean()), wrong_length=wrong_length))
    return dict(training=curve, final_reward=curve[-1]['reward'])
