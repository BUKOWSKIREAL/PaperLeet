import torch
from paperleet.common.rl_envs import Bandit


def run(ppo_loss, seed=42):
    gen = torch.Generator().manual_seed(seed)
    env = Bandit(5)
    logits = torch.zeros(5, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=.04)
    curve = []
    for step in range(30):
        probs = logits.detach().softmax(0)
        actions = torch.multinomial(probs, 256, replacement=True, generator=gen)
        old = probs.log()[actions]
        rewards = env.reward(actions)
        adv = rewards - (probs * env.means).sum()
        for _ in range(10):
            for ids in torch.randperm(256, generator=gen).split(64):
                lp = logits.log_softmax(0)[actions[ids]]
                zero = torch.zeros_like(lp)
                loss = ppo_loss(lp, old[ids], adv[ids], zero, zero, zero, vf_coef=0)['loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        curve.append(dict(step=step + 1, reward=float((logits.detach().softmax(0) * env.means).sum()),
                          best_prob=float(logits.detach().softmax(0)[-1])))
    probe = torch.zeros(5, requires_grad=True)
    opt = torch.optim.Adam([probe], lr=.04)
    old_probs = probe.detach().softmax(0)
    actions = torch.multinomial(old_probs, 256, replacement=True, generator=gen)
    old_lp, adv = old_probs.log()[actions], env.reward(actions) - env.means.mean()
    kl_curve = []
    for epoch in range(20):
        lp = probe.log_softmax(0)[actions]
        zero = torch.zeros_like(lp)
        loss = ppo_loss(lp, old_lp, adv, zero, zero, zero, vf_coef=0)['loss']
        opt.zero_grad()
        loss.backward()
        opt.step()
        kl_curve.append(dict(epoch=epoch + 1, kl=float((old_probs * (old_probs.log() - probe.detach().log_softmax(0))).sum())))
    return dict(training=curve, trust_region=kl_curve, best_prob=curve[-1]['best_prob'], max_kl=max(x['kl'] for x in kl_curve))
