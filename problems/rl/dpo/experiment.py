import torch


def run(dpo_loss, seed=42):
    gen = torch.Generator().manual_seed(seed)
    reward = torch.tensor([-1., -.6, -.15, .25, .65, 1.])
    reference = torch.tensor([.26, .21, .18, .15, .12, .08])
    beta = .7
    pairs = torch.randint(0, 6, (20000, 2), generator=gen)
    choose_first = torch.rand(20000, generator=gen) < (reward[pairs[:, 0]] - reward[pairs[:, 1]]).sigmoid()
    chosen = torch.where(choose_first, pairs[:, 0], pairs[:, 1])
    rejected = torch.where(choose_first, pairs[:, 1], pairs[:, 0])
    logits = reference.log().clone().requires_grad_()
    optimizer = torch.optim.Adam([logits], lr=.06)
    curve = []
    for step in range(250):
        lp = logits.log_softmax(0)
        loss, _, _ = dpo_loss(lp[chosen], lp[rejected], reference.log()[chosen], reference.log()[rejected], beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == 249:
            implicit = beta * (logits.detach().log_softmax(0) - reference.log())
            implicit = implicit - implicit.mean() + reward.mean()
            curve.append(dict(step=step, loss=float(loss.detach()), rewards=implicit.tolist()))
    policy = logits.detach().softmax(0)
    optimum = (reference.log() + reward / beta).softmax(0)
    kl = float((policy * (policy.log() - optimum.log())).sum())
    return dict(training=curve, true_rewards=reward.tolist(), policy=policy.tolist(), optimum=optimum.tolist(),
                kl=kl, rank_order=torch.argsort(implicit).tolist())
