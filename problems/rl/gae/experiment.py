import torch
from paperleet.common.rl_envs import RandomWalkMDP


def run(compute_gae, lambdas=(0., .5, .9, 1.), seed=42):
    env = RandomWalkMDP(7)
    gamma = .99
    values, q = env.exact_values(gamma)
    rollout = env.sample_episodes(4000, generator=torch.Generator().manual_seed(seed))
    perturbation = torch.linspace(-1.2, 1.2, 9, dtype=torch.float64)
    perturbation[[0, -1]] = 0
    approximate = values + perturbation
    v = approximate[rollout['states']] * rollout['mask']
    nv = approximate[rollout['next_states']]
    target = .25 * (q[4, 1] - q[4, 0])
    score = rollout['actions'][0].double() - .5
    result = []
    for lam in lambdas:
        adv, _ = compute_gae(rollout['rewards'], v, nv, rollout['terminated'],
                             torch.zeros_like(rollout['terminated']), gamma, lam)
        gradient = score * adv[0]
        result.append(dict(lam=float(lam), bias=abs(float(gradient.mean() - target)),
                           variance=float(gradient.var()), mean=float(gradient.mean()), target=float(target),
                           standard_error=float(gradient.std() / 4000 ** .5)))
    return result
