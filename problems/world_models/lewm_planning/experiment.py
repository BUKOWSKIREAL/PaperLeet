"""A supplied CEM sampling loop calls the learner's scoring and elite-refit functions."""
import torch


def run(terminal_cost, cem_refit):
    generator = torch.Generator().manual_seed(42)
    dtype = torch.float64
    horizon, population, elites = 5, 128, 16
    mean = torch.zeros(horizon, 2, dtype=dtype)
    variance = torch.ones_like(mean)
    goal = torch.tensor([2., -1.], dtype=dtype)
    initial = torch.tensor([-.5, .5], dtype=dtype)
    best_costs, mean_costs, spreads = [], [], []
    initial_cost = (initial - goal).square().sum().item()
    for _ in range(14):
        noise = torch.randn(population, horizon, 2, dtype=dtype, generator=generator)
        candidates = (mean + variance.sqrt() * noise).clamp(-1., 1.)
        # Frozen z_next = z + action world model; no gradients are required by CEM.
        rollouts = initial + candidates.cumsum(1)
        costs = terminal_cost(rollouts, goal)
        mean, variance, _ = cem_refit(candidates, costs, elites)
        best_costs.append(costs.min().item())
        mean_costs.append(((initial + mean.sum(0) - goal).square().sum()).item())
        spreads.append(variance.mean().item())
    path = torch.cat((initial[None], initial + mean.cumsum(0)))
    return {
        'kind': 'toy_experiment', 'problem': 'world_models/lewm_planning', 'title': 'CEM planning in a tiny latent world',
        'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2603.19312',
            'anchor': 'Section 3.2 Eqs. (4)-(5), Appendix B Algorithm 2',
            'adaptation': 'A known two-dimensional additive latent transition replaces the learned image model. Fourteen CEM iterations optimize five bounded actions. This isolates terminal scoring and elite statistics; it is not Push-T or closed-loop MPC evaluation.'},
        'metrics': {'initial_cost': initial_cost, 'final_mean_plan_cost': mean_costs[-1],
            'final_best_sample_cost': best_costs[-1], 'final_mean_variance': spreads[-1]},
        'charts': [
            {'kind': 'line', 'title': 'Distance to goal', 'x_label': 'CEM iteration', 'x': list(range(1, 15)),
                'series': [{'name': 'Best sampled plan', 'values': best_costs}, {'name': 'Refitted mean plan', 'values': mean_costs}]},
            {'kind': 'line', 'title': 'Final planned trajectory', 'x_label': 'Time', 'x': list(range(6)),
                'series': [{'name': 'x', 'values': path[:, 0].tolist()}, {'name': 'Goal x', 'values': [goal[0].item()] * 6},
                    {'name': 'y', 'values': path[:, 1].tolist()}, {'name': 'Goal y', 'values': [goal[1].item()] * 6}]},
            {'kind': 'line', 'title': 'Sampling variance', 'x_label': 'CEM iteration', 'x': list(range(1, 15)),
                'series': [{'name': 'Mean elite variance', 'values': spreads}]},
        ],
    }
