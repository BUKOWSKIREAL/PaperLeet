"""Teaching adaptation: fit a nonlinear per-token target with the submitted FFN."""
import torch


def target(x):
    """Independent analytic task; never call the reference or learner to construct labels."""
    return torch.stack((x[..., 0].abs() + .3 * x[..., 1],
                        (x[..., 1] - .2).clamp_min(0) - .4 * x[..., 0]), -1)


def run(ffn, seed=42, steps=300):
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(128, 2, generator=generator) * 2 - 1
    heldout = torch.rand(96, 2, generator=generator) * 2 - 1
    y, expected = target(x), target(heldout)
    w1 = (torch.randn(16, 2, generator=generator) * .5).requires_grad_()
    b1 = (torch.randn(16, generator=generator) * .2).requires_grad_()
    w2 = (torch.randn(2, 16, generator=generator) * .2).requires_grad_()
    b2 = torch.zeros(2, requires_grad=True)
    parameters = (w1, b1, w2, b2)
    optimizer = torch.optim.Adam(parameters, lr=.04)
    curve = []
    for step in range(steps + 1):
        loss = (ffn(x, *parameters) - y).square().mean()
        if step % 10 == 0 or step == steps:
            curve.append([step, loss.detach().item()])
        if step == steps:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        actual = ffn(heldout, *parameters)
        mse = (actual - expected).square().mean().item()
        scan = torch.stack((torch.linspace(-1, 1, 41), torch.zeros(41)), -1)
        prediction = ffn(scan, *parameters)
    return {'kind': 'toy_experiment', 'title': 'Learn a nonlinear map independently at each token',
            'problem': 'architecture/positionwise_ffn',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/1706.03762',
                           'anchor': 'Section 3.3, Eq. (2)',
                           'adaptation': '128 two-dimensional training tokens, 96 held-out tokens and a width-16 FFN; not a translation benchmark.'},
            'metrics': {'initial_training_mse': curve[0][1], 'final_training_mse': curve[-1][1],
                        'heldout_mse': mse, 'training_steps': steps},
            'loss_curve': curve,
            'charts': [{'kind': 'line', 'title': 'Training loss', 'x_label': 'Step',
                        'x': [p[0] for p in curve], 'series': [{'name': 'MSE', 'values': [p[1] for p in curve]}]},
                       {'kind': 'line', 'title': 'Learn absolute value along x₂ = 0', 'x_label': 'x₁',
                        'x': scan[:, 0].tolist(),
                        'series': [{'name': 'Implementation', 'values': prediction[:, 0].tolist()},
                                   {'name': 'Analytic target', 'values': target(scan)[:, 0].tolist()}]}]}
