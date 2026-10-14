"""LeetPaper teaching adaptation: recover a known rank-two change to a frozen map."""
import torch


def run(delta_fn, linear_fn, seed=42, steps=300):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(6, 8, generator=generator) * .2
    target_delta = torch.randn(6, 2, generator=generator) @ torch.randn(2, 8, generator=generator) * .2
    x = torch.randn(96, 8, generator=generator)
    y = x @ (weight + target_delta).T
    a = (torch.randn(2, 8, generator=generator) * .1).requires_grad_()
    b = torch.zeros(6, 2, requires_grad=True)
    optimizer = torch.optim.Adam([a, b], lr=.06)
    curve = []
    for step in range(steps + 1):
        prediction = linear_fn(x, weight, a, b, alpha=4.)
        loss = (prediction - y).square().mean()
        curve.append(dict(step=step, mse=float(loss.detach())))
        if step == steps:
            break
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        merged = weight + delta_fn(a, b, alpha=4.)
        merge_error = float((linear_fn(x, weight, a, b, alpha=4.) - x @ merged.T).abs().max())
    out = dict(provenance='LeetPaper synthetic rank-two regression; not a paper benchmark',
                final_mse=curve[-1]['mse'], merge_error=merge_error, training=curve,
                adapter_parameters=a.numel() + b.numel(), base_parameters=weight.numel(),
                charts=[dict(kind='line', title='Learn a low-rank update', x=[p['step'] for p in curve],
                             series=[dict(name='Regression MSE', values=[p['mse'] for p in curve])])])

    out['loss_curve'] = [[row['step'], row['mse']] for row in out['training']]
    out['metrics'] = {key: value for key, value in out.items() if key in ('final_kl', 'final_mse', 'merge_error', 'quantization_rmse', 'adapter_parameters', 'base_parameters', 'code_count', 'scale_count')}
    out.update(kind='toy_experiment', title='Recover a known low-rank update', provenance={'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2106.09685v2', 'adaptation': 'LeetPaper rank-two synthetic regression, not a language-model benchmark.'})
    return out
