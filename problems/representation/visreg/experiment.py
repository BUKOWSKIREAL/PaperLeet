"""Teaching adaptation: separate scale, shape and center with fixed 1D sketches."""
import torch


def run(visreg_terms, view_invariance):
    dtype = torch.float64
    n = 24
    ranks = torch.arange(1, n + 1, dtype=dtype) / (n + 1)
    normal = 2.**.5 * torch.erfinv(2 * ranks - 1)
    normal = normal / normal.std(correction=0)
    # Three deterministic views of the same aligned examples, with finite near-collapse spread.
    base = torch.stack((normal, normal.roll(7)), dim=-1)
    views = torch.stack((.15 * base + .9, .18 * base + .8, .12 * base + 1.1)).requires_grad_()
    directions = torch.tensor([[1., 0., 1., 2.], [0., 1., 1., -1.]], dtype=dtype)
    steps, losses, spreads, centers, shapes = [], [], [], [], []
    initial = None
    for step in range(81):
        terms = torch.stack([torch.stack(visreg_terms(view, directions)) for view in views]).mean(0)
        loss = .5 * view_invariance(views, 2) + .5 * terms.sum()
        if initial is None: initial = float(loss.detach())
        if step % 2 == 0:
            steps.append(step); losses.append(float(loss.detach()))
            spreads.append(float(views.detach().std(dim=1, correction=0).mean()))
            centers.append(float(views.detach().mean(1).square().mean()))
            shapes.append(float(terms[1].detach()))
        if step < 80:
            gradient, = torch.autograd.grad(loss, views)
            with torch.no_grad(): views -= 8. * gradient
    measured_spread = float(views.detach().std(dim=1, correction=0).mean())
    measured_center = float(views.detach().mean(1).square().mean())
    return {
        'kind': 'toy_experiment', 'title': 'Separate embedding scale and shape', 'problem': 'representation/visreg',
        'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2606.02572',
                       'anchor': 'Section 3.1, Algorithm 1 and Eqs. (2), (8)-(9)',
                       'adaptation': 'Optimize three synthetic 24-by-2 feature tables with fixed projection directions. Inspired by collapse analysis, not a reproduced ImageNet experiment.'},
        'metrics': {'initial_objective': initial, 'final_objective': losses[-1],
                    'initial_mean_std': spreads[0], 'final_mean_std': measured_spread,
                    'initial_center_squared': centers[0], 'final_center_squared': measured_center},
        'charts': [{'kind': 'line', 'title': 'Submitted VISReg objective', 'x_label': 'Update', 'x': steps,
                    'series': [{'name': '0.5 invariance + 0.5 regularization', 'values': losses}]},
                   {'kind': 'line', 'title': 'Measured feature statistics', 'x_label': 'Update', 'x': steps,
                    'series': [{'name': 'Mean population standard deviation', 'values': spreads},
                               {'name': 'Squared feature mean', 'values': centers}]},
                   {'kind': 'line', 'title': 'Submitted shape term', 'x_label': 'Update', 'x': steps,
                    'series': [{'name': 'Sorted projection discrepancy', 'values': shapes}]}]}
