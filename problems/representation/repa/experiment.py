"""Teaching adaptation: learn a tiny projector against fixed clean features."""
import math
import torch


def run(repa_alignment, repa_objective):
    dtype = torch.float64
    grid = torch.arange(36, dtype=dtype).reshape(2, 6, 3)
    clean = torch.sin(grid * .43) + torch.cos(grid * .17)
    hidden = clean + .06 * torch.cos(grid * 1.7)
    teacher_matrix = torch.tensor([[1., .6, -.2], [-.3, .8, .4], [.2, -.5, 1.1]], dtype=dtype)
    teacher = clean @ teacher_matrix
    projector = torch.tensor([[.3, -.3, .1], [.2, .2, -.2], [-.1, .4, .2]], dtype=dtype, requires_grad=True)
    steps, cosine, objectives = [], [], []
    zero = torch.zeros(1, dtype=dtype)
    for step in range(81):
        projected = hidden @ projector
        loss = repa_objective(zero, zero, projected, teacher, weight=.5)
        steps.append(step)
        objectives.append(float(loss.detach()))
        # Measure with an independent scalar cosine, not the submitted loss sign.
        rows = zip(projected.detach().reshape(-1, 3).tolist(), teacher.reshape(-1, 3).tolist())
        cosines = [sum(a*b for a,b in zip(p,t)) / math.sqrt(sum(a*a for a in p)*sum(b*b for b in t)) for p,t in rows]
        cosine.append(sum(cosines) / len(cosines))
        if step < 80:
            grad, = torch.autograd.grad(loss, projector)
            with torch.no_grad():
                projector -= .8 * grad
    direct = float(repa_alignment(hidden @ projector, teacher).detach())
    return {
        'kind': 'toy_experiment', 'title': 'A tiny REPA projector', 'problem': 'representation/repa',
        'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2410.06940',
                       'anchor': 'Representation alignment objective; author loss.py',
                       'adaptation': 'Synthetic clean/noisy patch features and an 81-step CPU projector curve; not an original paper experiment or image generation benchmark.'},
        'metrics': {'initial_cosine': cosine[0], 'final_cosine': cosine[-1],
                    'final_objective': objectives[-1], 'alignment_formula_error': abs(direct + cosine[-1])},
        'charts': [{'kind': 'line', 'title': 'Patch alignment during optimization', 'x_label': 'Update',
                    'x': steps, 'series': [{'name': 'Measured mean cosine', 'values': cosine}]},
                   {'kind': 'line', 'title': 'Submitted combined objective', 'x_label': 'Update',
                    'x': steps, 'series': [{'name': 'MSE + 0.5 alignment', 'values': objectives}]}]}
