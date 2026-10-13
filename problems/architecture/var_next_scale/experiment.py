"""Teaching adaptation: trace causal influence through a tiny scale-wise attention layer."""
import math
import torch


def run(next_scale_inputs, scale_causal_mask):
    scales = [1, 2, 4]
    codebook = torch.tensor([[-1., -.4], [0., .7], [1., -.2]], dtype=torch.float64)
    tokens = [torch.tensor([[[2]]]), torch.tensor([[[0, 1], [2, 1]]]), (torch.arange(16) % 3).reshape(1, 4, 4)]
    projections = torch.stack([torch.eye(2, dtype=torch.float64), torch.tensor([[.8, .2], [-.1, .9]], dtype=torch.float64), torch.eye(2, dtype=torch.float64)])
    mask = scale_causal_mask(scales)
    positions = torch.stack((torch.arange(21).sin(), torch.arange(21).cos()), dim=-1).to(torch.float64) * .07

    def attend(maps):
        inputs = next_scale_inputs(maps, codebook, projections)
        x = torch.cat((codebook.new_zeros(1, 1, 2), inputs), dim=1) + positions
        weights = (x @ x.transpose(-1, -2) / math.sqrt(2)).masked_fill(~mask, -torch.inf).softmax(-1)
        return (weights @ x)[0]

    baseline = attend(tokens)
    influence = []
    for k in range(3):
        changed = [t.clone() for t in tokens]
        changed[k] = (changed[k] + 1) % 3
        difference = (attend(changed) - baseline).norm(dim=-1)
        influence.append([float(difference[a:b].max()) for a, b in ((0, 1), (1, 5), (5, 21))])
    forbidden = max(influence[k][q] for k in range(3) for q in range(k+1))
    allowed_effect = min(influence[0][1], influence[0][2], influence[1][2])
    return {'kind': 'toy_experiment', 'title': 'Which target scale can affect each prediction?',
            'problem': 'architecture/var_next_scale',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/html/2404.02905v2',
                           'anchor': 'Section 3.2; Figure 4; official idxBl_to_var_input and scale mask',
                           'adaptation': 'Three tiny code maps and one fixed attention layer trace information flow. Channel projections replace trained spatial Phi layers; this is not an ImageNet experiment.'},
            'metrics': {'forbidden_target_influence': forbidden, 'minimum_prefix_influence': allowed_effect,
                        'within_scale_2_visible_pairs': int(mask[1:5, 1:5].sum()), 'total_tokens': 21},
            'charts': [{'kind': 'bar', 'title': 'Output change after perturbing target tokens', 'x': ['Predict 1×1', 'Predict 2×2', 'Predict 4×4'],
                        'series': [{'name': f'Change {n}×{n} targets', 'values': values} for n, values in zip(scales, influence)]},
                       {'kind': 'line', 'title': 'Visible key positions per query', 'x_label': 'Query position', 'x': list(range(21)),
                        'series': [{'name': 'VAR scale-causal mask', 'values': mask.sum(-1).tolist()},
                                   {'name': 'Raster causal baseline', 'values': list(range(1, 22))}]}]}
