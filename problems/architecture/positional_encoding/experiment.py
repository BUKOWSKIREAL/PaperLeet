"""Teaching adaptation of Transformer §3.5; the paper does not report this toy."""
import math
import torch


def run(encoding):
    length, width, offset = 32, 8, 7
    base = encoding(length, width, dtype=torch.float64)
    shifted = encoding(length, width, start=offset, dtype=torch.float64)
    # Independent analytic rotation at each paper frequency, acting on [sin, cos].
    predicted = torch.empty_like(base)
    for pair in range(width // 2):
        angle = offset / (10000.0 ** (2 * pair / width))
        sine, cosine = base[:, 2 * pair], base[:, 2 * pair + 1]
        predicted[:, 2 * pair] = sine * math.cos(angle) + cosine * math.sin(angle)
        predicted[:, 2 * pair + 1] = cosine * math.cos(angle) - sine * math.sin(angle)
    pairwise = torch.pdist(base)
    rotation_error = (shifted - predicted).abs().max().item()
    minimum_distance = pairwise.min().item()
    return {'kind': 'toy_experiment', 'title': 'Relative offsets are pairwise rotations',
            'problem': 'architecture/positional_encoding',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/1706.03762',
                           'anchor': 'Section 3.5: positional encoding',
                           'adaptation': '32 positions, width 8, offset 7; algebraic demonstration, not a paper experiment.'},
            'metrics': {'maximum_rotation_error': rotation_error, 'minimum_position_distance': minimum_distance},
            'charts': [{'kind': 'line', 'title': 'Position plus seven: implementation versus analytic rotation',
                        'x_label': 'Position', 'x': list(range(length)),
                        'series': [{'name': 'Implementation sine', 'values': shifted[:, 0].tolist()},
                                   {'name': 'Rotation sine', 'values': predicted[:, 0].tolist()},
                                   {'name': 'Implementation cosine', 'values': shifted[:, 1].tolist()}]}]}
