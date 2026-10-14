"""Teaching adaptation: fit predictions for supplied masked latent targets."""
import torch


def run(masked_targets, masked_prediction_loss, ema_update):
    dtype = torch.float64
    grid = torch.arange(48, dtype=dtype).reshape(2, 6, 4)
    features = torch.sin(grid * .3) + .1 * grid
    masks = [torch.tensor([[4, 1], [0, 5]]), torch.tensor([[2, 3], [4, 1]])]
    targets = masked_targets(features, masks, 2)
    prediction = torch.zeros_like(targets, requires_grad=True)
    steps, losses, errors, ema_gaps = [], [], [], []
    teacher = torch.zeros(3, dtype=dtype)
    online = torch.tensor([1., -2., .5], dtype=dtype)
    for step in range(61):
        loss = masked_prediction_loss(prediction, targets)
        steps.append(step); losses.append(float(loss.detach()))
        errors.append(float((prediction.detach() - targets.detach()).abs().mean()))
        ema_gaps.append(float((teacher - online).norm()))
        if step < 60:
            gradient, = torch.autograd.grad(loss, prediction)
            with torch.no_grad(): prediction -= 16. * gradient
            teacher = ema_update(teacher, online, .9)
    # Independent scalar layer-normalization + block/context/batch gather oracle.
    oracle = []
    for mask in masks:
        block = []
        for batch, indices in enumerate(mask.tolist()):
            row = []
            for index in indices:
                values = features[batch, index].tolist()
                mean = sum(values) / len(values)
                variance = sum((v - mean)**2 for v in values) / len(values)
                row.append([(v - mean) / (variance + 1e-5)**.5 for v in values])
            block.append(row)
        oracle.extend(block * 2)
    oracle_error = float((targets.detach() - torch.tensor(oracle, dtype=dtype)).abs().max())
    expected_ema = (1 - .9**60) * online
    return {
        'kind': 'toy_experiment', 'title': 'Predict a masked latent block', 'problem': 'representation/ijepa',
        'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2301.08243',
                       'anchor': 'Target encoder / joint-embedding prediction; official src/train.py',
                       'adaptation': 'Fit a small prediction table to supplied synthetic target blocks, and show an independent fixed-online EMA trajectory. No images, transformer or representation pretraining.'},
        'metrics': {'initial_loss': losses[0], 'final_loss': losses[-1], 'final_mean_absolute_error': errors[-1],
                    'target_oracle_error': oracle_error, 'ema_oracle_error': float((teacher - expected_ema).abs().max())},
        'charts': [{'kind': 'line', 'title': 'Masked target fitting', 'x_label': 'Update', 'x': steps,
                    'series': [{'name': 'Submitted smooth L1', 'values': losses}, {'name': 'Measured absolute error', 'values': errors}]},
                   {'kind': 'line', 'title': 'Separate fixed-online momentum example', 'x_label': 'Update', 'x': steps,
                    'series': [{'name': 'Target-online parameter distance', 'values': ema_gaps}]}]}
