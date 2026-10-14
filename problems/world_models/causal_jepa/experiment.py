"""Reconstruct a hidden object trajectory from an interacting visible object."""
import torch


def run(object_mask_inputs, masked_prediction_loss):
    dtype = torch.float64
    b, history_length, total, slots, d = 16, 4, 6, 2, 1
    velocities = torch.linspace(-1., 1., b, dtype=dtype)
    time = torch.arange(total, dtype=dtype)
    targets = torch.zeros(b, total, slots, d, dtype=dtype)
    targets[:, :, 0, 0] = velocities[:, None] * time
    targets[:, :, 1, 0] = .5 * velocities[:, None] * time.square()
    mask = torch.zeros(b, slots, dtype=torch.bool); mask[:, 1] = True
    inputs, target_mask = object_mask_inputs(targets[:, :history_length], mask,
        torch.ones(1, 1, dtype=dtype), torch.zeros(total, 1, dtype=dtype), torch.zeros(1, dtype=dtype))
    # A supplied linear probe learns each time/slot output from object 0's visible motion.
    # No target history from masked object 1 is used as a feature.
    features = inputs[:, 1, 0, 0]
    coefficients = torch.zeros(total, slots, dtype=dtype, requires_grad=True)
    optimizer = torch.optim.Adam([coefficients], lr=.25)
    curves, h_curve, f_curve = [], [], []
    for step in range(101):
        predictions = features[:, None, None, None] * coefficients[None, :, :, None]
        loss, history, future = masked_prediction_loss(predictions, targets, history_length, mask)
        curves.append(loss.item()); h_curve.append(history.item()); f_curve.append(future.item())
        if step < 100:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    # Perturb exactly the information the masking is meant to hide.
    changed = targets[:, :history_length].clone(); changed[:, 1:, 1] += 100.
    altered, _ = object_mask_inputs(changed, mask, torch.ones(1, 1, dtype=dtype),
        torch.zeros(total, 1, dtype=dtype), torch.zeros(1, dtype=dtype))
    return {
        'kind': 'toy_experiment', 'problem': 'world_models/causal_jepa', 'title': 'An object hidden across history',
        'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2602.11389',
            'anchor': 'Section 4.2 Eqs. (3)-(6)',
            'adaptation': 'Sixteen synthetic two-object sequences use one visible velocity to predict a coupled object trajectory. A supplied linear probe replaces the transformer and frozen scalar slots replace an image encoder. This checks observability and objectives, not causal identification or CLEVRER performance.'},
        'metrics': {'initial_loss': curves[0], 'final_loss': curves[-1], 'masked_history_mse': h_curve[-1],
            'future_mse': f_curve[-1], 'hidden_history_leakage': (inputs - altered).abs().max().item(),
            'masked_tokens_per_sample': float(target_mask[0].sum())},
        'charts': [
            {'kind': 'line', 'title': 'Joint masked prediction', 'x_label': 'Step', 'x': list(range(0, 101, 2)),
                'series': [{'name': 'History MSE', 'values': h_curve[::2]}, {'name': 'Future MSE', 'values': f_curve[::2]}]},
            {'kind': 'line', 'title': 'Hidden object trajectory', 'x_label': 'Time', 'x': list(range(total)),
                'series': [{'name': 'Predicted', 'values': predictions[-1, :, 1, 0].detach().tolist()}, {'name': 'Target', 'values': targets[-1, :, 1, 0].tolist()}]},
            {'kind': 'line', 'title': 'Target mask for hidden object', 'x_label': 'Time', 'x': list(range(total)),
                'series': [{'name': '1 = masked target', 'values': target_mask[0, :, 1].int().tolist()}]},
        ],
    }
