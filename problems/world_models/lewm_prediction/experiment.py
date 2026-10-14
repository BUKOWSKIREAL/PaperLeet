"""Tiny action-conditioned dynamics: a teaching adaptation of LeWM, not pixel training."""
import torch


def run(lewm_objective, latent_rollout):
    generator = torch.Generator().manual_seed(42)
    dtype = torch.float64
    initial = torch.randn(24, 1, generator=generator, dtype=dtype)
    actions = torch.randn(24, 4, 1, generator=generator, dtype=dtype) * .3
    states = [initial]
    for t in range(4):
        states.append(.8 * states[-1] + .6 * actions[:, t])
    observations = torch.stack(states, 1)
    encoder_scale = torch.tensor(1.1, dtype=dtype, requires_grad=True)
    dynamics = torch.tensor([.3, -.2], dtype=dtype, requires_grad=True)
    optimizer = torch.optim.Adam([encoder_scale, dynamics], lr=.04)
    knots = torch.linspace(0., 3., 17, dtype=dtype)
    window = torch.exp(-knots.square() / 2)
    weights = torch.full_like(knots, 2 * 3 / 16); weights[[0, -1]] = 3 / 16

    def regularizer(time_batch_features):
        # D=1, unit projection +1: the fixed-direction official SIGReg quadrature.
        phase = time_batch_features[..., 0, None] * knots
        error = (phase.cos().mean(1) - window).square() + phase.sin().mean(1).square()
        return (error @ (weights * window)).mean() * time_batch_features.shape[1]

    losses, pred_losses, scales = [], [], []
    for step in range(121):
        embeddings = encoder_scale * observations
        predictor = lambda z, a: dynamics[0] * z + dynamics[1] * a
        total, prediction, _ = lewm_objective(embeddings, actions, predictor, regularizer, .02)
        losses.append(total.item()); pred_losses.append(prediction.item()); scales.append(encoder_scale.item())
        if step < 120:
            optimizer.zero_grad(); total.backward(); optimizer.step()
    predicted = latent_rollout(encoder_scale * initial, actions, lambda z, a: dynamics[0] * z + dynamics[1] * a)
    truth = encoder_scale * observations[:, 1:]
    return {
        'kind': 'toy_experiment', 'problem': 'world_models/lewm_prediction',
        'title': 'Learn action-conditioned latent dynamics',
        'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/abs/2603.19312',
            'anchor': 'Section 3.1 Eqs. (1)-(3), Appendix A',
            'adaptation': 'A trainable scalar encoder and linear N=1 predictor fit 24 deterministic one-dimensional trajectories. The supplied single-projection SIGReg uses the official [0,3] quadrature. This is not a pixel-world-model benchmark.'},
        'metrics': {'initial_prediction_loss': pred_losses[0], 'final_prediction_loss': pred_losses[-1],
            'rollout_mse': (predicted - truth).square().mean().item(), 'encoder_scale': encoder_scale.item(),
            'learned_state_coefficient': dynamics[0].item(), 'learned_action_coefficient': dynamics[1].item()},
        'charts': [
            {'kind': 'line', 'title': 'Joint training', 'x_label': 'Step', 'x': list(range(0, 121, 3)),
                'series': [{'name': 'Prediction MSE', 'values': pred_losses[::3]}, {'name': 'Total objective', 'values': losses[::3]}]},
            {'kind': 'line', 'title': 'Autoregressive training trajectory', 'x_label': 'Future step', 'x': [1, 2, 3, 4],
                'series': [{'name': 'Prediction', 'values': predicted[0, :, 0].detach().tolist()}, {'name': 'Encoded observation', 'values': truth[0, :, 0].detach().tolist()}]},
            {'kind': 'line', 'title': 'Encoder scale', 'x_label': 'Step', 'x': list(range(0, 121, 3)),
                'series': [{'name': 'Jointly learned encoder', 'values': scales[::3]}]},
        ],
    }
