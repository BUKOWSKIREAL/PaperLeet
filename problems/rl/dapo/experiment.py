"""LeetPaper teaching adaptation: dynamic group filtering and a clip sweep."""
import torch


def run(group_fn, loss_fn, seed=42):
    del seed
    rewards = torch.tensor([[0., 0., 0., 0.], [0., 1., 0., 1.], [1., 1., 1., 1.]])
    kept = group_fn(rewards)
    ratios = torch.linspace(.5, 1.5, 41)
    positive, negative = [], []
    for ratio in ratios:
        logp = ratio.log().reshape(1, 1)
        old, mask = torch.zeros_like(logp), torch.ones_like(logp)
        positive.append(float(loss_fn(logp, old, torch.ones(1), mask)))
        negative.append(float(loss_fn(logp, old, -torch.ones(1), mask)))
    out = dict(provenance='LeetPaper teaching adaptation; no AIME reproduction',
                retained=kept.tolist(), ratios=ratios.tolist(), positive_loss=positive, negative_loss=negative,
                charts=[dict(kind='line', title='Asymmetric policy clipping', x_label='importance ratio', y_label='loss', x=ratios.tolist(),
                             series=[dict(name='Positive advantage', values=positive),
                                     dict(name='Negative advantage', values=negative)])])

    out['x_label'] = 'importance ratio'
    out['loss_curve'] = list(zip(out['ratios'], out['positive_loss']))
    out['metrics'] = {'retained_groups': sum(out['retained']), 'upper_plateau_loss': out['positive_loss'][-1], 'lower_plateau_loss': out['negative_loss'][0]}
    out.update(kind='toy_experiment', title='Dynamic groups and asymmetric clipping', provenance={'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/html/2503.14476v1#S3', 'adaptation': 'LeetPaper synthetic binary-reward groups and ratio sweep; no AIME reproduction.'})
    return out
