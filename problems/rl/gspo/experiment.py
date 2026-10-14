"""PaperLeet teaching adaptation: heterogeneous token ratios and sequence clipping."""
import torch


def run(ratio_fn, loss_fn, seed=42):
    del seed  # This diagnostic is analytic and deterministic.
    ratios = torch.tensor([[2., .5, 1.], [1.5, 1.5, 1.5], [.5, .5, .5]])
    mask = torch.tensor([[1., 1., 0.], [1., 1., 1.], [1., 1., 1.]])
    logp = ratios.log().requires_grad_()
    old = torch.zeros_like(logp)
    advantages = torch.tensor([1., 1., -1.])
    sequence = ratio_fn(logp, old, mask)
    loss = loss_fn(logp, old, advantages, mask)
    gradient = torch.autograd.grad(loss, logp)[0]
    out = dict(provenance='PaperLeet teaching adaptation; not a paper benchmark',
                sequence_ratios=sequence.detach().tolist(), token_ratios=ratios.tolist(),
                gradient=gradient.tolist(), loss=float(loss.detach()),
                charts=[dict(kind='bar', title='Token ratios cancel in log space',
                             x=['canceling tokens', 'positive clipped', 'negative clipped'],
                             series=[dict(name='Sequence ratio', values=sequence.detach().tolist())])])

    out['metrics'] = {'canceling_sequence_ratio': out['sequence_ratios'][0], 'loss': out['loss']}
    out.update(kind='toy_experiment', title='Sequence clipping and canceling token ratios', provenance={'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/html/2507.18071v1#S4.SS1', 'adaptation': 'Analytic three-response diagnostic written for PaperLeet; not Qwen training results.'})
    return out
