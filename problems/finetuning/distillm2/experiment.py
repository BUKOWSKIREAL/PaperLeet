"""Small conditional-vocabulary optimization; every update uses the submission."""
import torch


def run(skew_kl, skew_reverse_kl, distillm2_loss):
    g = torch.Generator().manual_seed(42)
    p_t = torch.tensor([[[3., 1., -2.], [0., 2., -1.]]], dtype=torch.float64)
    p_s = torch.tensor([[[-1., 1., 2.], [2., -2., 0.]]], dtype=torch.float64)
    q_t = torch.randn(1, 2, 3, generator=g, dtype=torch.float64, requires_grad=True)
    q_s = torch.randn(1, 2, 3, generator=g, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    losses, tgaps, sgaps = [], [], []
    for step in range(61):
        loss = distillm2_loss(p_t, q_t, mask, p_s, q_s, mask, .15, .35, .6)
        losses.append(float(loss.detach()))
        tgaps.append(float(skew_kl(p_t, q_t, .15).mean().detach()))
        sgaps.append(float(skew_reverse_kl(p_s, q_s, .35).mean().detach()))
        if step < 60:
            grads = torch.autograd.grad(loss, (q_t, q_s))
            with torch.no_grad():
                q_t -= 2.0 * grads[0]
                q_s -= 2.0 * grads[1]
    return dict(kind='toy_experiment', title='Two response sources, two divergences',
                problem='finetuning/distillm2',
                provenance=dict(kind='teaching_adaptation', source='https://arxiv.org/html/2503.07067v2',
                                anchor='Eq. (2)', adaptation='Two synthetic conditional contexts per response source; full-vocabulary KL, fixed coefficients, no language model or curriculum.'),
                metrics=dict(initial_loss=losses[0], final_loss=losses[-1], loss_ratio=losses[-1]/losses[0]),
                charts=[dict(kind='line', title='Submitted objective during optimization', x_label='Update', x=list(range(61)),
                             series=[dict(name='Eq. 2 objective', values=losses), dict(name='Teacher-response SKL', values=tgaps),dict(name='Student-response SRKL', values=sgaps)])])
