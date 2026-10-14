"""LeetPaper teaching adaptation of GKD lambda=1 on a three-token Markov LM."""
import torch


def run(sample_fn, loss_fn, seed=42, steps=120):
    generator = torch.Generator().manual_seed(seed)
    teacher = torch.tensor([[.1, .8, .1], [.1, .1, .8], [.8, .1, .1], [.6, .3, .1]]).log()
    student = torch.zeros_like(teacher, requires_grad=True)
    optimizer = torch.optim.Adam([student], lr=.12)
    curve, visits = [], torch.zeros(4)
    for step in range(steps + 1):
        with torch.no_grad():
            kl = (teacher.exp() * (teacher - student.log_softmax(-1))).sum(-1).mean()
        curve.append(dict(step=step, kl=float(kl)))
        if step == steps:
            break
        tokens, prefixes = sample_fn(student, 48, 5, generator)
        visits += torch.bincount(prefixes.flatten(), minlength=4)
        # Crucial: teacher and student are queried at the SAME student prefixes.
        loss = loss_fn(student[prefixes], teacher[prefixes], torch.ones_like(tokens), direction='reverse')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    out = dict(provenance='LeetPaper teaching adaptation of Algorithm 1, lambda=1',
                final_kl=curve[-1]['kl'], training=curve, prefix_visits=visits.tolist(),
                student=student.softmax(-1).detach().tolist(), teacher=teacher.exp().tolist(),
                charts=[dict(kind='line', title='Teacher match at student prefixes',
                             x=[row['step'] for row in curve],
                             series=[dict(name='Mean forward KL (evaluation)', values=[row['kl'] for row in curve])])])

    out['loss_curve'] = [[row['step'], row['kl']] for row in out['training']]
    out['metrics'] = {key: value for key, value in out.items() if key in ('final_kl', 'final_mse', 'merge_error', 'quantization_rmse', 'adapter_parameters', 'base_parameters', 'code_count', 'scale_count')}
    out.update(kind='toy_experiment', title='Student-prefix distillation', provenance={'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/html/2306.13649v3#S3', 'adaptation': 'LeetPaper fixed-length categorical Markov model, not the original T5 or Gaussian-mixture experiment.'})
    return out
