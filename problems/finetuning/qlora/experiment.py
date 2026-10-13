"""PaperLeet teaching adaptation: frozen NF4 base and trainable rank-two correction."""
import torch


def run(quantize_fn, dequantize_fn, linear_fn, seed=42, steps=220):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(6, 8, generator=generator) * .3
    codes, scales = quantize_fn(weight, 16)
    base = dequantize_fn(codes, scales, 16)
    # Independent target decode: a wrong decoder must not redefine the task.
    levels = weight.new_tensor([-1., -.6961928, -.52507305, -.39491749, -.28444138, -.18477343,
                               -.09105004, 0., .07958030, .16093020, .24611230, .33791524,
                               .44070983, .56261700, .72295684, 1.])
    target_base = (levels[codes.long().flatten()] * scales.repeat_interleave(16)).reshape_as(weight)
    delta = torch.randn(6, 2, generator=generator) @ torch.randn(2, 8, generator=generator) * .15
    x = torch.randn(96, 8, generator=generator)
    y = x @ (target_base + delta).T
    a = (torch.randn(2, 8, generator=generator) * .1).requires_grad_()
    b = torch.zeros(6, 2, requires_grad=True)
    optimizer = torch.optim.Adam([a, b], lr=.06)
    curve = []
    for step in range(steps + 1):
        prediction = linear_fn(x, codes, scales, a, b, alpha=4., block_size=16)
        loss = (prediction - y).square().mean()
        curve.append(dict(step=step, mse=float(loss.detach())))
        if step == steps:
            break
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    out = dict(provenance='PaperLeet NF4 adapter regression; unpacked indices, no paged optimizer',
                final_mse=curve[-1]['mse'], training=curve,
                quantization_rmse=float((weight - base).square().mean().sqrt()),
                code_count=codes.numel(), scale_count=scales.numel(),
                charts=[dict(kind='line', title='Adapt a frozen NF4 base', x=[p['step'] for p in curve],
                             series=[dict(name='Regression MSE', values=[p['mse'] for p in curve])])])

    out['loss_curve'] = [[row['step'], row['mse']] for row in out['training']]
    out['metrics'] = {key: value for key, value in out.items() if key in ('final_kl', 'final_mse', 'merge_error', 'quantization_rmse', 'adapter_parameters', 'base_parameters', 'code_count', 'scale_count')}
    out.update(kind='toy_experiment', title='Adapt a frozen NF4 base', provenance={'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/html/2305.14314v1#S3', 'adaptation': 'PaperLeet synthetic regression with unpacked NF4 indices; does not measure full QLoRA memory savings.'})
    return out
