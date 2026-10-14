import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.ijepa.experiment import run


@pytest.mark.t0
def test_shape_dtype_no_mutation(impl):
    """Targets expand K target masks across C contexts without changing input tensors."""
    for dtype in (torch.float32, torch.float64):
        x = torch.arange(30, dtype=dtype).reshape(2, 3, 5).transpose(1, 2)
        masks = [torch.tensor([[4, 1], [2, 0]]), torch.tensor([[0, 3], [1, 4]])]
        before = x.clone(); out = impl.masked_targets(x, masks, 3)
        assert out.shape == (12, 2, 3) and out.dtype == dtype and not out.requires_grad
        close(x, before)
        t = torch.ones(2, dtype=dtype); q = torch.zeros(2, dtype=dtype)
        got = impl.ema_update(t, q, .8); close(t, torch.ones_like(t)); close(q, torch.zeros_like(q))
        assert got.dtype == dtype and impl.masked_prediction_loss(out, out).shape == ()


@pytest.mark.t0
def test_validation(impl):
    """Supplied guards reject invalid masks, target shapes, context count and EMA momentum."""
    x = torch.randn(2, 4, 3); mask = torch.tensor([[0, 1], [2, 3]])
    for masks in ([], [mask.float()], [mask + 4], [mask, mask[:, :1]]):
        with pytest.raises(ValueError): impl.masked_targets(x, masks)
    with pytest.raises(ValueError): impl.masked_targets(x, [mask], 0)
    with pytest.raises(ValueError): impl.masked_prediction_loss(x, x[:1])
    for m in (-.1, 1.1, float('nan')):
        with pytest.raises(ValueError): impl.ema_update(x, x, m)


@pytest.mark.t1
def test_target_scalar_oracle_and_context_order(impl):
    """Per-token scalar layer norm and explicit block/context/batch loops distinguish repeat from repeat_interleave."""
    x = torch.tensor([[[1., 3., 6.], [2., 2., 2.], [-3., 2., 4.]], [[7., 1., 0.], [3., 6., -2.], [-1., -2., 4.]]], dtype=torch.float64)
    masks = [torch.tensor([[2, 0], [1, 2]]), torch.tensor([[1, 2], [0, 1]])]
    expected = []
    for mask in masks:
        for context in range(2):
            for batch in range(2):
                row = []
                for index in mask[batch].tolist():
                    values = x[batch, index].tolist(); mean = sum(values) / 3
                    variance = sum((v - mean)**2 for v in values) / 3
                    row.append([(v - mean) / math.sqrt(variance + 1e-5) for v in values])
                expected.append(row)
    close(impl.masked_targets(x, masks, 2), x.new_tensor(expected), atol=1e-12, rtol=1e-12)


@pytest.mark.t1
def test_smooth_l1_and_ema_scalar_oracles(impl):
    """Residuals below, at and above beta=1 check the released code loss convention; unequal momentum catches reversal."""
    p = torch.tensor([[-2., -.3, 0.], [.5, 1., 4.]], dtype=torch.float64)
    t = torch.zeros_like(p)
    values = [(.5*v*v if abs(v) < 1 else abs(v) - .5) for v in p.flatten().tolist()]
    close(impl.masked_prediction_loss(p, t), p.new_tensor(sum(values) / len(values)), atol=1e-12, rtol=1e-12)
    target = p.new_tensor([2., -3.]); online = p.new_tensor([-1., 5.])
    close(impl.ema_update(target, online, .8), p.new_tensor([1.4, -1.4]), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_target_and_loss_gradient_boundaries(impl):
    """The teacher path is frozen inside both operations, but prediction gradients match the piecewise analytic derivative."""
    x = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    target = impl.masked_targets(x, [torch.tensor([[0, 1], [2, 3]])])
    assert not target.requires_grad
    p = torch.tensor([[-2., -.3, 0.], [.5, 1.3, 4.]], dtype=torch.float64, requires_grad=True)
    t = torch.zeros_like(p, requires_grad=True)
    gp, gt = torch.autograd.grad(impl.masked_prediction_loss(p, t), (p, t), allow_unused=True)
    close(gp, p.detach().clamp(-1, 1) / p.numel(), atol=1e-12, rtol=1e-12)
    assert gt is None or not gt.any()


@pytest.mark.t2
def test_ema_endpoints_freezing_and_recurrence(impl):
    """Functional EMA never mutates or backpropagates into parameters; constant-online recurrence has a closed form."""
    initial = torch.tensor([2., -3.], dtype=torch.float64, requires_grad=True)
    online = torch.tensor([-1., 5.], dtype=torch.float64, requires_grad=True)
    close(impl.ema_update(initial, online, 0.), online)
    close(impl.ema_update(initial, online, 1.), initial)
    current = initial
    for _ in range(4): current = impl.ema_update(current, online, .7)
    close(current, .7**4 * initial + (1 - .7**4) * online, atol=1e-12, rtol=1e-12)
    assert not current.requires_grad
    close(initial, initial.new_tensor([2., -3.])); close(online, online.new_tensor([-1., 5.]))


@pytest.mark.t2
def test_token_shift_invariance_and_feature_permutation(impl):
    """Layer norm is per-token: arbitrary token offsets cancel, and feature permutations commute with extraction."""
    x = torch.randn(2, 4, 3, dtype=torch.float64); masks = [torch.tensor([[0, 1], [2, 3]])]
    got = impl.masked_targets(x, masks)
    shifted = impl.masked_targets(x + torch.arange(8, dtype=x.dtype).reshape(2, 4, 1), masks)
    close(shifted, got, atol=1e-12, rtol=1e-12)
    permuted = impl.masked_targets(x[..., [2, 0, 1]], masks)
    close(permuted, got[..., [2, 0, 1]], atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_loss_permutation_duplication_and_exact_match(impl):
    """Mean reduction preserves loss under duplication, and matching targets have exactly zero loss/gradient."""
    p = torch.tensor([-.3, 2., 4.], dtype=torch.float64, requires_grad=True); t = p.detach() + .4
    base = impl.masked_prediction_loss(p, t)
    close(impl.masked_prediction_loss(p.repeat(3), t.repeat(3)), base)
    loss = impl.masked_prediction_loss(p, p.detach()); close(loss, p.new_zeros(()))
    grad, = torch.autograd.grad(loss, p); close(grad, torch.zeros_like(p))


@pytest.mark.t3
def test_masked_prediction_experiment(impl, artifact):
    """Submitted target, loss and EMA operations drive latent fitting and a checked momentum trajectory."""
    result = run(impl.masked_targets, impl.masked_prediction_loss, impl.ema_update)
    artifact('ijepa_masked_latents', result); m = result['metrics']
    assert m['target_oracle_error'] < 1e-12 and m['ema_oracle_error'] < 1e-12
    assert m['final_loss'] < .01 * m['initial_loss'] and m['final_mean_absolute_error'] < .01
