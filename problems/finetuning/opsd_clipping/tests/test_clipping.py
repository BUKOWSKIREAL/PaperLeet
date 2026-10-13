import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.finetuning.opsd_clipping.experiment import run


def scalar_loss(p, q, tau, reverse=False):
    return sum(min(a * math.log(a / b), tau) for a, b in zip(p if reverse else q, q if reverse else p))


@pytest.mark.t0
def test_contract_and_validation(impl):
    """Scalar output preserves dtype/device; supplied validation rejects invalid thresholds/directions."""
    for dtype in (torch.float32, torch.float64):
        x = torch.zeros(2, 3, 4, dtype=dtype)
        mask = torch.ones(2, 3, dtype=torch.bool)
        actual = impl.clipped_distillation_loss(x, x, mask, .1)
        assert actual.shape == () and actual.dtype == dtype and actual.device == x.device
        for tau in (0, -1, float('nan'), float('inf')):
            with pytest.raises(ValueError):
                impl.clipped_distillation_loss(x, x, mask, tau)
        with pytest.raises(ValueError):
            impl.clipped_distillation_loss(x, x, mask, .1, 'js')


@pytest.mark.t1
def test_signed_pointwise_clipping_scalar_oracle(impl):
    """Clip individual signed vocabulary terms before summation, in both KL directions."""
    p, q = [.03, .17, .8], [.8, .15, .05]
    s, t = torch.tensor(p, dtype=torch.float64).log()[None, None], torch.tensor(q, dtype=torch.float64).log()[None, None]
    for reverse in (False, True):
        for tau in (.01, .3, 10.):
            actual = impl.clipped_distillation_loss(s, t, torch.ones(1, 1, dtype=torch.bool), tau, 'reverse' if reverse else 'forward')
            close(actual, torch.tensor(scalar_loss(p, q, tau, reverse), dtype=s.dtype), atol=1e-12, rtol=1e-12)
    assert scalar_loss(p, q, .01) < 0  # The clipped objective need not be a divergence.


@pytest.mark.t1
def test_response_weighting_and_mask(impl):
    """Each response has equal weight despite different token counts; empty responses contribute zero."""
    s = torch.tensor([[[.8, .2], [.1, .9]], [[.3, .7], [.9, .1]], [[.6, .4], [.2, .8]]], dtype=torch.float64).log()
    t = torch.tensor([.4, .6], dtype=s.dtype).log().expand_as(s)
    mask = torch.tensor([[True, True], [True, False], [False, False]])
    expected = ((scalar_loss([.8,.2],[.4,.6],.2)+scalar_loss([.1,.9],[.4,.6],.2))/2 + scalar_loss([.3,.7],[.4,.6],.2))/3
    close(impl.clipped_distillation_loss(s, t, mask, .2), torch.tensor(expected, dtype=s.dtype))


@pytest.mark.t2
def test_finite_difference_gradient_and_teacher_stop(impl):
    """Finite differences away from clipping boundaries match student gradients; teacher receives none."""
    s = torch.tensor([[[.2, -.5, 1.1]]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[[1.4, .2, -.3]]], dtype=s.dtype, requires_grad=True)
    mask = torch.ones(1, 1, dtype=torch.bool)
    for direction in ('forward', 'reverse'):
        loss = impl.clipped_distillation_loss(s, t, mask, .2, direction)
        gs, gt = torch.autograd.grad(loss, (s, t), allow_unused=True)
        assert gt is None
        def oracle(x):
            return scalar_loss(x.softmax(-1).flatten().tolist(), t.detach().softmax(-1).flatten().tolist(), .2, direction == 'reverse')
        finite = []
        for i in range(3):
            plus, minus = s.detach().clone(), s.detach().clone()
            plus[0,0,i] += 1e-5; minus[0,0,i] -= 1e-5
            finite.append((oracle(plus)-oracle(minus))/2e-5)
        close(gs.flatten(), torch.tensor(finite, dtype=s.dtype), atol=1e-9, rtol=1e-7)


@pytest.mark.t2
def test_empty_mask_is_differentiable_zero(impl):
    """All masked batches and zero-length responses yield zero student gradient."""
    for width in (0, 3):
        s = torch.randn(2, width, 4, dtype=torch.float64, requires_grad=True)
        loss = impl.clipped_distillation_loss(s, s.detach() + .2, torch.zeros(2, width, dtype=torch.bool), .1)
        assert loss.item() == 0
        close(torch.autograd.grad(loss, s)[0], torch.zeros_like(s))


@pytest.mark.t2
def test_logit_shift_and_extreme_finite_inputs(impl):
    """Softmax shift invariance and stable large-logit evaluation hold without input mutation."""
    s = torch.tensor([[[1000., -1000., 0.]]], dtype=torch.float64)
    t = -s; before = s.clone(), t.clone(); mask = torch.ones(1,1,dtype=torch.bool)
    for direction in ('forward','reverse'):
        value = impl.clipped_distillation_loss(s, t, mask, .5, direction)
        assert torch.isfinite(value)
        close(value, impl.clipped_distillation_loss(s+300, t-200, mask, .5, direction))
    close(s,before[0]); close(t,before[1])


@pytest.mark.t3
def test_categorical_threshold_sweep(impl, artifact):
    """The submitted loss reproduces the pointwise threshold sweep against scalar arithmetic."""
    result = run(impl.clipped_distillation_loss)
    artifact('clipping_thresholds', result)
    assert result['metrics']['maximum_formula_error'] < 1e-12
    assert result['metrics']['small_tau_forward_loss'] < 0
