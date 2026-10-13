import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.rl.glm5_offpolicy.experiment import run


@pytest.mark.t0
def test_contract_and_empty_mask(impl):
    """Weights preserve shape/dtype and an empty valid set yields differentiable zero."""
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    old = torch.zeros_like(x, requires_grad=True)
    empty = torch.zeros_like(x)
    weights = impl.masked_is_weights(x, old, empty)
    assert weights.shape == x.shape and weights.dtype == x.dtype
    assert not weights.requires_grad
    close(weights, torch.zeros_like(x))
    loss = impl.offpolicy_loss(x, old, torch.ones(2, dtype=x.dtype), empty)
    assert loss.ndim == 0
    close(loss, torch.zeros((), dtype=x.dtype))
    loss.backward()
    close(x.grad, torch.zeros_like(x))
    for args in [(x, old[:, :1], empty, .2, .2), (x, old, empty, 1., .2), (x, old, empty, .2, -1.)]:
        with pytest.raises(ValueError):
            impl.masked_is_weights(*args)


@pytest.mark.t1
def test_scalar_loop_oracle(impl):
    """Independent scalar-loop evaluation fixes the original valid-token denominator."""
    old = torch.full((2, 4), -2., dtype=torch.float64)
    ratios = torch.tensor([[.9, 1.1, .2, 9.], [1.05, 1.3, .95, .7]], dtype=old.dtype)
    x = old + ratios.log()
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=old.dtype)
    advantages = torch.tensor([2., -3.], dtype=old.dtype)
    expected_weights, total, count = torch.zeros_like(x), 0., 0
    for i in range(2):
        for j in range(4):
            if mask[i, j]:
                count += 1
                r = math.exp(float(x[i, j] - old[i, j]))
                if .8 < r < 1.2:
                    expected_weights[i, j] = r
                    total -= r * float(advantages[i]) * float(x[i, j])
    close(impl.masked_is_weights(x, old, mask), expected_weights, atol=1e-12, rtol=1e-12)
    close(impl.offpolicy_loss(x, old, advantages, mask), x.new_tensor(total / count), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_strict_boundary_and_asymmetric_widths(impl):
    """The exact endpoints are excluded, unlike current Slime's inclusive MIS helper."""
    ratios = torch.tensor([[.25, .5, .75, 1., 1.5, 2., 3.]], dtype=torch.float64)
    x = ratios.log()
    expected = torch.tensor([[0., 0., .75, 1., 1.5, 0., 0.]], dtype=x.dtype)
    close(impl.masked_is_weights(x, torch.zeros_like(x), torch.ones_like(x), .5, 1.), expected)
    for dtype in (torch.float32, torch.float64):
        for low, high in [(.2, .2), (.125, .75), (.5, 1.)]:
            endpoints = torch.tensor([[1 - low, 1 + high]], dtype=dtype)
            log_endpoints = endpoints.log()
            close(impl.masked_is_weights(log_endpoints, torch.zeros_like(log_endpoints), torch.ones_like(log_endpoints), low, high), torch.zeros_like(endpoints), atol=0., rtol=0.)



@pytest.mark.t2
def test_detached_weight_advantage_and_rollout_gradient(impl):
    """Only direct current-logp factors differentiate, giving -weight*advantage/N."""
    ratios = torch.tensor([[.9, 1.1, 2.], [1.05, .5, 1.]], dtype=torch.float64)
    x = (ratios.log() - 2).requires_grad_()
    old = torch.full_like(x, -2., requires_grad=True)
    advantages = torch.tensor([2., -3.], dtype=x.dtype, requires_grad=True)
    mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]], dtype=x.dtype)
    loss = impl.offpolicy_loss(x, old, advantages, mask)
    loss.backward()
    expected = torch.tensor([[-.36, -.44, 0.], [.63, 0., 0.]], dtype=x.dtype)
    close(x.grad, expected, atol=1e-12, rtol=1e-12)
    assert old.grad is None or not torch.count_nonzero(old.grad)
    assert advantages.grad is None or not torch.count_nonzero(advantages.grad)


@pytest.mark.t2
def test_padding_extremes_and_input_immutability(impl):
    """NaN padding and extreme valid off-policy ratios cannot poison weights or gradients."""
    x = torch.tensor([[-2., 10000., -10000., float('nan')]], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([[-2., -2., -2., float('nan')]], dtype=x.dtype)
    mask = torch.tensor([[1., 1., 1., 0.]], dtype=x.dtype)
    advantages = torch.ones(1, dtype=x.dtype)
    before = x.detach().clone()
    weights = impl.masked_is_weights(x, old, mask)
    close(weights, x.new_tensor([[1., 0., 0., 0.]]))
    loss = impl.offpolicy_loss(x, old, advantages, mask)
    close(loss, x.new_tensor(2 / 3))
    loss.backward()
    close(x.grad, x.new_tensor([[-1/3, 0., 0., 0.]]))
    assert torch.equal(torch.isnan(x), torch.isnan(before))
    close(x.detach()[:, :3], before[:, :3])


@pytest.mark.t2
def test_valid_token_reduction_and_replication(impl):
    """Duplicating the complete token population preserves the loss; dropping differs from renormalizing."""
    x = torch.tensor([[-2., -2.], [-1., -10.]], dtype=torch.float64)
    old = torch.full_like(x, -2.)
    mask = torch.ones_like(x)
    adv = torch.tensor([1., -1.], dtype=x.dtype)
    expected = x.new_tensor(1.)  # two retained logp=-2 contributions / four valid tokens
    close(impl.offpolicy_loss(x, old, adv, mask), expected)
    close(impl.offpolicy_loss(x.repeat(1, 3), old.repeat(1, 3), adv, mask.repeat(1, 3)), expected)


@pytest.mark.t3
def test_stale_rollout_bandit_and_ratio_sweep(impl, artifact):
    """Submitted weights/gradients obey the rule and improve a two-action stale-rollout bandit."""
    result = run(impl.masked_is_weights, impl.offpolicy_loss)
    artifact("offpolicy_bandit", result)
    assert result["metrics"]["max_gradient_error"] < 1e-12
    assert result["metrics"]["max_weight_error"] < 1e-12
    assert result["metrics"]["initial_reward"] == 0.
    assert result["metrics"]["final_reward"] > .9
