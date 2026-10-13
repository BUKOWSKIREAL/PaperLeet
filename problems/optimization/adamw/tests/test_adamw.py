import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.optimization.adamw.experiment import run


@pytest.mark.t0
def test_shapes_dtype_and_no_mutation(impl):
    """Functional optimizer preserves tensor shape/dtype and leaves every input untouched."""
    args = [torch.randn(2, 3, dtype=torch.float64) for _ in range(3)]
    args.append(torch.rand(2, 3, dtype=torch.float64))
    before = [x.clone() for x in args]
    output = impl.adamw_step(*args, step=3)
    assert len(output) == 3
    for x in output:
        assert x.shape == (2, 3) and x.dtype == torch.float64
    for x, original in zip(args, before):
        close(x, original)


@pytest.mark.t1
def test_first_step_analytic_and_epsilon(impl):
    """At step one corrected moments are gradient and gradient squared, with eps outside sqrt."""
    p = torch.tensor([2., -3.], dtype=torch.float64)
    g = torch.tensor([1e-5, -2.], dtype=p.dtype)
    updated, m, v = impl.adamw_step(p, g, torch.zeros_like(p), torch.zeros_like(p), 1,
                                    lr=.2, betas=(.7, .8), eps=1e-3, weight_decay=.4)
    close(updated, p * .92 - .2 * g / (g.abs() + 1e-3), atol=1e-12, rtol=1e-12)
    close(m, .3 * g)
    close(v, .2 * g.square())


@pytest.mark.t2
def test_zero_gradient_decay_and_jacobian(impl):
    """Without gradient or history, decay scales every parameter equally and has a known Jacobian."""
    p = torch.tensor([1., 10., -3.], dtype=torch.float64, requires_grad=True)
    zero = torch.zeros_like(p)
    updated, m, v = impl.adamw_step(p, zero, zero, zero, 1, lr=.1, weight_decay=.2)
    close(updated, p * .98)
    close(m, zero)
    close(v, zero)
    close(torch.autograd.grad(updated.sum(), p)[0], torch.full_like(p, .98))


@pytest.mark.t2
def test_decay_does_not_enter_moments(impl):
    """Changing weight decay alters parameters by lr*delta_decay*p and leaves optimizer state identical."""
    p, g, m, v = torch.randn(4), torch.randn(4), torch.randn(4), torch.rand(4)
    without = impl.adamw_step(p, g, m, v, 7, lr=.03, weight_decay=0)
    with_decay = impl.adamw_step(p, g, m, v, 7, lr=.03, weight_decay=.5)
    close(with_decay[0] - without[0], -.015 * p)
    close(with_decay[1], without[1])
    close(with_decay[2], without[2])


@pytest.mark.t2
def test_matches_independent_torch_optimizer(impl):
    """Multiple updates, nondefault betas and changing learning rates agree with torch.optim.AdamW."""
    p = torch.tensor([2., -4., 1.], dtype=torch.float64)
    reference = p.clone().requires_grad_()
    optimizer = torch.optim.AdamW([reference], lr=.05, betas=(.65, .8), eps=1e-5, weight_decay=.13)
    m, v = torch.zeros_like(p), torch.zeros_like(p)
    for step in range(1, 9):
        g = torch.tensor([(-1.) ** step, step / 3, -0.1], dtype=p.dtype)
        lr = .05 / step
        optimizer.param_groups[0]["lr"] = lr
        reference.grad = g.clone()
        optimizer.step()
        p, m, v = impl.adamw_step(p, g, m, v, step, lr=lr, betas=(.65, .8), eps=1e-5, weight_decay=.13)
        close(p, reference.detach(), atol=1e-12, rtol=1e-12)
        close(m, optimizer.state[reference]["exp_avg"], atol=1e-12, rtol=1e-12)
        close(v, optimizer.state[reference]["exp_avg_sq"], atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_parameter_permutation_equivariance(impl):
    """Reordering parameter coordinates and all associated state simply reorders the result."""
    tensors = [torch.randn(5), torch.randn(5), torch.randn(5), torch.rand(5)]
    permutation = torch.tensor([3, 0, 4, 2, 1])
    actual = impl.adamw_step(*[x[permutation] for x in tensors], 4)
    expected = impl.adamw_step(*tensors, 4)
    for x, y in zip(actual, expected):
        close(x, y[permutation])


@pytest.mark.t3
def test_anisotropic_quadratic(impl, artifact):
    """The submitted update optimizes a deterministic two-coordinate quadratic over 200 steps."""
    result = run(impl.adamw_step)
    artifact("quadratic_optimization", result)
    assert result["final_loss"] < .01
    assert result["final_loss"] < result["initial_loss"] * 1e-3
