import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.vicreg.experiment import run


def _scalar_oracle(x, y, eps):
    a, b = x.tolist(), y.tolist(); n, d = x.shape
    invariance = sum((a[i][j] - b[i][j]) ** 2 for i in range(n) for j in range(d)) / (n * d)
    variance, covariance = 0., 0.
    for rows in (a, b):
        means = [sum(row[j] for row in rows) / n for j in range(d)]
        cov = [[sum((row[j] - means[j]) * (row[k] - means[k]) for row in rows) / (n - 1) for k in range(d)] for j in range(d)]
        variance += sum(max(0., 1 - math.sqrt(cov[j][j] + eps)) for j in range(d)) / (2 * d)
        covariance += sum(cov[j][k] ** 2 for j in range(d) for k in range(d) if j != k) / d
    return x.new_tensor([invariance, variance, covariance])


@pytest.mark.t0
def test_shapes_dtypes_and_no_mutation(impl):
    """All three terms and total are scalar tensors; inputs retain dtype, device and contents."""
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([[1., 2.], [-1., 3.], [.2, -.5]], dtype=dtype); y = x + .4
        before = (x.clone(), y.clone())
        for value in (*impl.vicreg_terms(x, y), impl.vicreg_loss(x, y)):
            assert value.shape == () and value.dtype == dtype and value.device == x.device
        close(x, before[0]); close(y, before[1])


@pytest.mark.t0
def test_supplied_validation(impl):
    """A singleton batch cannot define the required sample covariance."""
    for x in (torch.ones(1, 2), torch.ones(0, 2), torch.ones(2, 0), torch.ones(3), torch.ones(3, 2).long(), torch.full((3, 2), float('nan'))):
        with pytest.raises(ValueError): impl.vicreg_terms(x, x)
    x = torch.ones(3, 2)
    with pytest.raises(ValueError): impl.vicreg_terms(x, x.double())
    with pytest.raises(ValueError): impl.vicreg_terms(x, x, eps=0)
    with pytest.raises(ValueError): impl.vicreg_loss(x, x, cov_coeff=-1)


@pytest.mark.t1
def test_scalar_loop_oracle_and_code_normalization(impl):
    """Scalar arithmetic distinguishes B-1, variance-view averaging and covariance-view summation."""
    x = torch.tensor([[.2, .1, -.5], [.6, -.1, .4], [-.3, .5, .7]], dtype=torch.float64)
    y = x.roll(1, 0) * 1.2 + .3
    expected = _scalar_oracle(x, y, .03)
    close(torch.stack(impl.vicreg_terms(x, y, .03)), expected, atol=1e-12, rtol=1e-12)
    close(impl.vicreg_loss(x, y, 2., 3., 7., .03), expected @ x.new_tensor([2., 3., 7.]), atol=1e-12, rtol=1e-12)


@pytest.mark.t1
def test_one_dimension_and_satisfied_variance_floor(impl):
    """One feature has no off-diagonal pairs; variance above the floor incurs no hinge penalty."""
    x = torch.tensor([[-2.], [2.]], dtype=torch.float64)
    invariance, variance, covariance = impl.vicreg_terms(x, x)
    close(torch.stack((invariance, variance, covariance)), torch.zeros(3, dtype=x.dtype))


@pytest.mark.t2
def test_exact_collapse_stationary_but_nonconstant_gradient(impl):
    """Positive collapse penalty does not imply a nonzero gradient at exact collapse."""
    x = torch.zeros(4, 2, dtype=torch.float64, requires_grad=True)
    y = torch.zeros_like(x, requires_grad=True)
    inv, var, cov = impl.vicreg_terms(x, y)
    close(inv, x.new_tensor(0.)); close(var, x.new_tensor(.99)); close(cov, x.new_tensor(0.))
    for gradient in torch.autograd.grad(impl.vicreg_loss(x, y), (x, y)): close(gradient, torch.zeros_like(gradient))
    small = torch.tensor([[-.02, .01], [.02, -.01], [-.01, -.02], [.01, .02]], dtype=x.dtype, requires_grad=True)
    grad, = torch.autograd.grad(impl.vicreg_terms(small, small.detach())[1], small)
    assert torch.isfinite(grad).all() and (grad * small).sum() < 0


@pytest.mark.t2
def test_autograd_both_views_and_analytic_variance_gradient(impl):
    """No target detach: sample variance gives an analytic gradient in each active hinge."""
    x = torch.tensor([[-.3, .2], [.1, -.4], [.2, .3]], dtype=torch.float64, requires_grad=True)
    y = (x.detach() * .8 + .1).requires_grad_()
    var = impl.vicreg_terms(x, y, .02)[1]
    gx, gy = torch.autograd.grad(var, (x, y))
    for rows, got in ((x, gx), (y, gy)):
        centered = rows.detach() - rows.detach().mean(0)
        std = torch.sqrt(torch.var(rows.detach(), dim=0, correction=1) + .02)
        expected = -centered / (2 * rows.shape[1] * (rows.shape[0] - 1) * std)
        close(got, expected, atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(lambda a, b: impl.vicreg_loss(a, b, 2., 3., 1., .02), (x, y))


@pytest.mark.t2
def test_shared_translation_view_swap_and_permutations(impl):
    """Featurewise translations preserve regularization, while view matching retains paired ordering."""
    x = torch.randn(7, 3, dtype=torch.float64); y = torch.randn_like(x)
    base = torch.stack(impl.vicreg_terms(x, y))
    shift = x.new_tensor([5., -2., .7])
    close(torch.stack(impl.vicreg_terms(x + shift, y + shift)), base, atol=1e-12, rtol=1e-12)
    close(torch.stack(impl.vicreg_terms(y, x)), base, atol=1e-12, rtol=1e-12)
    close(torch.stack(impl.vicreg_terms(x.flip(0)[:, [2, 0, 1]], y.flip(0)[:, [2, 0, 1]])), base, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_covariance_targets_redundancy_not_diagonal(impl):
    """Equal per-feature spread can hide duplicated features; covariance exposes that redundancy."""
    spread = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=torch.float64)
    redundant = spread[:, :1].repeat(1, 2)
    good = impl.vicreg_terms(spread, spread); bad = impl.vicreg_terms(redundant, redundant)
    close(good[1], bad[1]); close(good[2], spread.new_tensor(0.))
    close(bad[2], spread.new_tensor(32 / 9), atol=1e-12, rtol=1e-12)


@pytest.mark.t3
def test_noncollapsed_geometry_experiment(impl, artifact):
    """Submitted loss expands nonconstant embeddings and reduces cross-feature redundancy."""
    result = run(impl.vicreg_terms, impl.vicreg_loss); artifact('vicreg_geometry', result)
    m = result['metrics']
    assert m['final_loss'] < .01 * m['initial_loss']
    assert m['minimum_std'] > .95 and m['off_diagonal_abs'] < .1
    assert m['final_invariance'] < .001 and m['collapsed_variance'] > .98
