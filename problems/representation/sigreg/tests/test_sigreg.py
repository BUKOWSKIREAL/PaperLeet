import cmath
import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.sigreg.experiment import run


def _scalar_full_grid_oracle(projected, nodes):
    rows = projected.tolist(); n, m = projected.shape
    full_t = [-v for v in nodes.tolist()[1:][::-1]] + nodes.tolist()
    out = []
    for j in range(m):
        values = []
        for t in full_t:
            phi = math.exp(-t*t/2)
            ecf = sum(cmath.exp(1j * t * row[j]) for row in rows) / n
            values.append(abs(ecf - phi) ** 2 * phi)
        out.append(n * sum((full_t[k+1] - full_t[k]) * (values[k+1] + values[k]) / 2 for k in range(len(full_t)-1)))
    return projected.new_tensor(out)


@pytest.mark.t0
def test_shapes_dtype_and_immutable_inputs(impl):
    """Epps-Pulley returns M statistics; SIGReg returns one scalar without changing supplied geometry."""
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([[.1, .4], [1., -.3], [-1., .7]], dtype=dtype)
        a = torch.eye(2, dtype=dtype); t = torch.linspace(0, 3, 17, dtype=dtype)
        saved = [v.clone() for v in (x, a, t)]
        stats = impl.epps_pulley(x @ a, t); loss = impl.sigreg_loss(x, a, t)
        assert stats.shape == (2,) and loss.shape == () and loss.dtype == dtype and stats.dtype == dtype
        for got, before in zip((x, a, t), saved): close(got, before)


@pytest.mark.t0
def test_supplied_validation(impl):
    """Fixed directions must be unit length and the positive grid starts at zero."""
    x = torch.ones(3, 2); a = torch.eye(2); t = torch.linspace(0, 3, 17)
    for wrong_t in (t[1:], t.flip(0), torch.tensor([0.]), torch.tensor([0., 1., 1.]), t.double()):
        with pytest.raises(ValueError): impl.epps_pulley(x, wrong_t)
    for wrong_a in (2*a, a[:1], torch.ones(2, 0), a * float('nan')):
        with pytest.raises(ValueError): impl.sigreg_loss(x, wrong_a, t)
    for wrong_x in (torch.ones(0, 2), torch.ones(3), x.long()):
        with pytest.raises(ValueError): impl.epps_pulley(wrong_x, t)


@pytest.mark.t1
def test_complex_full_grid_scalar_oracle(impl):
    """Asymmetric samples and nonuniform nodes expose missing sine, window, symmetry or N factors."""
    x = torch.tensor([[.2, -.5], [1.5, .3], [-.1, 2.1]], dtype=torch.float64)
    t = x.new_tensor([0., .2, .9, 1.7, 3.])
    expected = _scalar_full_grid_oracle(x, t)
    close(impl.epps_pulley(x, t), expected, atol=1e-12, rtol=1e-12)
    a = x.new_tensor([[1., 0., .6], [0., 1., .8]])
    close(impl.sigreg_loss(x, a, t), _scalar_full_grid_oracle(x @ a, t).mean(), atol=1e-12, rtol=1e-12)


@pytest.mark.t1
def test_singleton_and_zero_grid_origin(impl):
    """N=1 remains well-defined; at t=0 empirical and target characteristic functions agree exactly."""
    x = torch.tensor([[1.2]], dtype=torch.float64); t = x.new_tensor([0., .7])
    phi = math.exp(-.7**2/2)
    expected = .7 * ((math.cos(.84) - phi)**2 + math.sin(.84)**2) * phi
    close(impl.epps_pulley(x, t), x.new_tensor([expected]), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_autograd_features_and_fixed_geometry(impl):
    """Features retain gradients; externally supplied sampled directions and quadrature nodes are constants."""
    x = torch.tensor([[.2, -.4], [1., .1], [-.3, .7]], dtype=torch.float64, requires_grad=True)
    a = torch.eye(2, dtype=x.dtype, requires_grad=True); t = torch.linspace(0, 3, 9, dtype=x.dtype, requires_grad=True)
    grads = torch.autograd.grad(impl.sigreg_loss(x, a, t), (x, a, t), allow_unused=True)
    assert torch.isfinite(grads[0]).all() and grads[0].abs().sum() > 0
    assert all(v is None or not v.any() for v in grads[1:])
    assert torch.autograd.gradcheck(lambda z: impl.sigreg_loss(z, a.detach(), t.detach()), (x,))


@pytest.mark.t2
def test_exact_collapse_is_stationary_and_nearby_spread_improves(impl):
    """All-zero collapse has positive loss but zero gradient; a nonconstant perturbation has an escape direction."""
    zero = torch.zeros(4, 1, dtype=torch.float64, requires_grad=True)
    a = torch.ones(1, 1, dtype=zero.dtype); t = torch.linspace(0, 3, 17, dtype=zero.dtype)
    loss = impl.sigreg_loss(zero, a, t)
    gradient, = torch.autograd.grad(loss, zero)
    assert loss > 0; close(gradient, torch.zeros_like(zero))
    small = torch.tensor([[-.03], [-.01], [.01], [.03]], dtype=zero.dtype, requires_grad=True)
    lower = impl.sigreg_loss(small, a, t); grad, = torch.autograd.grad(lower, small)
    assert lower < loss and (grad * small).sum() < 0


@pytest.mark.t2
def test_slice_signs_joint_rotation_and_sample_count(impl):
    """Rotation with directions and slice signs preserve the statistic; duplicated observations double N scaling."""
    x = torch.randn(6, 2, dtype=torch.float64); a = x.new_tensor([[1., 0., .6], [0., 1., .8]])
    t = torch.linspace(0, 3, 17, dtype=x.dtype)
    value = impl.sigreg_loss(x, a, t); q = x.new_tensor([[.8, -.6], [.6, .8]])
    close(impl.sigreg_loss(x @ q, q.T @ a, t), value, atol=1e-12, rtol=1e-12)
    close(impl.sigreg_loss(x.flip(0), -a.flip(1), t), value, atol=1e-12, rtol=1e-12)
    close(impl.sigreg_loss(x.repeat(2, 1), a, t), 2 * value, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_gaussian_target_retains_location_and_scale(impl):
    """SIGReg is not a normality test that first fits or standardizes the observed distribution."""
    q = torch.erfinv(2 * (torch.arange(128, dtype=torch.float64) + .5) / 128 - 1) * 2 ** .5
    x = q[:, None]; t = torch.linspace(0, 3, 17, dtype=x.dtype); a = torch.ones(1, 1, dtype=x.dtype)
    good = impl.sigreg_loss(x, a, t)
    assert impl.sigreg_loss(x + 2, a, t) > 100 * good
    assert impl.sigreg_loss(x * .1, a, t) > 100 * good


@pytest.mark.t3
def test_fixed_slice_distribution_experiment(impl, artifact):
    """Submitted statistics optimize a distorted feature cloud and detect shape differences beyond moments."""
    result = run(impl.epps_pulley, impl.sigreg_loss); artifact('sigreg_distribution', result)
    m = result['metrics']
    assert m['final_loss'] < .01 * m['initial_loss'] and m['oracle_absolute_error'] < 1e-10
    assert m['mean_norm'] < .25 and m['minimum_std'] > .9
    assert m['binary_statistic'] > 100 * m['normal_quantile_statistic']
