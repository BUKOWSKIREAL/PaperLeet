import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion.daps.experiment import run


@pytest.mark.t0
def test_shapes_dtype_and_no_mutation(impl):
    """A rectangular operator and independent batch measurements retain the input contract."""
    for dtype in (torch.float32, torch.float64):
        x, anchor = torch.randn(3, 4, dtype=dtype), torch.randn(3, 4, dtype=dtype)
        a, y = torch.randn(2, 4, dtype=dtype), torch.randn(3, 2, dtype=dtype)
        saved = [v.clone() for v in (x, anchor, a, y)]
        score = impl.posterior_score(x, anchor, a, y, .7, .2)
        for actual in (score, impl.langevin_step(x, score, .01, torch.ones_like(x)), impl.renoise(x, .2, torch.zeros_like(x))):
            assert actual.shape == x.shape and actual.dtype == dtype and actual.device == x.device
        for value, before in zip((x, anchor, a, y), saved): close(value, before)


@pytest.mark.t0
def test_validation_and_zero_step(impl):
    """Zero integration/re-noise steps are identities; variances must remain positive."""
    x = torch.ones(2, 3)
    close(impl.langevin_step(x, x, 0., x), x)
    close(impl.renoise(x, 0., x), x)
    for invalid in (-1., float('inf'), float('nan')):
        with pytest.raises(ValueError): impl.langevin_step(x, x, invalid, x)
        with pytest.raises(ValueError): impl.renoise(x, invalid, x)
    for r, beta in ((0., 1.), (1., 0.), (-1., 1.)):
        with pytest.raises(ValueError): impl.posterior_score(x, x, torch.ones(1, 3), torch.ones(2, 1), r, beta)
    with pytest.raises(ValueError): impl.posterior_score(x, x, torch.ones(1, 2), torch.ones(2, 1), 1., 1.)


@pytest.mark.t1
def test_score_from_scalar_energy_finite_differences(impl):
    """Independent scalar energy derivatives verify the likelihood transpose and variance scaling."""
    x = torch.tensor([[.2, -.4, 1.], [-.3, 2., .5]], dtype=torch.float64)
    anchor = x.new_tensor([[1., 0., -1.], [.5, -.2, .1]])
    a = x.new_tensor([[2., -1., .3], [.4, .7, -2.]])
    y = x.new_tensor([[.4, -.3], [1.2, .7]])
    r, beta = .7, .3
    def energy(row, center, measure):
        prior = sum((v - c) ** 2 for v, c in zip(row, center)) / (2 * r * r)
        data = sum((sum(v * c for v, c in zip(row, ar)) - yi) ** 2
                   for ar, yi in zip(a.tolist(), measure)) / (2 * beta * beta)
        return prior + data
    expected = torch.empty_like(x)
    for b, row in enumerate(x.tolist()):
        for d in range(3):
            plus, minus = row.copy(), row.copy()
            plus[d] += 1e-5; minus[d] -= 1e-5
            expected[b, d] = -(energy(plus, anchor[b].tolist(), y[b].tolist()) - energy(minus, anchor[b].tolist(), y[b].tolist())) / 2e-5
    close(impl.posterior_score(x, anchor, a, y, r, beta), expected, atol=1e-8, rtol=1e-8)


@pytest.mark.t1
def test_explicit_noise_step_formula(impl):
    """Supplied deterministic noise fixes both Langevin diffusion and fresh-noise transition exactly."""
    x = torch.tensor([[1., -2.]], dtype=torch.float64)
    score, noise = x.new_tensor([[3., 4.]]), x.new_tensor([[-2., .5]])
    close(impl.langevin_step(x, score, .125, noise), x.new_tensor([[.375, -1.25]]))
    close(impl.renoise(x, .4, noise), x.new_tensor([[.2, -1.8]]))


@pytest.mark.t2
def test_gaussian_posterior_stationary_point(impl):
    """Completing the square gives a score-zero mean and precision-matrix score differences."""
    dtype = torch.float64
    a = torch.tensor([[1., 2.], [-.4, .3], [.2, -.7]], dtype=dtype)
    y, anchor = torch.tensor([[.2, 1., -.3]], dtype=dtype), torch.tensor([[.5, -1.]], dtype=dtype)
    r, beta = .8, .6
    h = torch.eye(2, dtype=dtype) / r ** 2 + a.T @ a / beta ** 2
    mean = torch.linalg.solve(h, anchor[0] / r ** 2 + a.T @ y[0] / beta ** 2)[None]
    close(impl.posterior_score(mean, anchor, a, y, r, beta), torch.zeros_like(mean), atol=1e-12, rtol=1e-12)
    delta = torch.tensor([[.7, -.3]], dtype=dtype)
    close(impl.posterior_score(mean + delta, anchor, a, y, r, beta), -delta @ h, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_batch_independence_and_unobserved_direction(impl):
    """Observation-null coordinates retain the prior force and different samples never mix."""
    x = torch.tensor([[1., 2.], [-1., 3.]], dtype=torch.float64)
    anchor = torch.zeros_like(x); a = x.new_tensor([[1., 0.]]); y = x.new_tensor([[5.], [-2.]])
    score = impl.posterior_score(x, anchor, a, y, 2., .5)
    close(score[:, 1], -x[:, 1] / 4)
    for b in range(2): close(score[b:b + 1], impl.posterior_score(x[b:b + 1], anchor[b:b + 1], a, y[b:b + 1], 2., .5))


@pytest.mark.t2
def test_noise_covariance_and_independent_renoising(impl):
    """The Langevin increment variance is 2 eta; re-noising uses sigma_next, not a noise-level difference."""
    generator = torch.Generator().manual_seed(71)
    z = torch.randn(12000, 2, generator=generator, dtype=torch.float64)
    fresh = torch.randn(z.shape, generator=generator, dtype=z.dtype)
    zero = torch.zeros_like(z)
    increment = impl.langevin_step(z, zero, .07, fresh) - z
    close(increment.var(0, unbiased=False), z.new_full((2,), .14), atol=.004, rtol=0)
    renewed = impl.renoise(z, .6, fresh) - z
    close(renewed.var(0, unbiased=False), z.new_full((2,), .36), atol=.01, rtol=0)
    assert (renewed * z).mean(0).abs().max() < .02


@pytest.mark.t2
def test_composed_transition_has_two_noise_draws(impl):
    """A complete tiny transition uses distinct MCMC and annealing noises and a fixed ODE anchor."""
    x = torch.tensor([[.2, -.1]], dtype=torch.float64)
    anchor = x.clone(); a = x.new_tensor([[1., 0.]]); y = x.new_tensor([[1.]])
    eps1, eps2 = x.new_tensor([[.5, -1.]]), x.new_tensor([[-.3, .7]])
    score = impl.posterior_score(x, anchor, a, y, .5, .4)
    clean = impl.langevin_step(x, score, .02, eps1)
    actual = impl.renoise(clean, .1, eps2)
    expected = x + .02 * x.new_tensor([[5., 0.]]) + .2 * eps1 + .1 * eps2
    close(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.t3
def test_analytic_local_posterior_experiment(impl, artifact):
    """Submitted functions produce Gaussian conditional samples and fresh-noise artifacts."""
    result = run(impl.posterior_score, impl.langevin_step, impl.renoise)
    artifact('daps_local_gaussian_posterior', result)
    m = result['metrics']
    assert m['mean_error'] < .05 and m['ula_covariance_error'] < .06
    assert m['final_measurement_mse'] < .2 * m['initial_measurement_mse']
    assert m['renoise_variance_error'] < .006 and m['renoise_formula_error'] < 1e-12
