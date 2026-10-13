import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion.fd_population.experiment import run


@pytest.mark.t0
def test_shapes_dtype_and_no_mutation(impl):
    """Both functions preserve dtype/device and leave current features and history untouched."""
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([[1., 2.], [3., 4.], [-1., 0.]], dtype=dtype)
        old_mu, old_m2 = torch.ones(2, dtype=dtype), 3 * torch.eye(2, dtype=dtype)
        saved = [v.clone() for v in (x, old_mu, old_m2)]
        mu, m2 = impl.batch_moments(x)
        result = impl.ema_population_stats(x, old_mu, old_m2, .8)
        for value, shape in zip((mu, m2, *result), ((2,), (2, 2), (2,), (2, 2), (2, 2))):
            assert value.shape == shape and value.dtype == dtype and value.device == x.device
        for value, before in zip((x, old_mu, old_m2), saved): close(value, before)


@pytest.mark.t0
def test_contract_validation(impl):
    """Supplied validators reject invalid shapes, nonfinite data and invalid EMA decay."""
    x = torch.ones(2, 3); mu = torch.zeros(3); m2 = torch.eye(3)
    for wrong in (torch.ones(3), torch.ones(0, 3), torch.ones(2, 0), x.long(), x * float('nan')):
        with pytest.raises(ValueError): impl.batch_moments(wrong)
    for beta in (-.1, 1.1, float('nan'), float('inf')):
        with pytest.raises(ValueError): impl.ema_population_stats(x, mu, m2, beta)
    with pytest.raises(ValueError): impl.ema_population_stats(x, mu[:2], m2, .9)
    with pytest.raises(ValueError): impl.ema_population_stats(x, mu, m2.double(), .9)


@pytest.mark.t1
def test_raw_moments_scalar_oracle(impl):
    """Raw outer products use B, and remain meaningful for a single sample with nonzero mean."""
    for rows in ([[2., -1.]], [[1., 2.], [3., -4.], [-2., 1.]]):
        x = torch.tensor(rows, dtype=torch.float64); n = len(rows)
        mean = [sum(row[j] for row in rows) / n for j in range(2)]
        second = [[sum(row[j] * row[k] for row in rows) / n for k in range(2)] for j in range(2)]
        got_mu, got_second = impl.batch_moments(x)
        close(got_mu, x.new_tensor(mean), atol=1e-12, rtol=1e-12)
        close(got_second, x.new_tensor(second), atol=1e-12, rtol=1e-12)


@pytest.mark.t1
def test_weighted_population_oracle(impl):
    """An explicitly weighted joint population checks EMA covariance including between-batch mean shifts."""
    old = torch.tensor([[-3., 1.], [-1., 2.], [0., -1.]], dtype=torch.float64)
    x = torch.tensor([[3., -2.], [5., 4.]], dtype=torch.float64); beta = .7
    old_mu = old.mean(0); old_second = old.T @ old / len(old)
    rows = torch.cat((old, x)); weights = x.new_tensor([beta / 3] * 3 + [(1 - beta) / 2] * 2)
    expected_mean = (weights[:, None] * rows).sum(0)
    centered = rows - expected_mean
    expected_cov = sum(w * torch.outer(row, row) for w, row in zip(weights, centered))
    mu, second, cov = impl.ema_population_stats(x, old_mu, old_second, beta)
    close(mu, expected_mean, atol=1e-12, rtol=1e-12)
    close(cov, expected_cov, atol=1e-12, rtol=1e-12)
    close(second, expected_cov + torch.outer(expected_mean, expected_mean), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_current_gradients_and_frozen_history(impl):
    """Analytic current-batch gradients survive while both old moment tensors are stopped."""
    x = torch.tensor([[1., 3.], [-2., 4.], [5., -1.]], dtype=torch.float64, requires_grad=True)
    old_mu = torch.tensor([.8, -.2], dtype=x.dtype, requires_grad=True)
    old_second = torch.eye(2, dtype=x.dtype, requires_grad=True); beta = .75
    mu, second, cov = impl.ema_population_stats(x, old_mu, old_second, beta)
    loss = mu.sum() + cov.trace()
    gradients = torch.autograd.grad(loss, (x, old_mu, old_second), allow_unused=True)
    expected = (1 - beta) / len(x) * (1 + 2 * (x.detach() - mu.detach()))
    close(gradients[0], expected, atol=1e-12, rtol=1e-12)
    assert all(g is None or not g.any() for g in gradients[1:])


@pytest.mark.t2
def test_decay_endpoints_and_singleton(impl):
    """Beta zero uses current population moments; beta one freezes history, even with a singleton current batch."""
    x = torch.tensor([[3., -2.]], dtype=torch.float64, requires_grad=True)
    old_mu = x.new_tensor([1., 2.]); old_cov = x.new_tensor([[2., .3], [.3, 1.]])
    old_second = old_cov + torch.outer(old_mu, old_mu)
    mu, second, cov = impl.ema_population_stats(x, old_mu, old_second, 0.)
    close(mu, x[0]); close(cov, torch.zeros_like(cov), atol=1e-12, rtol=1e-12)
    mu, second, cov = impl.ema_population_stats(x, old_mu, old_second, 1.)
    close(mu, old_mu); close(second, old_second); close(cov, old_cov)
    gradient, = torch.autograd.grad(mu.sum() + second.sum(), x)
    close(gradient, torch.zeros_like(x))


@pytest.mark.t2
def test_affine_equivariance_and_replication(impl):
    """Translations preserve covariance; rotations transform it; duplicating a batch preserves population statistics."""
    x = torch.randn(5, 3, dtype=torch.float64); old = torch.randn(7, 3, dtype=x.dtype)
    old_mu, old_second = old.mean(0), old.T @ old / len(old)
    mu, second, cov = impl.ema_population_stats(x, old_mu, old_second, .6)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=x.dtype)); shift = x.new_tensor([3., -2., .5])
    transformed_old = old @ rotation + shift
    transformed = impl.ema_population_stats(x @ rotation + shift, transformed_old.mean(0), transformed_old.T @ transformed_old / len(old), .6)
    close(transformed[0], mu @ rotation + shift, atol=1e-12, rtol=1e-12)
    close(transformed[2], rotation.T @ cov @ rotation, atol=1e-12, rtol=1e-12)
    for got, expected in zip(impl.ema_population_stats(x.repeat(2, 1), old_mu, old_second, .6), (mu, second, cov)):
        close(got, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_sequential_ema_matches_expanded_weights(impl):
    """Three updates match explicit geometric sample weights without mutating or retaining old graphs."""
    initial = torch.tensor([[1., -1.], [-1., 1.]], dtype=torch.float64)
    batches = [initial + 2., initial * .5 - 3., initial - .2]; beta = .4
    mu, second = impl.batch_moments(initial)
    for batch in batches:
        mu, second, cov = impl.ema_population_stats(batch, mu.detach(), second.detach(), beta)
    weights = [beta ** 3, (1 - beta) * beta ** 2, (1 - beta) * beta, 1 - beta]
    all_rows = torch.cat([initial, *batches]); all_weights = initial.new_tensor([w / 2 for w in weights for _ in range(2)])
    mean = (all_rows * all_weights[:, None]).sum(0)
    centered = all_rows - mean
    covariance = sum(w * torch.outer(row, row) for w, row in zip(all_weights, centered))
    close(mu, mean, atol=1e-12, rtol=1e-12); close(cov, covariance, atol=1e-12, rtol=1e-12)


@pytest.mark.t3
def test_microbatch_population_experiment(impl, artifact):
    """Submitted moments process alternating microbatches, reducing population-estimation error and retaining current gradients."""
    result = run(impl.batch_moments, impl.ema_population_stats)
    artifact('fd_population_microbatches', result)
    metrics = result['metrics']
    assert metrics['ema_average_error'] < .02 * metrics['batch_average_error']
    assert metrics['gradient_max_error'] < 1e-12
    assert metrics['final_mean_error'] < .15 and metrics['final_covariance_error'] < .03
