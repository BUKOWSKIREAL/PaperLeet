import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.diffusion.fd_loss.experiment import run


def _oracle2(mu, cov, ref_mu, ref_cov):
    # tr(sqrt(A B)) has this independent scalar identity for 2x2 SPD matrices.
    cross = torch.sqrt(torch.trace(cov @ ref_cov) + 2 * torch.sqrt(torch.det(cov) * torch.det(ref_cov)))
    return (mu - ref_mu).square().sum() + cov.trace() + ref_cov.trace() - 2 * cross


@pytest.mark.t0
def test_scalar_dtype_and_no_mutation(impl):
    """Scalar outputs retain dtype/device and all input statistics remain unchanged."""
    for dtype in (torch.float32, torch.float64):
        mu = torch.tensor([1., -2.], dtype=dtype); cov = torch.tensor([[2., .3], [.3, 1.]], dtype=dtype)
        ref_mu, ref_cov = torch.zeros_like(mu), torch.eye(2, dtype=dtype)
        saved = [v.clone() for v in (mu, cov, ref_mu, ref_cov)]
        fd = impl.frechet_distance(mu, cov, ref_mu, ref_cov)
        loss = impl.normalized_fd_loss(torch.stack((fd, fd * 2)), mu.new_tensor([.5, 1.]))
        for value in (fd, loss): assert value.shape == () and value.dtype == dtype and value.device == mu.device
        for value, before in zip((mu, cov, ref_mu, ref_cov), saved): close(value, before)


@pytest.mark.t0
def test_supplied_validation(impl):
    """Malformed Gaussian statistics, negative weights and invalid normalizers are rejected."""
    mu = torch.zeros(2); cov = torch.eye(2)
    for bad in (torch.eye(3), -cov, torch.tensor([[1., 1.], [0., 1.]]), cov * float('nan')):
        with pytest.raises(ValueError): impl.frechet_distance(mu, bad, mu, cov)
    with pytest.raises(ValueError): impl.frechet_distance(mu, cov, mu.double(), cov.double())
    for eps in (0., -1., float('nan'), float('inf')):
        with pytest.raises(ValueError): impl.normalized_fd_loss(mu + 1., mu + 1., eps)
    with pytest.raises(ValueError): impl.normalized_fd_loss(mu + 1., mu - 1.)


@pytest.mark.t1
def test_noncommuting_covariance_scalar_oracle(impl):
    """A noncommuting 2x2 oracle exposes elementwise roots, dropped correlation and nonsymmetric eigvalsh."""
    mu = torch.tensor([1.2, -.3], dtype=torch.float64); ref_mu = mu.new_tensor([-.2, .7])
    cov = mu.new_tensor([[2., .8], [.8, .9]]); ref_cov = mu.new_tensor([[1., -.4], [-.4, 3.]])
    assert not torch.allclose(cov @ ref_cov, ref_cov @ cov)
    expected = _oracle2(mu, cov, ref_mu, ref_cov)
    close(impl.frechet_distance(mu, cov, ref_mu, ref_cov), expected, atol=1e-12, rtol=1e-12)
    distances = mu.new_tensor([0., .2, 2., 30.]); weights = mu.new_tensor([1., .3, 0., 2.])
    expected_loss = sum(w * d / (d + .01) for d, w in zip(distances.tolist(), weights.tolist()))
    close(impl.normalized_fd_loss(distances, weights), distances.new_tensor(expected_loss), atol=1e-12, rtol=1e-12)


@pytest.mark.t1
def test_diagonal_and_psd_forward_values(impl):
    """Diagonal Gaussian FD has a closed form, including zero eigenvalues in forward-only tests."""
    mu = torch.tensor([.5, 1., -1.], dtype=torch.float64); ref_mu = torch.zeros_like(mu)
    variance = mu.new_tensor([0., 1., 4.]); ref_variance = mu.new_tensor([0., 9., 1.])
    expected = sum(m * m + (math.sqrt(a) - math.sqrt(b)) ** 2 for m, a, b in zip(mu.tolist(), variance.tolist(), ref_variance.tolist()))
    close(impl.frechet_distance(mu, variance.diag(), ref_mu, ref_variance.diag()), mu.new_tensor(expected), atol=1e-12, rtol=1e-12)
    close(impl.frechet_distance(mu, variance.diag(), mu, variance.diag()), mu.new_zeros(()), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_gradient_matches_independent_spd_oracle_and_freezes_reference(impl):
    """Generated mean and covariance gradients match a non-eigen oracle; reference statistics stay frozen."""
    mu = torch.tensor([.2, -.8], dtype=torch.float64, requires_grad=True)
    lower = torch.tensor([[1.2, 0.], [.3, .7]], dtype=mu.dtype, requires_grad=True)
    ref_mu = torch.tensor([-.5, 1.], dtype=mu.dtype, requires_grad=True)
    ref_cov = torch.tensor([[2., -.2], [-.2, .8]], dtype=mu.dtype, requires_grad=True)
    cov = lower @ lower.T
    actual = impl.frechet_distance(mu, cov, ref_mu, ref_cov)
    gradients = torch.autograd.grad(actual, (mu, lower, ref_mu, ref_cov), allow_unused=True, retain_graph=True)
    expected = torch.autograd.grad(_oracle2(mu, cov, ref_mu.detach(), ref_cov.detach()), (mu, lower))
    for got, want in zip(gradients[:2], expected): close(got, want, atol=1e-11, rtol=1e-11)
    assert all(g is None or not g.any() for g in gradients[2:])


@pytest.mark.t2
def test_repeated_positive_eigenvalues_and_one_dimensional_gradient(impl):
    """Repeated positive spectra have finite gradients; scalar covariance derivatives match the analytic formula."""
    mu = torch.tensor([1., -2., .5], dtype=torch.float64, requires_grad=True)
    cov = (2 * torch.eye(3, dtype=mu.dtype)).requires_grad_()
    ref_mu = torch.zeros_like(mu); ref_cov = 3 * torch.eye(3, dtype=mu.dtype)
    dmu, dcov = torch.autograd.grad(impl.frechet_distance(mu, cov, ref_mu, ref_cov), (mu, cov))
    close(dmu, 2 * mu.detach(), atol=1e-12, rtol=1e-12)
    close(dcov, (1 - math.sqrt(1.5)) * torch.eye(3, dtype=mu.dtype), atol=1e-12, rtol=1e-12)
    variance = torch.tensor([[.8]], dtype=mu.dtype, requires_grad=True)
    grad, = torch.autograd.grad(impl.frechet_distance(mu[:1], variance, ref_mu[:1], mu.new_tensor([[2.]])), variance)
    close(grad, mu.new_tensor([[1 - math.sqrt(2. / .8)]]), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_stopped_normalization_gradient_and_weight_freeze(impl):
    """Normalization differentiates only its numerator; finite differencing the moving denominator would be the wrong oracle."""
    distances = torch.tensor([0., .2, 4., 100.], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([1., .5, 2., 0.], dtype=distances.dtype, requires_grad=True); eps = .01
    grads = torch.autograd.grad(impl.normalized_fd_loss(distances, weights, eps), (distances, weights), allow_unused=True)
    close(grads[0], weights.detach() / (distances.detach() + eps), atol=1e-12, rtol=1e-12)
    assert grads[1] is None or not grads[1].any()
    # A one-representation normalized FD must preserve the generated-statistic gradient direction.
    mu = torch.tensor([1., -.5], dtype=distances.dtype, requires_grad=True); cov = torch.eye(2, dtype=mu.dtype)
    fd = impl.frechet_distance(mu, cov, torch.zeros_like(mu), cov)
    gradient, = torch.autograd.grad(impl.normalized_fd_loss(fd[None], fd.new_ones(1), eps), mu)
    close(gradient, 2 * mu.detach() / (mu.detach().square().sum() + eps), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_symmetry_rotation_and_common_translation(impl):
    """FD is symmetric and unchanged by a shared translation or orthogonal change of feature coordinates."""
    mu = torch.randn(3, dtype=torch.float64); ref_mu = torch.randn_like(mu)
    a, b = torch.randn(3, 3, dtype=mu.dtype), torch.randn(3, 3, dtype=mu.dtype)
    cov, ref_cov = a @ a.T + torch.eye(3, dtype=mu.dtype), b @ b.T + torch.eye(3, dtype=mu.dtype)
    fd = impl.frechet_distance(mu, cov, ref_mu, ref_cov)
    close(fd, impl.frechet_distance(ref_mu, ref_cov, mu, cov), atol=1e-11, rtol=1e-11)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=mu.dtype)); shift = mu.new_tensor([3., -2., 1.])
    transformed = impl.frechet_distance(mu @ q + shift, q.T @ cov @ q, ref_mu @ q + shift, q.T @ ref_cov @ q)
    close(fd, transformed, atol=1e-11, rtol=1e-11)


@pytest.mark.t3
def test_affine_generator_posttraining(impl, artifact):
    """Submitted FD and stopped normalization post-train a small correlated Gaussian generator using EMA statistics."""
    result = run(impl.frechet_distance, impl.normalized_fd_loss)
    artifact('fd_affine_posttraining', result)
    metrics = result['metrics']
    assert metrics['final_fd'] < .02 * metrics['initial_fd']
    assert metrics['final_mean_error'] < .2 and metrics['final_covariance_error'] < .3
    assert metrics['minimum_covariance_eigenvalue'] > .1
