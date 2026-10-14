import math
from statistics import NormalDist
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.visreg.experiment import run


@pytest.mark.t0
def test_output_contract(impl):
    """Both functions accept noncontiguous features and preserve dtype, values and scalar output shape."""
    for dtype in (torch.float32, torch.float64):
        z = torch.arange(24, dtype=dtype).reshape(3, 8).T
        directions = torch.eye(3, dtype=dtype)
        views = torch.stack((z, z + .2, z - .1)); before = views.clone()
        outputs = (*impl.visreg_terms(z, directions), impl.view_invariance(views))
        for value in outputs: assert value.shape == () and value.dtype == dtype and value.device == z.device and torch.isfinite(value)
        close(views, before)


@pytest.mark.t0
def test_validation(impl):
    """Supplied guards reject invalid sample counts, projections, epsilon and global-view count."""
    z = torch.randn(4, 2); w = torch.eye(2)
    for bad in (z[:1], z.long(), z * float('nan')):
        with pytest.raises(ValueError): impl.visreg_terms(bad, w)
    for bad in (torch.zeros_like(w), torch.eye(3), w.double()):
        with pytest.raises(ValueError): impl.visreg_terms(z, bad)
    with pytest.raises(ValueError): impl.visreg_terms(z, w, 0.)
    for n in (0, 4, 1.5):
        with pytest.raises(ValueError): impl.view_invariance(torch.stack((z, z, z)), n)


@pytest.mark.t1
def test_scalar_formula_oracle(impl):
    """Python scalar statistics and NormalDist quantify all three reductions independently."""
    rows = [[-2., .3], [.2, 2.1], [.6, -.4], [3., .8], [.9, -.7]]
    wrows = [[2., -1., .3], [1., 2., -2.]]; n, d, k, eps = 5, 2, 3, .02
    means = [sum(r[j] for r in rows) / n for j in range(d)]
    std = [math.sqrt(sum((r[j] - means[j])**2 for r in rows) / n) for j in range(d)]
    scale = sum((s - 1)**2 for s in std) / d
    center = sum(m*m for m in means) / d
    shape = 0.
    for j in range(k):
        norm = math.sqrt(sum(wrows[i][j]**2 for i in range(d)))
        projections = sorted(sum((r[i] - means[i]) / (std[i] + eps) * wrows[i][j] / norm for i in range(d)) for r in rows)
        shape += sum((p - NormalDist().inv_cdf((i + 1) / (n + 1)))**2 for i, p in enumerate(projections)) / (n*k)
    actual = impl.visreg_terms(torch.tensor(rows, dtype=torch.float64), torch.tensor(wrows, dtype=torch.float64), eps)
    for got, expected in zip(actual, (scale, shape, center)): close(got, got.new_tensor(expected), atol=1e-12, rtol=1e-12)


@pytest.mark.t1
def test_global_view_mean_not_all_views(impl):
    """The global anchor uses only the first G views, while all views contribute error."""
    views = torch.tensor([[[0., 2.]], [[2., 0.]], [[5., -1.]]], dtype=torch.float64)
    close(impl.view_invariance(views, 2), views.new_tensor(24. / 6.))
    close(impl.view_invariance(views[:1], 1), views.new_zeros(()))


@pytest.mark.t2
def test_shape_gradient_freezes_std_and_directions(impl):
    """Finite differences with a frozen denominator catch omission of stop-gradient; direction tensors are supplied constants."""
    z = torch.tensor([[-2., .3], [.2, 2.1], [.6, -.4], [3., .8], [.9, -.7]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([[2., -1.], [1., 2.]], dtype=z.dtype, requires_grad=True); eps = .01
    shape = impl.visreg_terms(z, w, eps)[1]
    grad, direction_grad = torch.autograd.grad(shape, (z, w), allow_unused=True)
    fixed_std = z.detach().std(0, correction=0) + eps
    unit = w.detach() / w.detach().norm(dim=0)
    q = z.new_tensor([NormalDist().inv_cdf(i / 6) for i in range(1, 6)])
    def oracle(x):
        projected = ((x - x.mean(0)) / fixed_std @ unit).sort(0).values
        return (projected - q[:, None]).square().mean()
    numerical = torch.zeros_like(z)
    for i in range(z.numel()):
        plus = z.detach().clone(); minus = z.detach().clone()
        plus.view(-1)[i] += 1e-6; minus.view(-1)[i] -= 1e-6
        numerical.view(-1)[i] = (oracle(plus) - oracle(minus)) / 2e-6
    close(grad, numerical, atol=1e-8, rtol=1e-6)
    assert direction_grad is None or not direction_grad.any()


@pytest.mark.t2
def test_global_anchor_receives_gradient(impl):
    """Global views receive the derivative through the shared anchor; local views do not contribute to that anchor."""
    views = torch.tensor([[[0., 2.]], [[2., 0.]], [[5., -1.]]], dtype=torch.float64, requires_grad=True)
    grad, = torch.autograd.grad(impl.view_invariance(views, 2), views)
    residual = views.detach() - views.detach()[:2].mean(0)
    expected = 2 * residual / views.numel()
    expected[:2] -= 2 * residual.sum(0) / (2 * views.numel())
    close(grad, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_permutation_translation_and_projection_scaling(impl):
    """Sorting removes sample order; center is the only translation-sensitive term; projection magnitude is irrelevant."""
    z = torch.tensor([[-2., .3], [.2, 2.1], [.6, -.4], [3., .8], [.9, -.7]], dtype=torch.float64)
    w = torch.tensor([[2., -1.], [1., 2.]], dtype=z.dtype)
    base = impl.visreg_terms(z, w)
    permuted = impl.visreg_terms(z[[4, 0, 2, 1, 3]], w * z.new_tensor([3., .2]))
    for got, expected in zip(permuted, base): close(got, expected, atol=1e-12, rtol=1e-12)
    shifted = impl.visreg_terms(z + 7, w)
    close(shifted[0], base[0]); close(shifted[1], base[1])


@pytest.mark.t2
def test_collapsed_features_have_finite_forward(impl):
    """Epsilon from Eq.2 keeps the fully collapsed forward finite; no claim of a unique sorting gradient at ties."""
    z = torch.full((4, 2), 3., dtype=torch.float64)
    scale, shape, center = impl.visreg_terms(z, torch.eye(2, dtype=z.dtype))
    close(scale, z.new_tensor(1.)); close(center, z.new_tensor(9.))
    q2 = sum(NormalDist().inv_cdf(i / 5)**2 for i in range(1, 5)) / 4
    close(shape, z.new_tensor(q2), atol=1e-12, rtol=1e-12)


@pytest.mark.t3
def test_scale_shape_experiment(impl, artifact):
    """Submitted functions optimize aligned feature views while independent statistics measure spread and centering."""
    result = run(impl.visreg_terms, impl.view_invariance); artifact('visreg_scale_shape', result)
    m = result['metrics']
    assert m['final_objective'] < .3 * m['initial_objective']
    assert m['final_mean_std'] > 2 * m['initial_mean_std']
    assert m['final_center_squared'] < .15 * m['initial_center_squared']
