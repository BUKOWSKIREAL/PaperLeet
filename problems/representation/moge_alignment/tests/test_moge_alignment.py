import itertools
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.moge_alignment.experiment import run


def scalar_objective(s, x, y, w, trunc):
    return sum(min(abs(s * a - b) * c, trunc) if trunc is not None else abs(s * a - b) * c
               for a, b, c in zip(x, y, w))


def arrangement_oracle(p, q, trunc):
    # Independent 2D line-arrangement oracle, including all truncation kinks.
    lines = []
    for a, b in zip(p.tolist(), q.tolist()):
        w = 1 / b[2]
        for d in range(3):
            for offset in ([0.] if trunc is None else [-trunc / w, 0., trunc / w]):
                lines.append((a[d], float(d == 2), b[d] + offset))
    values = []
    for (a, b, c), (d, e, f) in itertools.combinations(lines, 2):
        det = a * e - b * d
        if abs(det) < 1e-10:
            continue
        s, t = (c * e - b * f) / det, (a * f - c * d) / det
        values.append(sum(min(abs(s * pi[k] + (t if k == 2 else 0.) - qi[k]) / qi[2], trunc)
                          if trunc is not None else abs(s * pi[k] + (t if k == 2 else 0.) - qi[k]) / qi[2]
                          for pi, qi in zip(p.tolist(), q.tolist()) for k in range(3)))
    return min(values)


def objective(p, q, s, t, trunc):
    error = (s * p + p.new_tensor([0., 0., t]) - q).abs() / q[:, 2:3]
    return (error if trunc is None else error.clamp_max(trunc)).sum()


@pytest.mark.t0
def test_interface_and_preservation(impl):
    """Scalar outputs retain dtype/device and no solver mutates input tensors."""
    for dtype in (torch.float32, torch.float64):
        p = torch.tensor([[1., 2., 3.], [-1., 3., 5.]], dtype=dtype)
        q = 2 * p; q[:, 2] += 1
        saved = p.clone(), q.clone()
        s, t = impl.roe_align(p, q, torch.ones(2, dtype=torch.bool), .5)
        for out in (s, t):
            assert out.shape == () and out.dtype == dtype and out.device == p.device
        close(s, p.new_tensor(2.)); close(t, p.new_tensor(1.))
        close(p, saved[0]); close(q, saved[1])


@pytest.mark.t0
def test_invalid_contract(impl):
    """Supplied validation rejects negative weights, invalid caps, and empty valid point sets."""
    x = torch.ones(2)
    with pytest.raises(ValueError): impl.optimal_scale_1d(x, x, -x)
    for cap in (0., -1., float('inf'), float('nan')):
        with pytest.raises(ValueError): impl.optimal_scale_1d(x, x, x, cap)
    p = torch.ones(2, 3)
    with pytest.raises(ValueError): impl.roe_align(p, p, torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError): impl.roe_align(p, -p, torch.ones(2, dtype=torch.bool))


@pytest.mark.t1
def test_scalar_breakpoint_oracle(impl):
    """Exact objective is checked at every zero and saturation breakpoint using Python scalars."""
    x = torch.tensor([-2., 0., 1., 4., .2], dtype=torch.float64)
    y = torch.tensor([1., 7., 2., 6., -3.], dtype=x.dtype)
    w = torch.tensor([.3, 2., 4., .1, 0.], dtype=x.dtype)
    for cap in (None, .4, 2.):
        candidates = [0.]
        for a, b, c in zip(x.tolist(), y.tolist(), w.tolist()):
            if a and c:
                candidates.extend([(b + off) / a for off in ([0.] if cap is None else [-cap / c, 0., cap / c])])
        expected = min(scalar_objective(s, x.tolist(), y.tolist(), w.tolist(), cap) for s in candidates)
        s, loss = impl.optimal_scale_1d(x, y, w, cap)
        close(loss, x.new_tensor(expected), atol=1e-11, rtol=1e-11)
        close(loss, x.new_tensor(scalar_objective(s.item(), x.tolist(), y.tolist(), w.tolist(), cap)))


@pytest.mark.t1
def test_full_roe_arrangement_oracle(impl):
    """Full 2D arrangement search checks the global scale/z-shift optimum independently of anchoring."""
    p = torch.tensor([[1., -2., 1.], [.5, 1., 2.], [-1., 3., 4.]], dtype=torch.float64)
    q = torch.tensor([[2., -3., 3.], [4., 1.7, 5.], [-1., -1., 7.]], dtype=p.dtype)
    for cap in (None, .15, .8):
        s, t = impl.roe_align(p, q, torch.ones(3, dtype=torch.bool), cap)
        close(objective(p, q, s, t, cap), p.new_tensor(arrangement_oracle(p, q, cap)), atol=1e-10, rtol=1e-10)


@pytest.mark.t2
def test_weight_then_truncation_and_not_l2(impl):
    """Truncation caps weighted residuals, not raw distances; untruncated fitting is L1."""
    x = torch.ones(3, dtype=torch.float64)
    y = torch.tensor([1., 2., 20.], dtype=x.dtype)
    w = torch.tensor([.1, 2., .2], dtype=x.dtype)
    s, loss = impl.optimal_scale_1d(x, y, w, .5)
    close(s, x.new_tensor(2.)); close(loss, x.new_tensor(.6))
    s, _ = impl.optimal_scale_1d(x, y, torch.ones_like(x))
    close(s, x.new_tensor(2.))


@pytest.mark.t2
def test_masked_invalid_points_and_scale_equivariance(impl):
    """Masked NaNs cannot leak into fitting; joint scene-unit scaling preserves scale."""
    p = torch.tensor([[1., 0., 1.], [-2., 1., 2.], [2., -2., 4.], [float('nan'), 0., 0.]], dtype=torch.float64)
    q = 1.7 * p + p.new_tensor([0., 0., 3.])
    mask = torch.tensor([True, True, True, False])
    s, t = impl.roe_align(p, q, mask, .4)
    close(s, p.new_tensor(1.7)); close(t, p.new_tensor(3.))
    s2, t2 = impl.roe_align(7 * p, 7 * q, mask, .4)
    close(s2, s); close(t2, 7 * t)


@pytest.mark.t2
def test_degenerate_vectors_and_ties(impl):
    """Zero coefficients and weights have defined finite fallbacks; ties choose the lowest scale candidate."""
    z = torch.zeros(3, dtype=torch.float64)
    s, loss = impl.optimal_scale_1d(z, z + 2, z + 1, .5)
    close(s, z.new_tensor(0.)); close(loss, z.new_tensor(1.5))
    s, loss = impl.optimal_scale_1d(z + 1, z + 2, z)
    close(s, z.new_tensor(0.)); close(loss, z.new_tensor(0.))
    s, _ = impl.optimal_scale_1d(torch.ones(2), torch.tensor([-1., 3.]), torch.ones(2))
    close(s, torch.tensor(-1.))


@pytest.mark.t2
def test_inverse_depth_weighting_and_z_only_shift(impl):
    """Depth-derived weights change the optimum; free xy translation is not permitted."""
    p = torch.tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], dtype=torch.float64)
    q = torch.tensor([[2., 2., 1.], [8., 8., 20.], [9., 9., 30.]], dtype=p.dtype)
    s, t = impl.roe_align(p, q, torch.ones(3, dtype=torch.bool))
    close(s, p.new_tensor(2.)); close(t, p.new_tensor(-1.))


@pytest.mark.t3
def test_corrupted_point_cloud_recovery(impl, artifact):
    """Submitted ROE and scalar solvers drive the plotted outlier-recovery experiment."""
    result = run(impl.optimal_scale_1d, impl.roe_align)
    artifact('roe_outlier_recovery', result)
    m = result['metrics']
    assert m['inlier_max_error'] < 1e-10
    assert abs(m['scale'] - 1.5) < 1e-10 and abs(m['z_shift'] - 2.) < 1e-10
    assert m['untruncated_inlier_error'] > .1
