import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.situ_glu.experiment import run


def scalar_expected(gate, up, b1, b2):
    # Scalar math independently checks the tensor implementation.
    return b1 * math.tanh(gate / b1) / (1 + math.exp(-gate)) * b2 * math.tanh(up / b2)


@pytest.mark.t0
def test_shape_dtype_and_input_preservation(impl):
    """Both functions preserve arbitrary leading dimensions, empty inputs, dtype, device and input values."""
    for dtype in (torch.float32, torch.float64):
        for shape in ((), (6,), (2, 3, 4), (2, 0, 3)):
            gate, up = torch.randn(shape, dtype=dtype), torch.randn(shape, dtype=dtype)
            before = gate.clone(), up.clone()
            for actual in (impl.softcap(gate, 4.0), impl.situ_glu(gate, up)):
                assert actual.shape == gate.shape and actual.dtype == dtype
                assert actual.device == gate.device
            close(gate, before[0])
            close(up, before[1])


@pytest.mark.t0
def test_finite_positive_caps(impl):
    """The supplied cap validation rejects nonpositive and nonfinite scalars in either branch."""
    x = torch.tensor([1.0])
    for invalid in (0.0, -1.0, float('inf'), float('-inf'), float('nan')):
        with pytest.raises(ValueError):
            impl.softcap(x, invalid)
        with pytest.raises(ValueError):
            impl.situ_glu(x, x, beta1=invalid)
        with pytest.raises(ValueError):
            impl.situ_glu(x, x, beta2=invalid)


@pytest.mark.t1
def test_independent_scalar_formula(impl):
    """Independent scalar calculations cover different signs and distinct nondefault branch caps."""
    gate = torch.tensor([[-7., -1., 0.], [1.5, 4., 13.]], dtype=torch.float64)
    up = torch.tensor([[3., -2., 9.], [-6., .4, 42.]], dtype=gate.dtype)
    for b1, b2 in ((4.0, 25.0), (1.7, 3.1), (7.0, .2)):
        expected = torch.tensor([scalar_expected(g, u, b1, b2)
                                 for g, u in zip(gate.flatten().tolist(), up.flatten().tolist())],
                                dtype=gate.dtype).reshape(gate.shape)
        close(impl.situ_glu(gate, up, b1, b2), expected, atol=1e-11, rtol=1e-11)
        capped = torch.tensor([b1 * math.tanh(g / b1) for g in gate.flatten().tolist()],
                              dtype=gate.dtype).reshape(gate.shape)
        close(impl.softcap(gate, b1), capped, atol=1e-11, rtol=1e-11)


@pytest.mark.t1
def test_sigmoid_uses_raw_gate_and_branches_are_distinct(impl):
    """The sigmoid sees the uncapped gate, and gate/up are not interchangeable."""
    g, u = torch.tensor([8.0], dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64)
    actual = impl.situ_glu(g, u, 1.0, 3.0)
    close(actual, torch.tensor([scalar_expected(8, 2, 1, 3)], dtype=g.dtype), atol=1e-12, rtol=1e-12)
    wrongly_capped_sigmoid = math.tanh(8) / (1 + math.exp(-math.tanh(8))) * 3 * math.tanh(2 / 3)
    assert abs(actual.item() - wrongly_capped_sigmoid) > .4
    assert not torch.allclose(actual, impl.situ_glu(u, g, 1.0, 3.0))


@pytest.mark.t2
def test_global_bound_and_negative_gate_tail(impl):
    """Independent large branch values respect beta1*beta2; the negative gate tail vanishes."""
    axis = torch.tensor([-1e6, -100., -30., -4., 0., 4., 30., 100., 1e6], dtype=torch.float64)
    gate, up = torch.meshgrid(axis, axis, indexing='ij')
    for b1, b2 in ((4.0, 25.0), (2.0, 3.0)):
        actual = impl.situ_glu(gate, up, b1, b2)
        assert torch.isfinite(actual).all()
        assert actual.abs().max() <= b1 * b2
        assert actual[0].abs().max() < 1e-20
        close(actual[-1, -1], torch.tensor(b1 * b2, dtype=axis.dtype))
        close(actual[-1, 0], torch.tensor(-b1 * b2, dtype=axis.dtype))


@pytest.mark.t2
def test_analytic_gradients_through_both_branches(impl):
    """Analytic derivatives verify both branch gradients and smooth behavior beyond hard-cap locations."""
    gate = torch.tensor([-8., -4., -.3, 0., .7, 4., 8.], dtype=torch.float64, requires_grad=True)
    up = torch.tensor([50., 25., -2., .4, 3., -25., -50.], dtype=gate.dtype, requires_grad=True)
    weights = torch.tensor([1., -.5, .3, 2., -.1, .8, 1.2], dtype=gate.dtype)
    actual = impl.situ_glu(gate, up)
    dg, du = torch.autograd.grad((actual * weights).sum(), (gate, up))
    expected_g, expected_u = [], []
    for g, u, w in zip(gate.detach().tolist(), up.detach().tolist(), weights.tolist()):
        s, tg, tu = 1 / (1 + math.exp(-g)), math.tanh(g / 4), math.tanh(u / 25)
        expected_g.append(w * ((1 - tg * tg) * s + 4 * tg * s * (1 - s)) * 25 * tu)
        expected_u.append(w * 4 * tg * s * (1 - tu * tu))
    close(dg, torch.tensor(expected_g, dtype=gate.dtype), atol=1e-11, rtol=1e-11)
    close(du, torch.tensor(expected_u, dtype=gate.dtype), atol=1e-11, rtol=1e-11)
    cap_gradient = torch.autograd.grad(impl.softcap(gate, 4.0).sum(), gate)[0]
    close(cap_gradient, torch.tensor([1 - math.tanh(g / 4) ** 2 for g in gate.detach().tolist()],
                                    dtype=gate.dtype), atol=1e-11, rtol=1e-11)


@pytest.mark.t2
def test_local_swiglu_approximation_and_large_cap_limit(impl):
    """Near-zero inputs and increasingly large caps recover the corresponding SwiGLU outputs."""
    gate = torch.tensor([-.03, -.01, .01, .03], dtype=torch.float64)
    up = torch.tensor([.07, -.04, .02, -.08], dtype=gate.dtype)
    swiglu = gate * torch.sigmoid(gate) * up
    close(impl.situ_glu(gate, up), swiglu, atol=1e-10, rtol=3e-5)
    gate, up = gate * 100, up * 100
    swiglu = gate * torch.sigmoid(gate) * up
    close(impl.situ_glu(gate, up, 1e7, 1e7), swiglu, atol=1e-10, rtol=1e-10)


@pytest.mark.t2
def test_noncontiguous_and_elementwise_independence(impl):
    """Strided inputs work; changing one coordinate leaves all other outputs unchanged."""
    gate = torch.randn(2, 3, 4, dtype=torch.float64).transpose(0, 1)
    up = torch.randn(2, 3, 4, dtype=torch.float64).transpose(0, 1)
    actual = impl.situ_glu(gate, up)
    close(actual, impl.situ_glu(gate.contiguous(), up.contiguous()))
    changed = gate.clone()
    changed[1, 0, 2] += 6
    other = impl.situ_glu(changed, up)
    mask = torch.ones_like(actual, dtype=torch.bool)
    mask[1, 0, 2] = False
    close(other[mask], actual[mask])


@pytest.mark.t3
def test_paper_scalar_curve_recreation(impl, artifact):
    """The submitted functions produce Figure-4-style curves, bounded output and the near-origin match."""
    result = run(impl.softcap, impl.situ_glu)
    artifact('figure_4_scalar_curves', result)
    metrics = result['metrics']
    assert metrics['maximum_formula_error'] < 1e-10
    assert metrics['maximum_absolute_output'] <= 100.0
    assert 99.9 < metrics['output_at_x_100'] < 100.0
    assert metrics['local_relative_swiglu_error'] < 6e-5
