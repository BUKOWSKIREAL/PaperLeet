import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.attention.experiment import run


@pytest.mark.t0
def test_shapes_and_inputs(impl):
    """Attention supports distinct query/key/value dimensions without mutating inputs."""
    q, k, v = torch.randn(2, 3, 4, 5), torch.randn(2, 3, 6, 5), torch.randn(2, 3, 6, 7)
    before = [x.clone() for x in (q, k, v)]
    output, weights = impl.scaled_dot_product_attention(q, k, v)
    assert output.shape == (2, 3, 4, 7) and weights.shape == (2, 3, 4, 6)
    assert output.dtype == q.dtype
    for x, original in zip((q, k, v), before):
        close(x, original)


@pytest.mark.t1
def test_independent_two_key_formula(impl):
    """Two-key probability equals sigmoid of the scaled score difference."""
    q = torch.tensor([[[[2., 0., 0., 0.]]]], dtype=torch.float64)
    k = torch.tensor([[[[1., 0., 0., 0.], [0., 0., 0., 0.]]]], dtype=q.dtype)
    v = torch.tensor([[[[3., -2.], [7., 4.]]]], dtype=q.dtype)
    output, weights = impl.scaled_dot_product_attention(q, k, v)
    p = 1 / (1 + math.exp(-1))
    close(weights, torch.tensor([[[[p, 1 - p]]]], dtype=q.dtype))
    close(output, p * v[:, :, :1] + (1 - p) * v[:, :, 1:])


@pytest.mark.t2
def test_mask_broadcast_and_zero_rows(impl):
    """Allowed-key mask broadcasts across heads; empty rows have finite zero gradients."""
    q = torch.randn(2, 3, 2, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 3, 3, 4, dtype=q.dtype, requires_grad=True)
    v = torch.randn(2, 3, 3, 5, dtype=q.dtype, requires_grad=True)
    mask = torch.tensor([[True, False, False], [False, False, False]])
    output, weights = impl.scaled_dot_product_attention(q, k, v, mask)
    close(output[:, :, 0], v[:, :, 0])
    close(output[:, :, 1], torch.zeros_like(output[:, :, 1]))
    close(weights.sum(-1), torch.tensor([1., 0.], dtype=q.dtype).expand(2, 3, 2))
    output.sum().backward()
    for x in (q, k, v):
        assert torch.isfinite(x.grad).all()
    close(k.grad, torch.zeros_like(k))
    close(q.grad, torch.zeros_like(q))


@pytest.mark.t2
def test_key_permutation_and_shift(impl):
    """Jointly permuting keys and values or adding a common key shift leaves outputs unchanged."""
    q, k, v = [torch.randn(2, 2, 5, 3, dtype=torch.float64) for _ in range(3)]
    perm = torch.tensor([3, 0, 4, 1, 2])
    expected, weights = impl.scaled_dot_product_attention(q, k, v)
    got, _ = impl.scaled_dot_product_attention(q, k[:, :, perm], v[:, :, perm])
    close(got, expected)
    got, _ = impl.scaled_dot_product_attention(q, k + 100, v)
    close(got, expected, atol=1e-10, rtol=1e-10)
    close(weights.sum(-1), torch.ones_like(weights.sum(-1)))


@pytest.mark.t2
def test_gradients_against_independent_dense_expression(impl):
    """Masked attention has the same q/k/v derivatives as an independently constructed softmax."""
    tensors = [torch.randn(1, 2, 4, 3, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    clones = [x.detach().clone().requires_grad_() for x in tensors]
    mask = impl.causal_mask(4, 4)
    got, _ = impl.scaled_dot_product_attention(*tensors, mask)
    q, k, v = clones
    logits = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(3)
    expected = logits.masked_fill(~mask, -torch.inf).softmax(-1) @ v
    close(got, expected)
    weight = torch.randn_like(got)
    for a, b in zip(torch.autograd.grad((got * weight).sum(), tensors),
                    torch.autograd.grad((expected * weight).sum(), clones)):
        close(a, b, atol=1e-10, rtol=1e-10)


@pytest.mark.t2
def test_causal_future_independence(impl):
    """Changing future keys and values cannot influence a causal prefix."""
    q, k, v = [torch.randn(1, 2, 6, 4) for _ in range(3)]
    mask = impl.causal_mask(6, 6)
    output, _ = impl.scaled_dot_product_attention(q, k, v, mask)
    k[:, :, 3:] += 100
    v[:, :, 3:] -= 100
    changed, _ = impl.scaled_dot_product_attention(q, k, v, mask)
    close(changed[:, :, :3], output[:, :, :3])


@pytest.mark.t3
def test_prefix_averaging_experiment(impl, artifact):
    """Uniform causal scores compute exact prefix means and exclude changed future values."""
    result = run(impl.scaled_dot_product_attention)
    artifact("prefix_averaging", result)
    assert result["max_error"] < 1e-12
    assert result["prefix_leakage"] == 0
