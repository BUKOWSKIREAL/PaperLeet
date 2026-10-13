import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.optimization.muon_split.experiment import run


def gram_oracle(block):
    """Independent full-rank polar oracle via an eigen-decomposed Gram matrix."""
    if block.shape[0] <= block.shape[1]:
        values, vectors = torch.linalg.eigh(block @ block.T)
        return (vectors @ torch.diag(values.rsqrt()) @ vectors.T) @ block
    values, vectors = torch.linalg.eigh(block.T @ block)
    return block @ (vectors @ torch.diag(values.rsqrt()) @ vectors.T)


@pytest.mark.t0
def test_contract_and_no_mutation(impl):
    """Preserve shape/dtype and reject unsupported layouts without modifying input."""
    for dtype in (torch.float32, torch.float64):
        x = torch.randn(6, 5, dtype=dtype)
        before = x.clone()
        out = impl.headwise_orthogonalize(x, 3)
        assert out.shape == x.shape and out.dtype == dtype and out.device == x.device
        close(x, before)
    for x, heads in [(torch.ones(2, 3, 4), 2), (torch.ones(3, 4), 2), (torch.ones(2, 4), 0),
                     (torch.empty(0, 4), 2), (torch.empty(4, 0), 2), (torch.ones(4, 4, dtype=torch.int64), 2)]:
        with pytest.raises(ValueError):
            impl.headwise_orthogonalize(x, heads)


@pytest.mark.t1
def test_rectangular_gram_oracle(impl):
    """Tall and wide per-head matrices match independent Gram inverse-square-root formulas."""
    for rows, columns, heads in [(6, 5, 3), (8, 2, 2), (4, 4, 2)]:
        x = torch.randn(rows, columns, dtype=torch.float64)
        expected = torch.cat([gram_oracle(block) for block in x.chunk(heads, dim=0)])
        close(impl.headwise_orthogonalize(x, heads), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.t2
def test_orthogonality_and_positive_head_scaling(impl):
    """Each full-row-rank head has orthonormal rows, invariant to its own positive scale."""
    x = torch.randn(3, 2, 4, dtype=torch.float64)
    out = impl.headwise_orthogonalize(x.reshape(6, 4), 3).reshape(3, 2, 4)
    close(out @ out.transpose(-1, -2), torch.eye(2, dtype=x.dtype).expand(3, -1, -1), atol=1e-10, rtol=1e-10)
    scaled = x * torch.tensor([1e-5, 4., 1e5], dtype=x.dtype)[:, None, None]
    close(impl.headwise_orthogonalize(scaled.reshape(6, 4), 3), out.reshape(6, 4), atol=1e-10, rtol=1e-10)


@pytest.mark.t2
def test_rank_deficiency_zero_and_relative_cutoff(impl):
    """Null singular directions and entire zero heads must not invent an update."""
    x = torch.tensor([[3., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], dtype=torch.float64)
    expected = torch.zeros_like(x)
    expected[0, 0] = 1.
    close(impl.headwise_orthogonalize(x, 2), expected, atol=1e-12, rtol=1e-12)
    # Distinguish a relative numerical-rank convention from fixed absolute epsilon.
    tiny = torch.diag(torch.tensor([1e-20, 2e-20], dtype=x.dtype))
    close(impl.headwise_orthogonalize(tiny, 1), torch.eye(2, dtype=x.dtype))
    near = torch.diag(torch.tensor([1., 1e-18], dtype=x.dtype))
    close(impl.headwise_orthogonalize(near, 1), torch.diag(torch.tensor([1., 0.], dtype=x.dtype)))


@pytest.mark.t2
def test_head_permutation_and_input_basis(impl):
    """Permuting heads and rotating the input basis commute with per-head projection."""
    x = torch.randn(3, 2, 4, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(4, 4, dtype=x.dtype))
    out = impl.headwise_orthogonalize(x.reshape(6, 4), 3).reshape(3, 2, 4)
    permutation = torch.tensor([2, 0, 1])
    close(impl.headwise_orthogonalize(x[permutation].reshape(6, 4), 3), out[permutation].reshape(6, 4), atol=1e-10, rtol=1e-10)
    close(impl.headwise_orthogonalize((x @ q).reshape(6, 4), 3), (out @ q).reshape(6, 4), atol=1e-10, rtol=1e-10)


@pytest.mark.t2
def test_full_rank_autograd(impl):
    """The functional core retains a correct local derivative away from repeated singular values and rank boundaries."""
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: impl.headwise_orthogonalize(z, 2), (x,), atol=2e-5, rtol=2e-4)


@pytest.mark.t3
def test_head_scale_experiment(impl, artifact):
    """Student outputs retain equal head norms across six orders of magnitude of input scale."""
    result = run(impl.headwise_orthogonalize)
    artifact("head_scale_sweep", result)
    assert result["metrics"]["max_headwise_error"] < 1e-10
    assert abs(result["metrics"]["first_head_norm_min"] - 2 ** .5) < 1e-10
    assert abs(result["metrics"]["first_head_norm_max"] - 2 ** .5) < 1e-10
    assert result["metrics"]["whole_matrix_first_norm_min"] < .002
