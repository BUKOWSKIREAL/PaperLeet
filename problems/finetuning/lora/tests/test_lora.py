import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.finetuning.lora.experiment import run


@pytest.mark.t0
def test_shapes_and_validation(impl):
    """Adapters use A[r,in], B[out,r]; arbitrary leading input dimensions are preserved."""
    a, b = torch.randn(2, 5), torch.randn(3, 2)
    assert impl.lora_delta(a, b).shape == (3, 5)
    assert impl.lora_linear(torch.randn(4, 7, 5), torch.randn(3, 5), a, b).shape == (4, 7, 3)
    with pytest.raises(ValueError):
        impl.lora_delta(torch.empty(0, 5), torch.empty(3, 0))


@pytest.mark.t1
def test_reference(impl, ref):
    """Non-square matrices, non-unit alpha and bias agree with the reference."""
    x, w = torch.randn(4, 5, dtype=torch.float64), torch.randn(3, 5, dtype=torch.float64)
    a, b, bias = torch.randn(2, 5, dtype=x.dtype), torch.randn(3, 2, dtype=x.dtype), torch.randn(3, dtype=x.dtype)
    close(impl.lora_delta(a, b, 7.), ref.lora_delta(a, b, 7.))
    close(impl.lora_linear(x, w, a, b, 7., bias), ref.lora_linear(x, w, a, b, 7., bias))


@pytest.mark.t2
def test_scaling_rank_and_merge(impl):
    """The update has rank at most r and merged/unmerged linear maps are equivalent."""
    x, w, a, b = torch.randn(7, 5), torch.randn(4, 5), torch.randn(2, 5), torch.randn(4, 2)
    delta = impl.lora_delta(a, b, 4.)
    close(delta, 2 * (b @ a))
    assert torch.linalg.matrix_rank(delta) <= 2
    close(impl.lora_linear(x, w, a, b, 4.), x @ (w + delta).T)
    close(impl.lora_delta(a.repeat(2, 1), b.repeat(1, 2), 4.), delta)


@pytest.mark.t2
def test_frozen_base_and_input_gradient(impl):
    """Base weights and bias stay frozen while input and both nonzero adapters receive exact gradients."""
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, 5, dtype=x.dtype, requires_grad=True)
    a = torch.randn(2, 5, dtype=x.dtype, requires_grad=True)
    b = torch.randn(4, 2, dtype=x.dtype, requires_grad=True)
    bias = torch.randn(4, dtype=x.dtype, requires_grad=True)
    impl.lora_linear(x, w, a, b, 6., bias).sum().backward()
    assert w.grad is None or torch.count_nonzero(w.grad) == 0
    assert bias.grad is None or torch.count_nonzero(bias.grad) == 0
    close(x.grad, (w.detach() + 3 * b.detach() @ a.detach()).sum(0).expand_as(x))
    close(a.grad, 3 * b.detach().sum(0)[:, None] * x.detach().sum(0)[None, :])
    close(b.grad, 3 * (x.detach() @ a.detach().T).sum(0).expand_as(b))


@pytest.mark.t2
def test_zero_b_initialization(impl):
    """Zero B preserves the base output, initially giving zero A gradient but nonzero B gradient."""
    x, w = torch.randn(4, 5), torch.randn(3, 5)
    a = torch.randn(2, 5, requires_grad=True)
    b = torch.zeros(3, 2, requires_grad=True)
    value = impl.lora_linear(x, w, a, b)
    close(value, x @ w.T)
    value.square().sum().backward()
    close(a.grad, torch.zeros_like(a))
    assert torch.count_nonzero(b.grad) > 0


@pytest.mark.t3
def test_rank_two_regression(impl, artifact):
    """Adapter training recovers a known low-rank update and preserves merged inference equivalence."""
    out = run(impl.lora_delta, impl.lora_linear)
    artifact('low_rank_regression', out)
    assert out['final_mse'] < 1e-5, f"Regression MSE: {out['final_mse']}"
    assert out['merge_error'] < 1e-5
