import pytest
import torch
import torch.nn.functional as F
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.positionwise_ffn.experiment import run


def parameters(dtype=torch.float64):
    return [torch.randn(*shape, dtype=dtype) for shape in ((7, 3), (7,), (3, 7), (3,))]


@pytest.mark.t0
def test_shape_dtype_and_input_preservation(impl):
    """Arbitrary leading batch/sequence axes and input dtype are preserved without mutation."""
    for dtype in (torch.float32, torch.float64):
        for shape in ((3,), (2, 4, 3), (2, 1, 4, 3), (2, 0, 3)):
            inputs = [torch.randn(*shape, dtype=dtype), *parameters(dtype)]
            before = [p.clone() for p in inputs]
            got = impl.positionwise_ffn(*inputs)
            assert got.shape == inputs[0].shape and got.dtype == dtype
            assert got.device == inputs[0].device
            for p, old in zip(inputs, before):
                close(p, old)


@pytest.mark.t1
def test_hand_computed_relu_and_both_biases(impl):
    """A non-square hand example verifies input/output orientation, ReLU and both biases."""
    x = torch.tensor([[1., -2.], [-1., 1.]], dtype=torch.float64)
    w1 = torch.tensor([[1., 0.], [0., 1.], [1., -1.]], dtype=x.dtype)
    b1 = torch.tensor([-2., 1., -.5], dtype=x.dtype)
    w2 = torch.tensor([[2., -1., 1.], [1., 3., -2.]], dtype=x.dtype)
    b2 = torch.tensor([.5, -1.], dtype=x.dtype)
    # Hidden values are [0,0,2.5] and [0,2,0].
    expected = torch.tensor([[3., -6.], [-1.5, 5.]], dtype=x.dtype)
    close(impl.positionwise_ffn(x, w1, b1, w2, b2), expected)


@pytest.mark.t2
def test_token_permutation_and_independence(impl):
    """Permuting tokens only permutes results; modifying one token leaves every other token intact."""
    x, params = torch.randn(2, 5, 3, dtype=torch.float64), parameters()
    original = impl.positionwise_ffn(x, *params)
    perm = torch.tensor([3, 0, 4, 1, 2])
    close(impl.positionwise_ffn(x[:, perm], *params), original[:, perm])
    changed = x.clone()
    changed[0, 2] += 10
    actual = impl.positionwise_ffn(changed, *params)
    close(actual[1], original[1])
    close(actual[0, [0, 1, 3, 4]], original[0, [0, 1, 3, 4]])


@pytest.mark.t2
def test_gradients_for_input_and_all_parameters(impl):
    """Input, both weights and both biases have the independent linear-ReLU-linear gradients."""
    inputs = [p.requires_grad_() for p in [torch.randn(2, 4, 3, dtype=torch.float64), *parameters()]]
    clones = [p.detach().clone().requires_grad_() for p in inputs]
    x, w1, b1, w2, b2 = clones
    expected = F.linear(F.relu(F.linear(x, w1, b1)), w2, b2)
    got = impl.positionwise_ffn(*inputs)
    close(got, expected)
    probe = torch.randn_like(got)
    actual_grads = torch.autograd.grad((got * probe).sum(), inputs)
    expected_grads = torch.autograd.grad((expected * probe).sum(), clones)
    for actual, wanted in zip(actual_grads, expected_grads):
        close(actual, wanted, atol=1e-10, rtol=1e-10)


@pytest.mark.t2
def test_relu_gate_has_zero_negative_derivative(impl):
    """Strictly negative hidden activations close the gate and leave only the output bias."""
    x = torch.tensor([[-1., -2., -3.]], dtype=torch.float64, requires_grad=True)
    w1 = torch.eye(3, dtype=x.dtype)
    b1 = torch.full((3,), -.2, dtype=x.dtype)
    w2 = torch.tensor([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]], dtype=x.dtype)
    b2 = torch.tensor([.2, -.1, .8], dtype=x.dtype)
    got = impl.positionwise_ffn(x, w1, b1, w2, b2)
    close(got, b2[None])
    close(torch.autograd.grad(got.sum(), x)[0], torch.zeros_like(x))


@pytest.mark.t2
def test_noncontiguous_and_duplicate_tokens(impl):
    """Strided tensors remain valid and identical tokens receive identical outputs across batches."""
    x = torch.randn(4, 2, 3, dtype=torch.float64).transpose(0, 1)
    params = parameters()
    close(impl.positionwise_ffn(x, *params), impl.positionwise_ffn(x.contiguous(), *params))
    repeated = x[0, 0].expand(2, 5, 3)
    actual = impl.positionwise_ffn(repeated, *params)
    close(actual, actual[0, 0].expand_as(actual))


@pytest.mark.t3
def test_nonlinear_token_regression_experiment(impl, artifact):
    """The submitted FFN learns a nonlinear analytic map and generalizes to held-out tokens."""
    result = run(impl.positionwise_ffn)
    artifact('nonlinear_token_regression', result)
    metrics = result['metrics']
    assert metrics['final_training_mse'] < metrics['initial_training_mse'] * .05
    assert metrics['heldout_mse'] < .003
