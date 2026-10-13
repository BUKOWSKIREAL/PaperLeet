import math
import pytest
import torch
import torch.nn.functional as F
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.layer_norm.experiment import run


@pytest.mark.t0
def test_shapes_dtype_and_no_mutation(impl):
    """LayerNorm preserves arbitrary leading dimensions, dtype and all inputs."""
    for shape, dtype in [((7,), torch.float64), ((2, 3, 5, 7), torch.float32)]:
        x = torch.randn(shape, dtype=dtype)
        weight, bias = torch.randn(7, dtype=dtype), torch.randn(7, dtype=dtype)
        before = [value.clone() for value in (x, weight, bias)]
        output = impl.layer_norm(x, weight, bias)
        assert output.shape == x.shape and output.dtype == dtype and output.device == x.device
        for value, original in zip((x, weight, bias), before):
            close(value, original)


@pytest.mark.t1
def test_analytical_population_variance_and_affine(impl):
    """Three features have mean 3 and population variance 8/3 before affine scaling."""
    x = torch.tensor([[1., 3., 5.]], dtype=torch.float64)
    weight = torch.tensor([2., -1., .5], dtype=x.dtype)
    bias = torch.tensor([1., 4., -2.], dtype=x.dtype)
    eps = .25
    expected = torch.tensor([[-4. / math.sqrt(8. / 3 + eps) + 1., 4.,
                             1. / math.sqrt(8. / 3 + eps) - 2.]], dtype=x.dtype)
    close(impl.layer_norm(x, weight, bias, eps), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_constant_single_feature_and_small_variance(impl):
    """Constant rows return bias; one feature stays finite; epsilon stabilizes tiny variance."""
    for x in (torch.full((2, 3, 4), 8., dtype=torch.float64),
              torch.tensor([[[2.], [-7.]]], dtype=torch.float64)):
        weight, bias = torch.randn(x.shape[-1], dtype=x.dtype), torch.randn(x.shape[-1], dtype=x.dtype)
        output = impl.layer_norm(x, weight, bias)
        assert torch.isfinite(output).all()
        close(output, bias.expand_as(x), atol=1e-12, rtol=1e-12)
    x = torch.tensor([[1. - 1e-7, 1. + 1e-7]], dtype=torch.float64)
    weight, bias = torch.ones(2, dtype=x.dtype), torch.zeros(2, dtype=x.dtype)
    close(impl.layer_norm(x, weight, bias, 1e-4), F.layer_norm(x, (2,), weight, bias, 1e-4),
          atol=1e-12, rtol=1e-10)


@pytest.mark.t2
def test_token_independence_translation_and_scale(impl):
    """Statistics are token-local; shifts cancel and scale sensitivity follows epsilon exactly."""
    x = torch.randn(2, 3, 5, dtype=torch.float64)
    weight, bias = torch.ones(5, dtype=x.dtype), torch.zeros(5, dtype=x.dtype)
    output = impl.layer_norm(x, weight, bias, 1e-5)
    shifted = impl.layer_norm(x + torch.randn(2, 3, 1, dtype=x.dtype) * 20, weight, bias, 1e-5)
    close(shifted, output, atol=1e-12, rtol=1e-12)
    changed = x.clone()
    changed[1] *= 100
    changed[0, 1:] += 200
    close(impl.layer_norm(changed, weight, bias)[0, 0], output[0, 0], atol=1e-12, rtol=1e-12)
    # Scaling x by a and epsilon by a^2 preserves normalized values for a > 0.
    close(impl.layer_norm(7 * x, weight, bias, 49e-5), output, atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_values_and_gradients_against_library_oracle(impl):
    """Input, scale and bias gradients agree with PyTorch's independent LayerNorm operator."""
    values = [torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True),
              torch.randn(5, dtype=torch.float64, requires_grad=True),
              torch.randn(5, dtype=torch.float64, requires_grad=True)]
    clones = [value.detach().clone().requires_grad_() for value in values]
    output = impl.layer_norm(*values, eps=1e-4)
    expected = F.layer_norm(clones[0], (5,), clones[1], clones[2], eps=1e-4)
    close(output, expected, atol=1e-11, rtol=1e-11)
    upstream = torch.randn_like(output)
    for actual, reference in zip(torch.autograd.grad((output * upstream).sum(), values),
                                 torch.autograd.grad((expected * upstream).sum(), clones)):
        close(actual, reference, atol=1e-10, rtol=1e-10)


@pytest.mark.t2
def test_post_norm_residual_order_and_gradients(impl):
    """Add before normalization; both residual branches retain gradients and are not mutated."""
    x = torch.tensor([[[1., 2., 9.], [-4., 2., 0.]]], dtype=torch.float64, requires_grad=True)
    branch = torch.tensor([[[3., -1., 2.], [5., 7., -2.]]], dtype=x.dtype, requires_grad=True)
    weight, bias = torch.tensor([2., -.5, 1.], dtype=x.dtype), torch.tensor([1., 2., -1.], dtype=x.dtype)
    before = [value.detach().clone() for value in (x, branch)]
    clones = [value.detach().clone().requires_grad_() for value in (x, branch)]
    output = impl.post_norm_residual(x, branch, weight, bias, 1e-4)
    expected = F.layer_norm(clones[0] + clones[1], (3,), weight, bias, 1e-4)
    close(output, expected, atol=1e-12, rtol=1e-12)
    for actual, reference in zip(torch.autograd.grad(output.square().sum(), (x, branch)),
                                 torch.autograd.grad(expected.square().sum(), clones)):
        close(actual, reference, atol=1e-10, rtol=1e-10)
    for value, original in zip((x, branch), before):
        close(value, original)


@pytest.mark.t3
def test_varied_token_add_norm_experiment(impl, artifact):
    """Eight differently scaled tokens match scalar normalization and ignore other tokens."""
    result = run(impl.layer_norm, impl.post_norm_residual)
    artifact("token_local_add_norm", result)
    assert result["metrics"]["maximum_output_error"] < 1e-12
    assert result["metrics"]["maximum_mean_magnitude"] < 1e-12
    assert result["metrics"]["other_token_influence"] == 0
    assert result["metrics"]["composition_error"] == 0
