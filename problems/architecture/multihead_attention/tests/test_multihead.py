import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.multihead_attention.experiment import run


def inputs(batch=2, queries=3, keys=5, width=6, gradients=False):
    generator = torch.Generator().manual_seed(143)
    shapes = [(batch, queries, width), (batch, keys, width), (batch, keys, width)] + [(width, width)] * 4
    return [torch.randn(shape, generator=generator, dtype=torch.float64).requires_grad_(gradients) for shape in shapes]


def headwise_oracle(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    """Independent per-head projection and PyTorch SDPA, without packed reshapes."""
    width = query.shape[-1] // num_heads
    projected = []
    all_weights = []
    for head in range(num_heads):
        columns = slice(head * width, (head + 1) * width)
        q, k, v = query @ w_q[:, columns], key @ w_k[:, columns], value @ w_v[:, columns]
        visible = None if mask is None else torch.broadcast_to(mask, (query.shape[0], num_heads, query.shape[1], key.shape[1]))[:, head]
        projected.append(torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=visible))
        scores = torch.einsum("bqd,bkd->bqk", q, k) / math.sqrt(width)
        if visible is not None:
            scores = scores.masked_fill(~visible, -torch.inf)
            scores = torch.where(visible.any(-1, keepdim=True), scores, 0.)
        probabilities = scores.softmax(-1)
        all_weights.append(probabilities if visible is None else probabilities.masked_fill(~visible, 0.))
    return torch.cat(projected, dim=-1) @ w_o, torch.stack(all_weights, dim=1)


@pytest.mark.t0
def test_shapes_dtype_and_inputs(impl):
    """Cross-attention retains batch, query length, dtype, and all supplied tensors."""
    tensors = inputs()
    originals = [x.clone() for x in tensors]
    output, weights = impl.multi_head_attention(*tensors, 3)
    assert output.shape == (2, 3, 6) and weights.shape == (2, 3, 3, 5)
    assert output.dtype == weights.dtype == torch.float64
    assert output.device == tensors[0].device
    for tensor, original in zip(tensors, originals):
        close(tensor, original)


@pytest.mark.t1
def test_independent_headwise_projection(impl):
    """Each head uses its own projection columns before concatenation and output projection."""
    tensors = inputs()
    output, weights = impl.multi_head_attention(*tensors, 3)
    expected, expected_weights = headwise_oracle(*tensors, 3)
    close(output, expected, atol=1e-10, rtol=1e-10)
    close(weights, expected_weights, atol=1e-10, rtol=1e-10)


@pytest.mark.t2
def test_broadcast_masks_and_empty_rows(impl):
    """Per-batch masks broadcast across heads and completely hidden queries are finite zero."""
    tensors = inputs(gradients=True)
    mask = torch.tensor([[[[True, False, True, False, True], [False] * 5, [True] * 5]],
                         [[[False, True, False, True, False], [False] * 5, [True] * 5]]])
    output, weights = impl.multi_head_attention(*tensors, 3, mask)
    expected, expected_weights = headwise_oracle(*tensors, 3, mask)
    close(output, expected, atol=1e-10, rtol=1e-10)
    close(weights, expected_weights)
    assert torch.equal(output[:, 1], torch.zeros_like(output[:, 1]))
    assert torch.equal(weights[:, :, 1], torch.zeros_like(weights[:, :, 1]))
    empty_gradients = torch.autograd.grad(output[:, 1].sum(), tensors)
    for gradient in empty_gradients:
        assert torch.isfinite(gradient).all()
        close(gradient, torch.zeros_like(gradient))


@pytest.mark.t2
def test_all_projection_and_input_gradients(impl):
    """Gradients agree for q/k/v and all four projection matrices with head-specific masks."""
    tensors = inputs(batch=1, queries=2, keys=4, width=4, gradients=True)
    oracle_inputs = [x.detach().clone().requires_grad_() for x in tensors]
    mask = torch.tensor([[[[True, True, False, False], [True, True, True, False]],
                          [[False, True, True, False], [False, True, True, True]]]])
    output, weights = impl.multi_head_attention(*tensors, 2, mask)
    expected, expected_weights = headwise_oracle(*oracle_inputs, 2, mask)
    close(output, expected, atol=1e-10, rtol=1e-10)
    objective = output.square().sum() + .3 * weights.square().sum()
    expected_objective = expected.square().sum() + .3 * expected_weights.square().sum()
    for actual, reference in zip(torch.autograd.grad(objective, tensors),
                                 torch.autograd.grad(expected_objective, oracle_inputs)):
        close(actual, reference, atol=1e-9, rtol=1e-9)


@pytest.mark.t2
def test_single_head_and_noncontiguous_inputs(impl):
    """One head reduces to ordinary projected attention, including noncontiguous tensor views."""
    tensors = inputs(width=4)
    tensors = [tensor.transpose(-1, -2).contiguous().transpose(-1, -2) for tensor in tensors]
    assert not tensors[0].is_contiguous()
    output, weights = impl.multi_head_attention(*tensors, 1)
    expected, expected_weights = headwise_oracle(*tensors, 1)
    close(output, expected, atol=1e-10, rtol=1e-10)
    close(weights, expected_weights)


@pytest.mark.t2
def test_causal_future_independence(impl):
    """A future token cannot affect any earlier self-attention query."""
    tensors = inputs(queries=5, keys=5, width=4)
    tensors[1] = tensors[0].clone()
    tensors[2] = tensors[0].clone()
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    output, _ = impl.multi_head_attention(*tensors, 2, mask)
    for index in range(3):
        tensors[index] = tensors[index].clone()
        tensors[index][:, 3:] += 100
    changed, _ = impl.multi_head_attention(*tensors, 2, mask)
    close(changed[:, :3], output[:, :3])


@pytest.mark.t3
def test_two_head_routing_experiment(impl, artifact):
    """Two heads attend to different synthetic tokens and match the analytic mixed output."""
    result = run(impl.multi_head_attention)
    artifact("two_head_routing", result)
    assert result["metrics"]["maximum_error"] < 1e-12
    assert result["metrics"]["head_weight_error"] < 1e-12
    assert result["metrics"]["head_separation_l1"] > 1.8
