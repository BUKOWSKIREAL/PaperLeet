import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.flash_attention.experiment import run


@pytest.mark.t0
def test_shapes_partial_blocks(impl):
    """Rectangular attention accepts partial edge tiles and returns one log-sum-exp per query."""
    q, k, v = torch.randn(7, 3), torch.randn(11, 3), torch.randn(11, 5)
    output, lse = impl.block_attention(q, k, v, 4)
    assert output.shape == (7, 5) and lse.shape == (7,)
    assert output.dtype == q.dtype and torch.isfinite(output).all()


@pytest.mark.t1
def test_hand_calculated_two_blocks(impl):
    """Combining masses exp(0)=1 and exp(log 3)=3 produces weighted value 5 and LSE log 4."""
    m, l, acc = torch.tensor([-torch.inf], dtype=torch.float64), torch.zeros(1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64)
    for score, value in [(0., 2.), (math.log(3), 6.)]:
        m, l, acc = impl.online_softmax_update(torch.tensor([[score]], dtype=m.dtype), torch.tensor([[value]], dtype=m.dtype), m, l, acc)
    close(acc / l[:, None], torch.tensor([[5.]], dtype=m.dtype))
    close(m + l.log(), torch.tensor([math.log(4)], dtype=m.dtype))


@pytest.mark.t2
def test_dense_equivalence_and_block_partition(impl):
    """Different block boundaries preserve exact softmax attention and log-partition values."""
    q, k, v = torch.randn(9, 4, dtype=torch.float64), torch.randn(13, 4, dtype=torch.float64), torch.randn(13, 6, dtype=torch.float64)
    scores = q @ k.T / 2
    for size in (1, 2, 4, 20):
        output, lse = impl.block_attention(q, k, v, size)
        close(output, scores.softmax(-1) @ v, atol=1e-12, rtol=1e-12)
        close(lse, scores.logsumexp(-1), atol=1e-12, rtol=1e-12)


@pytest.mark.t2
def test_large_scores_stable_and_shift_invariant(impl):
    """Online accumulation remains finite for scores near 10000 and shifts only LSE."""
    scores = torch.tensor([[10000., -10000., 9999.], [-10000., 10000., -9999.]], dtype=torch.float64)
    values = torch.tensor([[1., 0.], [0., 2.], [-3., 4.]], dtype=scores.dtype)
    result = []
    for shift in (0., 2345.):
        m = scores.new_full((2,), -torch.inf)
        l, acc = scores.new_zeros(2), scores.new_zeros(2, 2)
        for j in range(3):
            m, l, acc = impl.online_softmax_update(scores[:, j:j+1] + shift, values[j:j+1], m, l, acc)
        result.append((acc / l[:, None], m + l.log()))
    close(result[0][0], scores.softmax(-1) @ values, atol=1e-10, rtol=1e-10)
    close(result[1][0], result[0][0], atol=1e-10, rtol=1e-10)
    close(result[1][1] - result[0][1], torch.full((2,), 2345., dtype=scores.dtype))


@pytest.mark.t2
def test_update_does_not_modify_state(impl):
    """State inputs remain reusable when branching a streaming computation."""
    scores, values = torch.randn(2, 3), torch.randn(3, 4)
    state = (torch.zeros(2), torch.ones(2), torch.randn(2, 4))
    before = [x.clone() for x in state]
    impl.online_softmax_update(scores, values, *state)
    for actual, expected in zip(state, before):
        close(actual, expected)


@pytest.mark.t2
def test_dense_gradients(impl):
    """The differentiable educational recurrence preserves dense q/k/v and LSE gradients."""
    args = [torch.randn(5, 3, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    clones = [x.detach().clone().requires_grad_() for x in args]
    out, lse = impl.block_attention(*args, 2)
    q, k, v = clones
    scores = q @ k.T / math.sqrt(3)
    target = scores.softmax(-1) @ v
    for actual, expected in zip(torch.autograd.grad(out.square().sum() + lse.sum(), args),
                                torch.autograd.grad(target.square().sum() + scores.logsumexp(-1).sum(), clones)):
        close(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.t3
def test_block_size_experiment(impl, artifact):
    """A 37-token experiment confirms exactness across tiny, partial, and oversized tiles."""
    result = run(impl.block_attention)
    artifact("block_size_exactness", result)
    assert max(row["output_error"] for row in result["rows"]) < 1e-10
    assert max(row["lse_error"] for row in result["rows"]) < 1e-10
