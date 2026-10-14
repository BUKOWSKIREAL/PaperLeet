import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.world_models.causal_jepa.experiment import run


def fixture(dtype=torch.float64):
    h = torch.arange(12, dtype=dtype).reshape(1, 3, 2, 2)
    mask = torch.tensor([[False, True]])
    p = torch.tensor([[1., 2.], [-1., .5]], dtype=dtype)
    time = torch.arange(10, dtype=dtype).reshape(5, 2) / 10
    token = torch.tensor([-.2, .3], dtype=dtype)
    return h, mask, p, time, token


@pytest.mark.t0
def test_shapes_dtype_and_inputs(impl):
    """Masked inputs have a future horizon and a boolean token-level target mask."""
    for dtype in (torch.float32, torch.float64):
        h, mask, p, time, token = fixture(dtype); before = h.clone()
        x, target_mask = impl.object_mask_inputs(h, mask, p, time, token)
        assert x.shape == (1, 5, 2, 2) and x.dtype == dtype
        assert target_mask.shape == (1, 5, 2) and target_mask.dtype == torch.bool
        close(h, before)


@pytest.mark.t0
def test_invalid_contracts(impl):
    """The future is nonempty and object selection is a boolean mask with matching batch/slots."""
    h, mask, p, time, token = fixture()
    with pytest.raises(ValueError): impl.object_mask_inputs(h, mask.long(), p, time, token)
    with pytest.raises(ValueError): impl.object_mask_inputs(h, mask, p, time[:3], token)
    target = torch.ones(1, 5, 2, 2)
    for length in (0, 5, True):
        with pytest.raises(ValueError): impl.masked_prediction_loss(target, target, length, mask)


@pytest.mark.t1
def test_anchor_query_and_visible_context_oracle(impl):
    """First frame stays visible; future queries combine projected identity, mask token and time."""
    h, mask, p, time, token = fixture()
    x, target_mask = impl.object_mask_inputs(h, mask, p, time, token)
    expected = torch.empty_like(x)
    expected_mask = torch.zeros_like(target_mask)
    for t in range(5):
        for s in range(2):
            hidden = t >= 3 or (t > 0 and s == 1)
            expected_mask[0, t, s] = hidden
            if hidden:
                expected[0, t, s] = torch.stack([sum(p[j, k] * h[0, 0, s, k] for k in range(2)) for j in range(2)]) + token + time[t]
            else:
                expected[0, t, s] = h[0, t, s] + time[t]
    close(x, expected); assert torch.equal(target_mask, expected_mask)


@pytest.mark.t1
def test_separately_normalized_loss_oracle(impl):
    """History and future are each mean-normalized, then added; visible history is excluded."""
    predictions = torch.arange(20, dtype=torch.float64).reshape(1, 5, 2, 2) / 10
    targets = torch.zeros_like(predictions); mask = torch.tensor([[False, True]])
    total, history, future = impl.masked_prediction_loss(predictions, targets, 3, mask)
    h = sum(predictions[0, t, 1, d].item() ** 2 for t in (1, 2) for d in range(2)) / 4
    f = sum(predictions[0, t, s, d].item() ** 2 for t in (3, 4) for s in range(2) for d in range(2)) / 8
    close(history, predictions.new_tensor(h)); close(future, predictions.new_tensor(f)); close(total, predictions.new_tensor(h + f))


@pytest.mark.t2
def test_hidden_trajectory_independence_and_slot_permutation(impl):
    """Changing masked post-anchor states cannot leak into inputs; slot permutation commutes with masking."""
    h, mask, p, time, token = fixture()
    x, tm = impl.object_mask_inputs(h, mask, p, time, token)
    changed = h.clone(); changed[:, 1:, 1] += 1000
    close(impl.object_mask_inputs(changed, mask, p, time, token)[0], x)
    permutation = torch.tensor([1, 0])
    xp, tmp = impl.object_mask_inputs(h[:, :, permutation], mask[:, permutation], p, time, token)
    close(xp, x[:, :, permutation]); assert torch.equal(tmp, tm[:, :, permutation])


@pytest.mark.t2
def test_prediction_gradient_and_frozen_targets(impl):
    """Only masked history and future predictions get gradients; frozen encoder targets get none."""
    predictions = torch.arange(20, dtype=torch.float64).reshape(1, 5, 2, 2).requires_grad_()
    target = torch.ones_like(predictions, requires_grad=True); mask = torch.tensor([[False, True]])
    total, _, _ = impl.masked_prediction_loss(predictions, target, 3, mask)
    gp, gt = torch.autograd.grad(total, (predictions, target), allow_unused=True)
    expected = torch.zeros_like(predictions)
    expected[:, 1:3, 1] = 2 * (predictions.detach()[:, 1:3, 1] - 1) / 4
    expected[:, 3:] = 2 * (predictions.detach()[:, 3:] - 1) / 8
    close(gp, expected); assert gt is None or not gt.any()


@pytest.mark.t2
def test_empty_history_mask_and_per_example_masks(impl):
    """Unmasked inference history contributes zero; per-example selections use the selected element count."""
    pred = torch.randn(2, 4, 2, 1, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(pred); none = torch.zeros(2, 2, dtype=torch.bool)
    total, history, future = impl.masked_prediction_loss(pred, target, 2, none)
    close(history, pred.new_tensor(0.)); close(total, future)
    mask = torch.tensor([[True, False], [True, True]])
    _, history, _ = impl.masked_prediction_loss(pred, target, 2, mask)
    expected = (pred[:, 1] - target[:, 1]).square()[mask].mean()
    close(history, expected)
    _, history, _ = impl.masked_prediction_loss(pred, target, 1, mask)
    close(history, pred.new_tensor(0.))


@pytest.mark.t3
def test_object_interaction_experiment(impl, artifact):
    """Submitted masking removes target-history shortcuts; submitted losses train a context-based probe."""
    result = run(impl.object_mask_inputs, impl.masked_prediction_loss)
    artifact('causal_jepa_object_masking', result)
    m = result['metrics']
    assert m['hidden_history_leakage'] == 0.
    assert m['final_loss'] < .001 * m['initial_loss']
    assert m['masked_tokens_per_sample'] == 7.
