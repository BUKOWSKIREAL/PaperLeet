import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.finetuning.opsd_alignment.experiment import run


def fixture(dtype=torch.float64):
    return (torch.arange(60, dtype=dtype).reshape(2, 6, 5), torch.arange(100, dtype=dtype).reshape(2, 10, 5),
            torch.tensor([2, 4]), torch.tensor([5, 7]), torch.tensor([3, 1]))


@pytest.mark.t0
def test_shapes_dtypes_and_empty_responses(impl):
    """Preserve dtype/device; all-empty responses have shape [B,0,V]."""
    for dtype in (torch.float32, torch.float64):
        s, t, sp, tp, length = fixture(dtype)
        a, b, mask = impl.align_rollout_logits(s, t, sp, tp, length)
        assert a.shape == b.shape == (2, 3, 5) and mask.shape == (2, 3)
        assert a.dtype == b.dtype == dtype and mask.dtype == torch.bool
        assert a.device == b.device == s.device
        a, b, mask = impl.align_rollout_logits(s, t, sp, tp, length * 0)
        assert a.shape == b.shape == (2, 0, 5) and mask.shape == (2, 0)


@pytest.mark.t1
def test_causal_offset_and_distinct_context_lengths(impl):
    """Prediction for response token zero comes from the final prompt token."""
    s, t, sp, tp, length = fixture()
    a, b, mask = impl.align_rollout_logits(s, t, sp, tp, length)
    expected_s, expected_t = torch.zeros(2, 3, 5, dtype=s.dtype), torch.zeros(2, 3, 5, dtype=s.dtype)
    for row in range(2):
        for token in range(length[row]):
            expected_s[row, token] = s[row, sp[row] - 1 + token]
            expected_t[row, token] = t[row, tp[row] - 1 + token]
    close(a, expected_s)
    close(b, expected_t)
    assert mask.tolist() == [[True, True, True], [True, False, False]]


@pytest.mark.t1
def test_prompt_length_one_and_truncated_padded_gather(impl):
    """A shorter row may run out of physical padding; masked gathers remain safe."""
    s, t = torch.randn(2, 4, 2), torch.randn(2, 5, 2)
    a, b, mask = impl.align_rollout_logits(s, t, torch.tensor([1, 4]), torch.tensor([2, 5]), torch.tensor([3, 0]))
    close(a[0], s[0, :3]); close(b[0], t[0, 1:4])
    assert torch.count_nonzero(a[1]) == torch.count_nonzero(b[1]) == 0


@pytest.mark.t2
def test_gradients_only_reach_selected_student_positions(impl):
    """Teacher values are detached; padding and prompt-only student positions get zero gradient."""
    s, t, sp, tp, length = fixture()
    s.requires_grad_(); t.requires_grad_()
    a, b, _ = impl.align_rollout_logits(s, t, sp, tp, length)
    gs, gt = torch.autograd.grad(a.sum() + b.sum(), (s, t), allow_unused=True)
    expected = torch.zeros_like(s)
    expected[0, 1:4] = 1; expected[1, 3] = 1
    close(gs, expected)
    assert gt is None


@pytest.mark.t2
def test_padding_invariance_and_no_mutation(impl):
    """Changing logits outside the causal response span cannot change aligned outputs."""
    s, t, sp, tp, length = fixture()
    before = s.clone(), t.clone()
    a, b, mask = impl.align_rollout_logits(s, t, sp, tp, length)
    close(s, before[0]); close(t, before[1])
    s[0, 4:] = 999; s[1, 4:] = -999
    t[0, :4] = 999; t[1, 7:] = -999
    aa, bb, mm = impl.align_rollout_logits(s, t, sp, tp, length)
    close(a, aa); close(b, bb); assert torch.equal(mask, mm)


@pytest.mark.t2
def test_noncontiguous_and_batch_permutation(impl):
    """Strided tensors and reordered examples produce the same per-example result."""
    s, t, sp, tp, length = fixture()
    s, t = s[..., ::2], t[..., ::2]
    args = s, t, sp, tp, length
    expected = impl.align_rollout_logits(*args)
    actual = impl.align_rollout_logits(*(x.flip(0) for x in args))
    for a, b in zip(actual, expected):
        assert torch.equal(a, b.flip(0))


@pytest.mark.t3
def test_two_context_training(impl, artifact):
    """The submitted alignment trains the actual rollout positions across unequal contexts."""
    result = run(impl.align_rollout_logits)
    artifact('two_context_training', result)
    assert result['metrics']['final_correct_probability'] > .92
