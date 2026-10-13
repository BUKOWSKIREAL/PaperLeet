import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.rl.opd.experiment import run


@pytest.mark.t0
def test_rollout_contract(impl):
    """Fixed-length rollout returns integer tokens and the exact preceding Markov states."""
    table = torch.zeros(4, 3, requires_grad=True)
    tokens, prefixes = impl.sample_student(table, 16, 6, torch.Generator().manual_seed(8))
    assert tokens.shape == prefixes.shape == (16, 6)
    assert tokens.dtype == prefixes.dtype == torch.long
    assert not tokens.requires_grad and not prefixes.requires_grad
    assert torch.all(prefixes[:, 0] == 3)
    assert torch.equal(prefixes[:, 1:], tokens[:, :-1])
    with pytest.raises(ValueError):
        impl.sample_student(torch.zeros(3, 3), 2, 2, torch.Generator())
    with pytest.raises(ValueError):
        impl.distillation_loss(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), torch.ones(1, 1), 'unknown')


@pytest.mark.t1
def test_both_divergences(impl, ref):
    """Forward and reverse full-vocabulary KL match the declared finite-logit contract."""
    s, t = torch.randn(3, 4, 7, dtype=torch.float64), torch.randn(3, 4, 7, dtype=torch.float64)
    mask = torch.tensor([[1., 1., 0., 0.], [1., 0., 0., 0.], [1., 1., 1., 1.]], dtype=s.dtype)
    for direction in ('forward', 'reverse'):
        close(impl.distillation_loss(s, t, mask, direction), ref.distillation_loss(s, t, mask, direction))


@pytest.mark.t2
def test_sampling_distribution_and_prefix_dependency(impl):
    """BOS follows student probabilities, and later samples depend on the sampled preceding token."""
    table = torch.tensor([[20., -20.], [-20., 20.], [.25, .75]]).clone()
    table[-1] = table[-1].log()
    tokens, prefixes = impl.sample_student(table, 4096, 3, torch.Generator().manual_seed(42))
    assert abs(float((tokens[:, 0] == 1).float().mean()) - .75) < .025
    assert torch.equal(tokens[:, 0], tokens[:, 1]) and torch.equal(tokens[:, 1], tokens[:, 2])
    assert torch.equal(prefixes[:, 1:], tokens[:, :-1])


@pytest.mark.t2
def test_kl_direction_and_teacher_frozen(impl):
    """Forward KL gradient is student minus teacher; teacher remains frozen for both directions."""
    s = torch.tensor([[[.7, .2, .1]]], dtype=torch.float64).log().requires_grad_()
    t = torch.tensor([[[.2, .3, .5]]], dtype=torch.float64).log().requires_grad_()
    loss = impl.distillation_loss(s, t, torch.ones(1, 1), 'forward')
    loss.backward()
    close(s.grad, s.detach().softmax(-1) - t.detach().softmax(-1))
    assert t.grad is None or torch.count_nonzero(t.grad) == 0
    s.grad = None
    reverse = impl.distillation_loss(s, t, torch.ones(1, 1), 'reverse')
    reverse.backward()
    assert t.grad is None or torch.count_nonzero(t.grad) == 0
    q, delta = s.detach().softmax(-1), s.detach().log_softmax(-1) - t.detach().log_softmax(-1)
    close(s.grad, q * (delta - (q * delta).sum(-1, keepdim=True)))
    assert abs(float(reverse.detach() - loss.detach())) > .01


@pytest.mark.t2
def test_zero_kl_padding_and_shift_invariance(impl):
    """Identical distributions yield zero KL; masked nonfinite logits and additive shifts cannot affect it."""
    s = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    t = torch.randn_like(s)
    mask = torch.tensor([[1., 0., 1.], [1., 0., 0.]], dtype=s.dtype)
    for direction in ('forward', 'reverse'):
        close(impl.distillation_loss(s, s.detach(), mask, direction), torch.zeros((), dtype=s.dtype), atol=1e-12)
        expected = impl.distillation_loss(s, t, mask, direction)
        dirty = s.detach().clone(); dirty[~mask.bool()] = float('nan'); dirty.requires_grad_()
        got = impl.distillation_loss(dirty, t + 1000, mask, direction)
        close(got, expected)
        got.backward()
        close(dirty.grad[~mask.bool()], torch.zeros_like(dirty.grad[~mask.bool()]))


@pytest.mark.t3
def test_student_prefix_distillation(impl, artifact):
    """A fresh student rollout on each update learns all teacher transition rows without teacher sampling."""
    out = run(impl.sample_student, impl.distillation_loss)
    artifact('student_prefix_training', out)
    assert out['final_kl'] < .002, f"Teacher match KL: {out['final_kl']}"
    assert min(out['prefix_visits']) > 100
