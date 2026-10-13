import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.rl.gspo.experiment import run


@pytest.mark.t0
def test_scalar_and_empty_response(impl):
    """Sequence outputs have one value per response; an empty batch mask has zero loss."""
    x = torch.randn(3, 4, requires_grad=True)
    mask = torch.zeros_like(x)
    close(impl.sequence_ratio(x, x, mask), torch.ones(3))
    out = impl.gspo_loss(x, x, torch.ones(3), mask)
    assert out.shape == ()
    close(out, torch.tensor(0.))
    out.backward()
    close(x.grad, torch.zeros_like(x))


@pytest.mark.t1
def test_reference(impl, ref):
    """Mixed advantage signs, response lengths, and clip widths match the formula."""
    x, old = torch.randn(7, 6, dtype=torch.float64), torch.randn(7, 6, dtype=torch.float64)
    mask = (torch.arange(6)[None] < torch.tensor([1, 2, 3, 4, 5, 6, 0])[:, None]).double()
    adv = torch.linspace(-2, 2, 7, dtype=x.dtype)
    close(impl.sequence_ratio(x, old, mask), ref.sequence_ratio(x, old, mask))
    close(impl.gspo_loss(x, old, adv, mask, .12), ref.gspo_loss(x, old, adv, mask, .12))


@pytest.mark.t2
def test_geometric_ratio_and_duplication(impl):
    """Ratios 2 and 0.5 yield sequence ratio 1; repeating tokens does not change it."""
    x = torch.tensor([[2., .5]], dtype=torch.float64).log()
    close(impl.sequence_ratio(x, torch.zeros_like(x), torch.ones_like(x)), torch.ones(1, dtype=x.dtype))
    y = torch.tensor([[1.1, 1.3]], dtype=torch.float64).log()
    expected = torch.tensor([(1.1 * 1.3) ** .5], dtype=y.dtype)
    close(impl.sequence_ratio(y.repeat(1, 3), torch.zeros(1, 6, dtype=y.dtype), torch.ones(1, 6, dtype=y.dtype)), expected)


@pytest.mark.t2
def test_equal_gradient_within_response(impl):
    """GSPO weights all valid log-probabilities equally even when token ratios differ."""
    x = torch.tensor([[2., .5]], dtype=torch.float64).log().requires_grad_()
    loss = impl.gspo_loss(x, torch.zeros_like(x), torch.tensor([1.], dtype=x.dtype), torch.ones_like(x))
    loss.backward()
    close(x.grad, torch.full_like(x, -.5))


@pytest.mark.t2
def test_clipping_sign_and_snapshot_gradients(impl):
    """Positive high ratios and negative low ratios clip; the opposite two cases still learn."""
    x = torch.tensor([[1.5], [.5], [.5], [1.5]], dtype=torch.float64).log().requires_grad_()
    old = torch.zeros_like(x, requires_grad=True)
    adv = torch.tensor([1., -1., 1., -1.], dtype=x.dtype, requires_grad=True)
    impl.gspo_loss(x, old, adv, torch.ones_like(x)).backward()
    close(x.grad[:, 0], torch.tensor([0., 0., -.125, .375], dtype=x.dtype))
    assert old.grad is None or torch.count_nonzero(old.grad) == 0
    assert adv.grad is None or torch.count_nonzero(adv.grad) == 0


@pytest.mark.t2
def test_padding_isolation(impl):
    """Extreme padding values are removed before exponential arithmetic and carry no gradient."""
    x = torch.tensor([[.1, 10000.], [-.1, -10000.]], requires_grad=True)
    mask = torch.tensor([[1., 0.], [1., 0.]])
    old = torch.zeros_like(x)
    adv = torch.tensor([1., -1.])
    value = impl.gspo_loss(x, old, adv, mask)
    close(value, impl.gspo_loss(x[:, :1], old[:, :1], adv, mask[:, :1]))
    value.backward()
    close(x.grad[:, 1], torch.zeros(2))


@pytest.mark.t3
def test_cancellation_experiment(impl, artifact):
    """The student-generated diagnostic distinguishes sequence clipping from token clipping."""
    out = run(impl.sequence_ratio, impl.gspo_loss)
    artifact('sequence_clipping', out)
    close(torch.tensor(out['sequence_ratios']), torch.tensor([1., 1.5, .5]))
    close(torch.tensor(out['gradient']), torch.tensor([[-1/6, -1/6, 0.], [0., 0., 0.], [0., 0., 0.]]))
