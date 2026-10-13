import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.rl.dapo.experiment import run


@pytest.mark.t0
def test_groups_and_contract(impl):
    """Only mixed-outcome prompts survive; malformed single-member groups are rejected."""
    rewards = torch.tensor([[1., 1., 1.], [-1., -1., -1.], [-1., 1., -1.]])
    out = impl.dynamic_group_mask(rewards)
    assert out.dtype == torch.bool and out.tolist() == [False, False, True]
    with pytest.raises(ValueError):
        impl.dynamic_group_mask(torch.ones(3, 1))


@pytest.mark.t1
def test_reference(impl, ref):
    """The asymmetric clipped surrogate agrees for heterogeneous response lengths."""
    x, old = torch.randn(6, 5, dtype=torch.float64), torch.randn(6, 5, dtype=torch.float64)
    mask = (torch.arange(5)[None] < torch.arange(6)[:, None]).double()
    adv = torch.linspace(-2, 2, 6, dtype=x.dtype)
    close(impl.dapo_loss(x, old, adv, mask, .13, .31), ref.dapo_loss(x, old, adv, mask, .13, .31))


@pytest.mark.t2
def test_token_average_gradient(impl):
    """Each token, rather than each response, gets equal normalization weight."""
    x = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1., 0., 0.], [1., 1., 1.]], dtype=x.dtype)
    impl.dapo_loss(x, x.detach(), torch.tensor([2., -1.], dtype=x.dtype), mask).backward()
    close(x.grad, torch.tensor([[-.5, 0., 0.], [.25, .25, .25]], dtype=x.dtype))


@pytest.mark.t2
def test_upper_and_lower_clip_independent(impl):
    """A positive ratio 1.25 still learns under clip_high=.28; a negative ratio .75 clips."""
    x = torch.tensor([[1.25], [.75]], dtype=torch.float64).log().requires_grad_()
    impl.dapo_loss(x, torch.zeros_like(x), torch.tensor([1., -1.], dtype=x.dtype), torch.ones_like(x)).backward()
    close(x.grad[:, 0], torch.tensor([-.625, 0.], dtype=x.dtype))


@pytest.mark.t2
def test_padding_and_empty_mask(impl):
    """Masked values cannot overflow or affect objective/gradients; empty masks yield finite zero."""
    x = torch.tensor([[.1, 10000.]], requires_grad=True)
    old, adv, mask = torch.zeros_like(x), torch.ones(1), torch.tensor([[1., 0.]])
    value = impl.dapo_loss(x, old, adv, mask)
    close(value, -torch.tensor(.1).exp())
    value.backward()
    close(x.grad[0, 1], torch.tensor(0.))
    close(impl.dapo_loss(x, old, adv, mask * 0), torch.tensor(0.))


@pytest.mark.t2
def test_sampling_snapshots_frozen(impl):
    """Behavior scores and outcome advantages are constants during policy optimization."""
    x = torch.zeros(2, 2, requires_grad=True)
    old = torch.zeros_like(x, requires_grad=True)
    adv = torch.tensor([1., -1.], requires_grad=True)
    impl.dapo_loss(x, old, adv, torch.ones_like(x)).backward()
    assert old.grad is None or torch.count_nonzero(old.grad) == 0
    assert adv.grad is None or torch.count_nonzero(adv.grad) == 0


@pytest.mark.t3
def test_group_and_clip_experiment(impl, artifact):
    """The experiment uses submitted functions to retain mixed groups and expose both clip plateaus."""
    out = run(impl.dynamic_group_mask, impl.dapo_loss)
    artifact('dynamic_sampling_and_clipping', out)
    assert out['retained'] == [False, True, False]
    close(torch.tensor(out['positive_loss'][-1]), torch.tensor(-1.28))
    close(torch.tensor(out['negative_loss'][0]), torch.tensor(.8))
