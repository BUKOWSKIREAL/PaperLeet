import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.rl.gae.experiment import run


def inputs(t=9, n=3):
    r, v, nv = [torch.randn(t, n, dtype=torch.float64) for _ in range(3)]
    term, trunc = torch.rand(t, n) < .2, torch.rand(t, n) < .2
    return r, v, nv, term, trunc


@pytest.mark.t0
def test_shape_dtype(impl):
    """GAE preserves the time/environment axes and floating dtype."""
    args = inputs()
    for i, x in enumerate(impl.compute_gae(*args)):
        assert x.shape == args[0].shape, f'output[{i}] expected shape {args[0].shape}, got {x.shape}'
        assert x.dtype == args[0].dtype, f'output[{i}] expected dtype {args[0].dtype}, got {x.dtype}'


@pytest.mark.t0
def test_single_transition(impl):
    """A single transition needs no future advantage accumulator."""
    r, v, nv, term, trunc = inputs(1, 1)
    a, _ = impl.compute_gae(r, v, nv, term, trunc)
    close(a, r + .99 * (~term) * nv - v)


@pytest.mark.t0
def test_finite(impl):
    """Finite bounded inputs produce finite advantages and returns."""
    for i, x in enumerate(impl.compute_gae(*inputs())):
        assert torch.isfinite(x).all(), f'output[{i}] expected all finite, got {x.tolist()}'


@pytest.mark.t1
def test_reference(impl, ref):
    """Random mixed episode endings follow the specified recurrence."""
    args=inputs(17, 5)
    for got, expected in zip(impl.compute_gae(*args), ref.compute_gae(*args)):
        close(got, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.t2
def test_lambda_zero_is_td(impl):
    """At lambda zero the future advantage coefficient vanishes."""
    r, v, nv, term, trunc = inputs()
    a, _ = impl.compute_gae(r, v, nv, term, trunc, lam=0)
    close(a, r + .99 * (~term) * nv - v)


@pytest.mark.t2
def test_lambda_one_telescopes(impl):
    """Consistent successive values telescope into a bootstrapped Monte Carlo return."""
    r, v, _, term, trunc = inputs()
    term.zero_(); trunc.zero_()
    final = torch.randn(1, 3, dtype=r.dtype)
    nv = torch.cat([v[1:], final])
    a, _ = impl.compute_gae(r, v, nv, term, trunc, gamma=.91, lam=1)
    expected = torch.stack([sum(.91 ** k * r[t+k] for k in range(len(r)-t)) + .91 ** (len(r)-t) * final[0] - v[t] for t in range(len(r))])
    close(a, expected)


@pytest.mark.t2
def test_episode_isolation(impl):
    """Changing a new episode cannot change advantages before an ending."""
    r, v, nv, term, trunc = inputs()
    term.zero_(); trunc.zero_(); term[3, 0]=True; trunc[3, 1:]=True
    original = impl.compute_gae(r, v, nv, term, trunc)[0]
    changed=r.clone(); changed[4:] += 100
    got=impl.compute_gae(changed, v, nv, term, trunc)[0]
    close(got[:4], original[:4])


@pytest.mark.t2
def test_termination_and_truncation_bootstrap(impl):
    """Only termination removes the next-state value; truncation retains bootstrap."""
    r=v=torch.zeros(1, 2, dtype=torch.float64)
    term=torch.tensor([[True, False]]); trunc=~term
    a,_=impl.compute_gae(r,v,torch.ones_like(r)*2,term,trunc,gamma=.9)
    close(a,torch.tensor([[0.,1.8]],dtype=r.dtype))


@pytest.mark.t2
def test_reward_linearity(impl):
    """GAE is affine in rewards, and linear after subtracting its zero-reward baseline."""
    r,v,nv,term,trunc=inputs()
    base=impl.compute_gae(torch.zeros_like(r),v,nv,term,trunc)[0]
    other=torch.randn_like(r)
    fn=lambda z:impl.compute_gae(z,v,nv,term,trunc)[0]-base
    close(fn(2*r-3*other),2*fn(r)-3*fn(other))


@pytest.mark.t2
def test_returns_contract(impl):
    """The value target is advantage plus current value, including lambda below one."""
    args=inputs(); a,ret=impl.compute_gae(*args,lam=.4)
    close(ret,a+args[1])


@pytest.mark.t3
def test_bias_variance_tradeoff(impl, artifact):
    """With an imperfect critic, lambda one removes gradient bias at the cost of variance."""
    result=run(impl.compute_gae)
    artifact('bias_variance',result)
    lo,hi=result[0],result[-1]
    assert lo['bias'] > hi['bias'], f'lambda endpoints expected bias(0)>bias(1), got {lo["bias"]}, {hi["bias"]}'
    assert hi['variance'] > lo['variance'], f'lambda endpoints expected var(1)>var(0), got {hi["variance"]}, {lo["variance"]}'
    threshold = 0.007998117879200427
    assert hi['bias'] < threshold, f'lambda=1 gradient expected bias<{threshold}, got {hi["bias"]}'
