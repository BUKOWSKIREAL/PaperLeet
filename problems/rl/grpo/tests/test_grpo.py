import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.rl.grpo.experiment import run


def loss_inputs():
    logp=torch.randn(4,7,dtype=torch.float64)*.3
    old=torch.randn_like(logp)*.3; ref=torch.randn_like(logp)*.3
    adv=torch.tensor([1.,-.5,2.,-1.],dtype=logp.dtype)
    mask=(torch.arange(7)[None,:]<torch.tensor([2,3,5,7])[:,None]).double()
    return logp,old,ref,adv,mask


@pytest.mark.t0
def test_shapes_and_group_validation(impl):
    """Group normalization preserves reward shape and rejects groups without a comparison."""
    out=impl.grpo_advantages(torch.arange(12.).double(),4)
    assert out.shape==(12,),f'advantage expected shape (12,), got {out.shape}'
    with pytest.raises(ValueError):impl.grpo_advantages(torch.ones(4),1)
    loss,stats=impl.grpo_loss(*loss_inputs())
    assert loss.shape==(),f'loss expected scalar, got {loss.shape}'
    for key in ('kl_mean','clipfrac','ratio_mean'):
        assert key in stats,f'stats expected {key}, got {list(stats)}'


@pytest.mark.t1
def test_reference(impl,ref):
    """Group std and per-response aggregation are pinned to the declared reference convention."""
    r=torch.randn(24,dtype=torch.float64)
    close(impl.grpo_advantages(r,8),ref.grpo_advantages(r,8))
    args=loss_inputs();got,gs=impl.grpo_loss(*args);expected,es=ref.grpo_loss(*args)
    close(got,expected)
    for key in es:close(gs[key],es[key])


@pytest.mark.t2
def test_group_advantage_invariants(impl):
    """Group centering removes offsets and sample std removes positive scale when epsilon is zero."""
    r=torch.tensor([1.,2.,4.,8.,20.,21.,22.,25.],dtype=torch.float64)
    a=impl.grpo_advantages(r,4,eps=0)
    close(a.reshape(-1,4).sum(-1),torch.zeros(2,dtype=r.dtype),atol=1e-10)
    close(impl.grpo_advantages(r*3+8,4,eps=0),a)
    close(impl.grpo_advantages(torch.ones(8),4),torch.zeros(8))
    close(impl.grpo_advantages(r,4,scale_std=False),r-r.reshape(-1,4).mean(-1).repeat_interleave(4))
    shifted=r.clone();shifted[4:]*=100
    close(impl.grpo_advantages(shifted,4)[:4],impl.grpo_advantages(r,4)[:4])


@pytest.mark.t2
def test_on_policy_token_gradient(impl):
    """At ratio one each effective token receives advantage divided by response length and batch size."""
    logp,_,ref,adv,mask=loss_inputs();logp.requires_grad_()
    loss,stats=impl.grpo_loss(logp,logp.detach(),ref,adv,mask,beta=0)
    loss.backward();expected=-adv[:,None]*mask/(mask.sum(-1)[:,None]*4)
    close(logp.grad,expected)
    close(stats['clipfrac'],torch.tensor(0.,dtype=logp.dtype))


@pytest.mark.t2
def test_k3_nonnegative(impl):
    """The exponential tangent inequality makes the k3 KL estimator nonnegative pointwise."""
    x=torch.full((2,3),-.5);mask=torch.ones_like(x);adv=torch.zeros(2)
    loss,stats=impl.grpo_loss(x,x,torch.zeros_like(x),adv,mask)
    assert loss>=0,f'KL-only loss expected >=0, got {loss}'
    assert stats['kl_mean']>=0,f'kl_mean expected >=0, got {stats["kl_mean"]}'
    _,equal=impl.grpo_loss(x,x,x,adv,mask)
    close(equal['kl_mean'],torch.tensor(0.))


@pytest.mark.t2
def test_padding_isolation(impl):
    """Padding carries no reward, KL contribution, or gradient, even with extreme logits."""
    logp,old,ref,adv,mask=loss_inputs()
    expected,_=impl.grpo_loss(logp,old,ref,adv,mask)
    changed=[]
    for i,x in enumerate((logp,old,ref)):
        y=x.clone();y[mask==0]=(-1)**i*1000;changed.append(y)
    changed[0].requires_grad_()
    got,_=impl.grpo_loss(*changed,adv,mask);got.backward()
    close(got,expected)
    close(changed[0].grad[mask==0],torch.zeros_like(changed[0].grad[mask==0]))


@pytest.mark.t2
def test_snapshots_no_gradient(impl):
    """Behavior and reference probabilities are constants for policy optimization."""
    logp,old,ref,adv,mask=loss_inputs()
    for x in (logp,old,ref):x.requires_grad_()
    loss,_=impl.grpo_loss(logp,old,ref,adv,mask);loss.backward()
    for name,x in [('old_logp',old),('ref_logp',ref)]:
        assert x.grad is None or torch.all(x.grad==0),f'{name} expected zero/no gradient, got {x.grad}'


@pytest.mark.t2
def test_response_duplication_invariance(impl):
    """Duplicating only one response's tokens leaves its within-response average unchanged."""
    logp,old,ref,adv,mask=loss_inputs()
    expected,_=impl.grpo_loss(logp,old,ref,adv,mask)
    arrays=[]
    for x in (logp,old,ref,mask):
        y=torch.zeros(4,14,dtype=x.dtype);y[:,:7]=x;y[0,7:]=x[0];arrays.append(y)
    got,_=impl.grpo_loss(*arrays[:3],adv,arrays[3])
    close(got,expected)


@pytest.mark.t3
def test_toy_language_policy_learns(impl,artifact):
    """Within-prompt relative rewards improve an EOS-aware categorical language policy."""
    out=run(impl.grpo_advantages,impl.grpo_loss);artifact('reward_training',out)
    assert out['final_reward']>=.9,f'step 150 expected average reward >=0.9, got {out["final_reward"]}'
