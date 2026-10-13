import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.rl.dr_grpo.experiment import run


@pytest.mark.t0
def test_shape(impl):
    """Centering preserves reward shape and fixed normalization yields a scalar loss."""
    adv=impl.drgrpo_advantages(torch.tensor([1.,2.,3.,4.]),2)
    assert adv.shape==(4,),f'advantage expected (4,), got {adv.shape}'
    z=torch.zeros(4,6);loss,stats=impl.drgrpo_loss(z,z,z,adv,torch.ones_like(z),8)
    assert loss.shape==(),f'loss expected (), got {loss.shape}'
    for key in ('kl_mean','clipfrac','ratio_mean'):
        assert key in stats,f'stats expected {key}, got {list(stats)}'


@pytest.mark.t1
def test_reference(impl,ref):
    """The variant changes only group scaling and loss normalization from the predecessor."""
    reward=torch.randn(24,dtype=torch.float64)
    close(impl.drgrpo_advantages(reward,8),ref.drgrpo_advantages(reward,8))
    args=[torch.randn(4,9,dtype=torch.float64)*.2 for _ in range(3)]
    adv=torch.randn(4,dtype=torch.float64);mask=(torch.rand(4,9)>.3).double()
    got,gs=impl.drgrpo_loss(*args,adv,mask,12);expected,es=ref.drgrpo_loss(*args,adv,mask,12)
    close(got,expected)
    for key in es:close(gs[key],es[key])


@pytest.mark.t2
def test_length_bias_removed(impl,ref,artifact):
    """A constant denominator gives equally negative responses identical per-token gradients."""
    lp=torch.zeros(2,20,requires_grad=True);mask=(torch.arange(20)[None,:]<torch.tensor([5,20])[:,None]).float()
    adv=-torch.ones(2)
    loss,_=impl.drgrpo_loss(lp,lp.detach(),lp.detach(),adv,mask,20,beta=0)
    loss.backward();ratio=lp.grad[1,0]/lp.grad[0,0]
    baseline=torch.zeros(2,20,requires_grad=True)
    rloss,_=ref.grpo_reference_loss(baseline,baseline.detach(),baseline.detach(),adv,mask,beta=0)
    rloss.backward();grpo_ratio=baseline.grad[1,0]/baseline.grad[0,0]
    artifact('length_bias',{'grpo_ratio':float(grpo_ratio),'dr_grpo_ratio':float(ratio)})
    close(grpo_ratio,torch.tensor(5/20))
    assert torch.isclose(ratio,torch.tensor(1.)),f'token [1,0]/[0,0] expected Dr.GRPO=1, GRPO comparison={float(grpo_ratio)} (5/20); got {float(ratio)}'


@pytest.mark.t2
def test_fixed_generation_limit(impl):
    """Changing an unrelated zero-advantage response cannot change the constant denominator."""
    grads=[]
    for other_length in (10,30):
        lp=torch.zeros(2,30,requires_grad=True)
        mask=(torch.arange(30)[None,:]<torch.tensor([5,other_length])[:,None]).float()
        loss,_=impl.drgrpo_loss(lp,lp.detach(),lp.detach(),torch.tensor([-1.,0.]),mask,40,beta=0)
        loss.backward();grads.append(lp.grad[0,:5])
    close(grads[0],grads[1]);close(grads[0],torch.full((5,),1/80))


@pytest.mark.t2
def test_advantage_scales_with_reward(impl):
    """Removing std scaling preserves reward units, so multiplying rewards multiplies advantages."""
    r=torch.tensor([1.,2.,4.,8.,-3.,1.,2.,10.],dtype=torch.float64)
    adv=impl.drgrpo_advantages(r,4)
    close(adv.reshape(-1,4).sum(-1),torch.zeros(2,dtype=r.dtype))
    close(impl.drgrpo_advantages(r*3.7,4),adv*3.7)


@pytest.mark.t3
def test_length_effect_display_only(impl,ref,artifact):
    """Empirical wrong-answer lengths are displayed without grading their trend on one seed."""
    out=run(impl.drgrpo_advantages,impl.drgrpo_loss,ref.grpo_reference_advantages,ref.grpo_reference_loss)
    artifact('length_training',out)
