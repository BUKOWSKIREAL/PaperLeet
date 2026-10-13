import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.rl.ppo_clip.experiment import run


def call(impl, lp, adv, **kw):
    zero=torch.zeros_like(lp)
    return impl.ppo_loss(lp,zero,adv,zero,zero,zero,norm_adv=False,**kw)


@pytest.mark.t0
def test_core_function_contract(impl):
    """The required core accepts three minibatch vectors and returns the minimized scalar loss."""
    old=torch.tensor([-.5,-1.,-2.],dtype=torch.float64)
    advantage=torch.tensor([1.,-2.,4.],dtype=old.dtype)
    got=impl.clipped_surrogate(old,old,advantage)
    assert got.shape==(),f'core loss expected scalar, got {got.shape}'
    assert got.dtype==old.dtype,f'core dtype expected {old.dtype}, got {got.dtype}'
    close(got,-advantage.mean())


@pytest.mark.t2
def test_core_four_quadrants_and_frozen_targets(impl):
    """The core alone implements clipping gradients and freezes old policy and advantage targets."""
    ratio=torch.tensor([1.5,.5,.5,1.5],dtype=torch.float64)
    new=ratio.log().requires_grad_()
    old=torch.zeros_like(new,requires_grad=True)
    advantage=torch.tensor([1.,1.,-1.,-1.],dtype=new.dtype,requires_grad=True)
    impl.clipped_surrogate(new,old,advantage,.2).backward()
    close(new.grad,torch.tensor([0.,-.125,0.,.375],dtype=new.dtype))
    for name,x in [('old_logp',old),('advantages',advantage)]:
        assert x.grad is None or torch.all(x.grad==0),f'{name} expected frozen target, got {x.grad}'


@pytest.mark.t0
def test_scalar_contract(impl):
    """The objective and every diagnostic aggregate a minibatch into one scalar."""
    out=call(impl,torch.zeros(4),torch.ones(4))
    required={'loss','pg_loss','v_loss','entropy','approx_kl','clipfrac'}
    assert required <= out.keys(), f'keys expected {required}, got {set(out)}'
    for key,x in out.items():
        assert x.shape == (),f'{key} expected scalar shape (), got {x.shape}'
        assert torch.isfinite(x),f'{key} expected finite, got {x}'


@pytest.mark.t1
def test_reference(impl,ref):
    """Joint policy, clipped value and entropy terms follow one consistent loss convention."""
    args=[torch.randn(13,dtype=torch.float64) for _ in range(7)]
    for vf in (None,.2):
        got=impl.ppo_loss(*args,vf_clip=vf,ent_coef=.07)
        exp=ref.ppo_loss(*args,vf_clip=vf,ent_coef=.07)
        for key in exp: close(got[key],exp[key])


@pytest.mark.t2
def test_on_policy_gradient(impl):
    """At ratio one the clipped surrogate equals the vanilla policy gradient."""
    lp=torch.zeros(4,requires_grad=True); adv=torch.tensor([1.,-2.,3.,-.5])
    out=call(impl,lp,adv); out['pg_loss'].backward()
    close(lp.grad,-adv/4)


@pytest.mark.t2
def test_four_quadrant_gradient(impl):
    """Clipping stops only improvements beyond the trust interval, keeping corrective gradients."""
    ratio=torch.tensor([1.5,.5,.5,1.5],dtype=torch.float64)
    lp=ratio.log().requires_grad_(); adv=torch.tensor([1.,1.,-1.,-1.],dtype=lp.dtype)
    call(impl,lp,adv)['pg_loss'].backward()
    close(lp.grad,torch.tensor([0.,-.125,0.,.375],dtype=lp.dtype))


@pytest.mark.t2
def test_old_values_are_constants(impl):
    """A behavior policy and old critic are snapshots and cannot receive gradients."""
    new=torch.tensor([.1,-.2],requires_grad=True); old=torch.zeros(2,requires_grad=True)
    value=torch.ones(2,requires_grad=True); old_value=torch.zeros(2,requires_grad=True)
    out=impl.ppo_loss(new,old,torch.tensor([1.,-1.]),value,old_value,torch.ones(2)*2,norm_adv=False)
    out['loss'].backward()
    for key,x in [('old_logp',old),('old_values',old_value)]:
        assert x.grad is None or torch.all(x.grad==0),f'{key} expected no gradient, got {x.grad}'


@pytest.mark.t2
def test_diagnostics(impl):
    """The k3 estimate is nonnegative and clip fraction counts both tails of the ratio."""
    lp=torch.tensor([-.7,0.,.7],requires_grad=True)
    out=call(impl,lp,torch.ones(3))
    assert out['approx_kl']>=0,f'approx_kl expected >=0, got {out["approx_kl"]}'
    close(out['clipfrac'],torch.tensor(2/3))
    close(call(impl,torch.zeros(3),torch.ones(3))['approx_kl'],torch.tensor(0.))
    for key in ('approx_kl','clipfrac'):
        assert not out[key].requires_grad,f'{key} expected detached, got requires_grad=True'


@pytest.mark.t2
def test_value_loss_lower_bound(impl):
    """The pessimistic maximum cannot reduce the unconstrained squared value error."""
    z=torch.zeros(3); v=torch.tensor([1.,2.,3.]); target=v.clone()
    clipped=impl.ppo_loss(z,z,z,v,z,target,norm_adv=False)['v_loss']
    expected=.5*((torch.full_like(v,.2)-target)**2).mean()
    close(clipped,expected)
    assert clipped>0,f'clipped value expected >0, got {clipped}'
    v=torch.tensor([.1,-.1,.05])
    a=impl.ppo_loss(z,z,z,v,z,target,norm_adv=False)['v_loss']
    b=impl.ppo_loss(z,z,z,v,z,target,norm_adv=False,vf_clip=None)['v_loss']
    close(a,b)


@pytest.mark.t2
def test_advantage_affine_invariance(impl):
    """Subtracting mean and dividing sample std removes a positive affine transformation."""
    z=torch.zeros(9,dtype=torch.float64); lp=torch.randn_like(z)*.1; a=torch.randn_like(z)
    first=impl.ppo_loss(lp,z,a,z,z,z)['pg_loss']
    second=impl.ppo_loss(lp,z,a*3.2+4,z,z,z)['pg_loss']
    close(first,second,atol=1e-7,rtol=1e-6)


@pytest.mark.t3
def test_bandit_learning_and_kl(impl,artifact):
    """Repeated clipped updates learn the best arm while limiting movement on a fixed batch."""
    out=run(impl.ppo_loss); artifact('bandit',out)
    assert out['best_prob']>.9,f'best arm after 30 rounds expected probability>0.9, got {out["best_prob"]}'
    threshold = 0.13986029103398323
    assert out['max_kl']<threshold,f'20-epoch batch expected KL<{threshold}, got {out["max_kl"]}'
