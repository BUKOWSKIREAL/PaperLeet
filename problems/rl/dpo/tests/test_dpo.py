import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.rl.dpo.experiment import run


@pytest.mark.t0
def test_shapes_single_batch(impl):
    """One sequence yields one score and one preference pair yields a scalar loss."""
    lp=impl.sequence_logprob(torch.randn(1,5,7),torch.randint(0,7,(1,5)),torch.ones(1,5))
    assert lp.shape==(1,),f'sequence scores expected (1,), got {lp.shape}'
    out=impl.dpo_loss(lp,lp,lp,lp)
    assert out[0].shape==(),f'loss expected scalar, got {out[0].shape}'
    for x in out[1:]: assert x.shape==(1,),f'rewards expected (1,), got {x.shape}'


@pytest.mark.t1
def test_reference(impl,ref):
    """Shifted sequence sums and the smoothed preference loss follow the frozen specification."""
    args=(torch.randn(3,8,11,dtype=torch.float64),torch.randint(0,11,(3,8)),torch.randint(0,2,(3,8)))
    close(impl.sequence_logprob(*args),ref.sequence_logprob(*args))
    p=[torch.randn(7,dtype=torch.float64) for _ in range(4)]
    for got,exp in zip(impl.dpo_loss(*p,label_smoothing=.2),ref.dpo_loss(*p,label_smoothing=.2)):close(got,exp)


@pytest.mark.t2
def test_sequence_double_loop(impl):
    """Each answer token is predicted by the preceding position and contributes a sum term."""
    logits=torch.randn(2,6,8,dtype=torch.float64); ids=torch.randint(0,8,(2,6))
    mask=torch.tensor([[0,0,1,1,1,0],[0,1,1,0,1,1]])
    exp=torch.zeros(2,dtype=logits.dtype)
    for b in range(2):
        for t in range(1,6):exp[b]+=logits[b,t-1].log_softmax(0)[ids[b,t]]*mask[b,t]
    close(impl.sequence_logprob(logits,ids,mask),exp)


@pytest.mark.t2
def test_shift_and_mask_isolation(impl):
    """The final logits and logits predicting masked labels cannot affect sequence probability."""
    logits=torch.randn(2,5,6); ids=torch.randint(0,6,(2,5));mask=torch.tensor([[0,0,1,1,0],[0,0,1,1,0]])
    exp=impl.sequence_logprob(logits,ids,mask)
    changed=logits.clone();changed[:,[0,3,4]]=torch.randn(2,3,6)*100
    close(impl.sequence_logprob(changed,ids,mask),exp)


@pytest.mark.t2
def test_policy_equals_reference(impl):
    """Identical policy and reference have zero preference margin and loss log two."""
    a,b=torch.randn(7),torch.randn(7)
    close(impl.dpo_loss(a,b,a,b)[0],torch.tensor(math.log(2)))


@pytest.mark.t2
def test_beta_margin_scaling(impl):
    """DPO depends on the product beta times the reference-adjusted margin."""
    a,b=torch.randn(7),torch.randn(7);zero=torch.zeros(7)
    close(impl.dpo_loss(a,b,zero,zero,beta=.3)[0],impl.dpo_loss(a/4,b/4,zero,zero,beta=1.2)[0])


@pytest.mark.t2
def test_extreme_margin_stability(impl):
    """Log-sigmoid remains finite even when the sigmoid itself underflows to zero."""
    chosen=torch.tensor([-1e4,1e4],requires_grad=True);zero=torch.zeros(2)
    loss,*_=impl.dpo_loss(chosen,zero,zero,zero,beta=1)
    loss.backward()
    assert torch.isfinite(loss),f'margins +/-10000 expected finite loss, got {loss}'
    close(chosen.grad,torch.tensor([-.5,0.]))


@pytest.mark.t2
def test_gradient_direction_and_rewards(impl):
    """Minimization promotes chosen responses and returned implicit rewards are detached snapshots."""
    a=torch.tensor([.2,.8],requires_grad=True);b=torch.tensor([.1,-.3],requires_grad=True)
    rc=torch.tensor([.4,.6],requires_grad=True);rr=torch.tensor([-.1,.3],requires_grad=True)
    loss,cr,rrw=impl.dpo_loss(a,b,rc,rr,beta=.7)
    loss.backward()
    assert (a.grad<0).all(),f'chosen indices expected negative gradients, got {a.grad}'
    assert (b.grad>0).all(),f'rejected indices expected positive gradients, got {b.grad}'
    for name,got,exp in [('chosen',cr,.7*(a-rc)),('rejected',rrw,.7*(b-rr))]:
        close(got,exp)
        assert not got.requires_grad,f'{name} reward expected detached, got requires_grad=True'
    for name,x in [('ref_chosen',rc),('ref_rejected',rr)]:
        assert x.grad is None or torch.all(x.grad==0),f'{name} expected no gradient, got {x.grad}'


@pytest.mark.t3
def test_closed_form_preference_optimum(impl,artifact):
    """Bradley–Terry preferences recover reward ranking and the KL-regularized closed-form policy."""
    out=run(impl.dpo_loss);artifact('preference_training',out)
    assert out['rank_order']==list(range(6)),f'reward rank expected [0,1,2,3,4,5], got {out["rank_order"]}'
    threshold = 0.0006743129342794418
    assert out['kl']<threshold,f'KL(policy || closed form) expected <{threshold}, got {out["kl"]}'
