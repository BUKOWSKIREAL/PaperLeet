from itertools import combinations, product
import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.rl.pkpo.experiment import run


def oracle(values,k):
    subsets=list(combinations(range(len(values)),k)); result=[0.]*len(values); objective=0.
    for subset in subsets:
        best=max(values[j] for j in subset); objective+=best/len(subsets)
        for i in subset:
            baseline=max(values[j] for j in subset if j!=i) if k>1 else 0.
            result[i]+=(best-baseline)/len(subsets)
    return objective,result


@pytest.mark.t0
def test_shapes_dtype_input_preservation(impl):
    for dtype in (torch.float32,torch.float64):
        rewards=torch.tensor([[.2,.5,.1,.9],[.1,.3,.7,.2]],dtype=dtype).T
        before=rewards.clone()
        out=impl.pkpo_weights(rewards,2)
        assert out.shape==rewards.shape and out.dtype==dtype and out.device==rewards.device
        assert impl.maxk_estimate(rewards,2).shape==(4,)
        assert impl.pkpo_loss(rewards,rewards,2).shape==()
        close(rewards,before)


@pytest.mark.t0
def test_supplied_validation(impl):
    rewards=torch.ones(4)
    for k in (0,5,1.5):
        with pytest.raises(ValueError):impl.maxk_estimate(rewards,k)
        with pytest.raises(ValueError):impl.pkpo_weights(rewards,k)
    with pytest.raises(ValueError):impl.pkpo_loss(rewards,torch.ones(3),2)


@pytest.mark.t1
def test_subset_oracle_real_rewards_all_k(impl):
    for values in ([.8,-.3,.1,1.2],[1.,0.,1.,0.],[-2.,-2.,-3.,.1]):
        rewards=torch.tensor(values,dtype=torch.float64)
        for k in range(1,5):
            metric,weights=oracle(values,k)
            close(impl.maxk_estimate(rewards,k),torch.tensor(metric,dtype=rewards.dtype),atol=1e-12,rtol=1e-12)
            close(impl.pkpo_weights(rewards,k),torch.tensor(weights,dtype=rewards.dtype),atol=1e-12,rtol=1e-12)


@pytest.mark.t1
def test_loss_uses_sum_of_normalized_weights(impl):
    rewards=torch.tensor([[.1,.9,.4],[1.,0.,0.]],dtype=torch.float64)
    logp=torch.tensor([[-1.,-2.,-3.],[-.5,-1.,-1.2]],dtype=rewards.dtype,requires_grad=True)
    weights=torch.tensor([oracle(v,2)[1] for v in rewards.tolist()],dtype=rewards.dtype)
    expected=-sum(float(logp[i,j])*float(weights[i,j]) for i in range(2) for j in range(3))/2
    out=impl.pkpo_loss(logp,rewards,2)
    close(out,torch.tensor(expected,dtype=rewards.dtype))
    close(torch.autograd.grad(out,logp)[0],-weights/2)


@pytest.mark.t2
def test_exact_binary_policy_gradient_is_pass_at_k(impl):
    # Enumerate all 2^N sample batches under Bernoulli(p), independently of the estimator.
    n=4;p=.31
    for k in (1,2,4):
        expected_gradient=0.
        for actions in product((0.,1.),repeat=n):
            c=sum(actions); probability=p**c*(1-p)**(n-c)
            reward=torch.tensor(actions,dtype=torch.float64)
            score=(reward-p)/(p*(1-p))
            expected_gradient+=probability*float((impl.pkpo_weights(reward,k)*score).sum())
        assert abs(expected_gradient-k*(1-p)**(k-1))<1e-11


@pytest.mark.t2
def test_k_equals_n_and_ties(impl):
    for values,expected in (([.2,.9,.4],[0.,.5,0.]),([.9,.9,.4],[0.,0.,0.]),([-.2,-.8,-.5],[.3,0.,0.])):
        reward=torch.tensor(values,dtype=torch.float64)
        close(impl.pkpo_weights(reward,3),torch.tensor(expected,dtype=reward.dtype),atol=1e-12,rtol=1e-12)
    for k in (2,4):close(impl.pkpo_weights(torch.ones(4),k),torch.zeros(4))


@pytest.mark.t2
def test_permutation_translation_and_frozen_rewards(impl):
    rewards=torch.tensor([.6,.2,1.,-.5],dtype=torch.float64,requires_grad=True)
    order=torch.tensor([3,0,2,1])
    for k in (1,2,4):
        weights=impl.pkpo_weights(rewards,k)
        assert not weights.requires_grad
        close(impl.pkpo_weights(rewards[order],k),weights[order])
        if k>1:close(impl.pkpo_weights(rewards+7,k),weights,atol=1e-12,rtol=1e-12)
        close(impl.maxk_estimate(rewards+7,k),impl.maxk_estimate(rewards,k)+7)
    logp=torch.tensor([-.8,-.4,-2.,-1.],dtype=rewards.dtype,requires_grad=True)
    grads=torch.autograd.grad(impl.pkpo_loss(logp,rewards,2),(logp,rewards),allow_unused=True)
    assert grads[1] is None or grads[1].abs().max()==0


@pytest.mark.t2
def test_large_group_stability_and_singleton(impl):
    # Large binomial counts require ratios, not fixed-width integer tensors.
    rewards=torch.linspace(-1,2,256,dtype=torch.float64)
    for k in (1,64,128,256):
        w=impl.pkpo_weights(rewards,k); value=impl.maxk_estimate(rewards,k)
        assert torch.isfinite(w).all() and torch.isfinite(value)
        assert -1 <= value <= 2+1e-12
        if k>1:assert w.min()>-1e-12
    close(impl.pkpo_weights(torch.tensor([.7]),1),torch.tensor([.7]))


@pytest.mark.t3
def test_native_gaussian_toy(impl,artifact):
    result=run(impl.maxk_estimate,impl.pkpo_weights,impl.pkpo_loss)
    artifact('gaussian_expected_maximum',result)
    metrics=result['metrics']
    assert metrics['gradient_identity_error']<1e-10
    assert .8 <= metrics['optimum_k1'] <= .9
    assert metrics['optimum_k8']>metrics['optimum_k1']
    assert metrics['maxk_gain_at_theta_1']>.35
