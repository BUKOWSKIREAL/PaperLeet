from pathlib import Path
import pytest
import torch
import yaml
from paperleet.common.asserts import assert_close_located as close
from paperleet.common.toydata import GaussianMixture
from paperleet.common.oracles import gmm_eps_oracle
from paperleet.common.metrics import mmd_rbf,mode_coverage


def schedule(T=1000):
    b=torch.linspace(1e-4,.02,T,dtype=torch.float64); ab=(1-b).cumprod(0)
    return b,ab,torch.cat((torch.ones(1,dtype=b.dtype),ab[:-1]))


def mixture():
    angle=torch.arange(8,dtype=torch.float64)*torch.pi/4
    return GaussianMixture(2*torch.stack((angle.cos(),angle.sin()),-1),.15)


@pytest.mark.t0
def test_batch_shape(impl):
    """后验支持同一 batch 中不同的时间点与任意空间维度。"""
    b,ab,p=schedule(); x=torch.randn(3,2,4,dtype=torch.float64); t=torch.tensor([0,50,999])
    mean,var=impl.q_posterior(x,x,t,b,ab,p)
    assert mean.shape==x.shape, f'mean.shape: expected {x.shape}, got {mean.shape}'
    assert torch.broadcast_shapes(var.shape,x.shape)==x.shape,f'var.shape: expected broadcast to {x.shape}, got {var.shape}'


@pytest.mark.t1
def test_reference(impl,ref):
    """固定噪声后，后验与两种方差的更新应唯一确定。"""
    b,ab,p=schedule(); x,x0,n=torch.randn(3,5,2,dtype=torch.float64);t=torch.tensor([0,1,50,200,999])
    for got,exp in zip(impl.q_posterior(x0,x,t,b,ab,p),ref.q_posterior(x0,x,t,b,ab,p)):
        close(got,exp,rtol=1e-10,atol=1e-10)
    for mode in ('posterior','beta'):
        close(impl.p_sample_step(x0,x,t,b,ab,p,n,mode),ref.p_sample_step(x0,x,t,b,ab,p,n,mode),rtol=1e-10,atol=1e-10)


@pytest.mark.t2
def test_precision_product(impl):
    """两个高斯相乘时精度相加，独立推导得到同一个后验。"""
    b,ab,p=schedule(); t=torch.tensor([1,10,300,999]); x,x0=torch.randn(2,4,3,dtype=torch.float64)
    prior_mean=p[t,None].sqrt()*x0; prior_var=1-p[t,None]
    likelihood_mean=x/(1-b[t,None]).sqrt(); likelihood_var=b[t,None]/(1-b[t,None])
    exp_var=1/(1/prior_var+1/likelihood_var)
    exp_mean=exp_var*(prior_mean/prior_var+likelihood_mean/likelihood_var)
    mean,var=impl.q_posterior(x0,x,t,b,ab,p)
    close(mean,exp_mean,rtol=1e-10,atol=1e-10);close(var,exp_var,rtol=1e-10,atol=1e-10)


@pytest.mark.t2
def test_posterior_bounds(impl):
    """观测 x0 会减少条件方差，且最后一步已完全确定 x0。"""
    b,ab,p=schedule(); x,x0=torch.randn(2,1000,2,dtype=torch.float64);t=torch.arange(1000)
    mean,var=impl.q_posterior(x0,x,t,b,ab,p)
    assert (var.flatten() <= b+1e-12).all(), f'variance bound: expected <= beta, got excess {(var.flatten()-b).max()}'
    close(var[0],torch.zeros_like(var[0]),rtol=0,atol=0)
    close(mean[0],x0[0],rtol=1e-10,atol=1e-10)


@pytest.mark.t2
def test_last_step_no_noise(impl):
    """最终输出应为后验均值，beta 方差选项也不能再注入噪声。"""
    b,ab,p=schedule();x,eps=torch.randn(2,3,2,dtype=torch.float64);t=torch.zeros(3,dtype=torch.long)
    for mode in ('posterior','beta'):
        close(impl.p_sample_step(eps,x,t,b,ab,p,torch.zeros_like(x),mode),impl.p_sample_step(eps,x,t,b,ab,p,torch.ones_like(x),mode),rtol=0,atol=0)


@pytest.mark.t2
def test_point_oracle(impl):
    """近单点数据的精确噪声预测应恢复该点并接入正确后验均值。"""
    b,ab,p=schedule(); mu=torch.tensor([[1.3,-.7]],dtype=torch.float64);g=GaussianMixture(mu,1e-3)
    x=torch.randn(4,2,dtype=torch.float64);t=torch.tensor([100,300,700,999]);eps=gmm_eps_oracle(g,ab)(x,t)
    got=impl.p_sample_step(eps,x,t,b,ab,p,torch.zeros_like(x))
    expected,_=impl.q_posterior(mu.expand_as(x),x,t,b,ab,p)
    close(got,expected,rtol=0,atol=1e-5)


@pytest.mark.t2
def test_beta_variance(impl):
    """beta 方差时两次更新的差应恰为 sqrt(beta_t) 乘给定噪声。"""
    b,ab,p=schedule();x,eps,n=torch.randn(3,4,2,dtype=torch.float64);t=torch.tensor([1,50,500,999])
    base=impl.p_sample_step(eps,x,t,b,ab,p,torch.zeros_like(n),'beta')
    close(impl.p_sample_step(eps,x,t,b,ab,p,n,'beta')-base,b[t,None].sqrt()*n,rtol=1e-9,atol=1e-9)


@pytest.mark.t2
def test_sampler_time_order(impl):
    """祖先采样从最噪时间开始并恰好调用每个离散时间一次。"""
    calls=[]
    def model(x,t):
        calls.append(int(t[0]));return torch.zeros_like(x)
    impl.ddpm_sample(model,(2,2),schedule(8)[0],torch.Generator().manual_seed(42))
    assert calls==list(reversed(range(8))),f'time indices: expected {list(reversed(range(8)))}, got {calls}'


@pytest.mark.t3
@pytest.mark.parametrize('var_type',['posterior','beta'])
def test_gmm_sampling(impl,artifact,var_type):
    """完美噪声预测器下，反向采样应恢复全部八个高斯模式及其分布。"""
    b,ab,_=schedule();g=mixture();generator=torch.Generator().manual_seed(42)
    samples=impl.ddpm_sample(gmm_eps_oracle(g,ab),(2000,2),b,generator,var_type)
    real=g.sample(2000,torch.Generator().manual_seed(123))
    metric=mmd_rbf(samples,real); coverage=mode_coverage(samples,g.means,.5)
    metadata=yaml.safe_load((Path(__file__).parents[1]/'problem.yaml').read_text())
    threshold=metadata['calibration']['mmd_thresholds'][var_type]
    artifact('samples',samples);artifact('reference_samples',real);artifact('mmd',metric);artifact('threshold',threshold);artifact('coverage',coverage)
    assert metric<threshold,f'{var_type}/MMD²: expected < {threshold}, got {metric}'
    for i,p in enumerate(coverage['proportions']):
        assert abs(p-.125)<.03,f'mode[{i}] proportion: expected 0.125 ± 0.03, got {p}'
    assert coverage['outlier_fraction']<.05,f'outlier fraction: expected < .05, got {coverage["outlier_fraction"]}'
