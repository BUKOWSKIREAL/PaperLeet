import pytest
import torch
from paperleet.common.asserts import assert_close_located as close


@pytest.mark.t0
def test_contract(impl):
    """调度使用 float64，前向加噪保持输入的形状与 dtype。"""
    for name in ('linear_betas','cosine_betas'):
        betas = getattr(impl,name)(1000)
        for key,value in impl.make_schedule(betas).items():
            assert value.shape == (1000,), f'{key}.shape: expected (1000,), got {value.shape}'
            assert value.dtype == torch.float64, f'{key}.dtype: expected float64, got {value.dtype}'
    for shape in [(4,3),(4,2,5,5)]:
        x = torch.randn(shape)
        y = impl.q_sample(x,torch.arange(4),torch.randn_like(x),torch.linspace(.99,.1,4))
        assert y.shape == x.shape and y.dtype == x.dtype, f'output contract: expected {x.shape}/{x.dtype}, got {y.shape}/{y.dtype}'


@pytest.mark.t1
def test_reference(impl,ref):
    """离散时间的下标和 cosine 比值约定必须精确一致。"""
    for name in ('linear_betas','cosine_betas'):
        got,exp = getattr(impl,name)(1000),getattr(ref,name)(1000)
        close(got,exp,rtol=1e-12,atol=1e-12)
        for key,value in ref.make_schedule(exp).items():
            close(impl.make_schedule(got)[key],value,rtol=1e-10,atol=1e-10)


@pytest.mark.t2
def test_monotone(impl):
    """每步保留的信息都会减少，累计 alpha 与 log-SNR 严格下降。"""
    for name in ('linear_betas','cosine_betas'):
        schedule = impl.make_schedule(getattr(impl,name)(1000))
        for key in ('alpha_bars','log_snr'):
            x = schedule[key]
            assert (x[1:]<x[:-1]).all(), f'{name}/{key}: expected strictly decreasing, got min decrement {(x[:-1]-x[1:]).min().item()}'
        x = schedule['alpha_bars']
        assert ((x>0)&(x<1)).all(), f'{name}/alpha_bars: expected (0,1), got [{x.min()}, {x.max()}]'


@pytest.mark.t2
def test_linear_endpoints(impl):
    """原始 DDPM 的线性调度终端累计 alpha 约为 4e-5 而不是零。"""
    b = impl.linear_betas(1000)
    close(b[[0,-1]],torch.tensor([1e-4,.02],dtype=torch.float64),rtol=0,atol=1e-12)
    close(impl.make_schedule(b)['alpha_bars'][-1],torch.tensor(4e-5,dtype=torch.float64),rtol=.05,atol=0)


@pytest.mark.t2
def test_cosine_clip(impl):
    """cosine 的最后一步需要截断以免累计信号彻底消失。"""
    b = impl.cosine_betas(1000)
    assert b.max() <= .999, f'beta.max: expected <= .999, got {b.max().item()}'
    close(impl.make_schedule(b)['alpha_bars'][500],torch.tensor(.5,dtype=torch.float64),rtol=0,atol=.01)


@pytest.mark.t2
def test_signal_and_noise(impl):
    """关闭信号或噪声的一项后，闭式表达式只剩另一项。"""
    x,n = torch.randn(4,3),torch.randn(4,3)
    ab = torch.tensor([.99,.8,.3,.01]); t=torch.arange(4)
    close(impl.q_sample(x,t,torch.zeros_like(x),ab),ab[:,None].sqrt()*x,rtol=1e-6,atol=1e-6)
    close(impl.q_sample(torch.zeros_like(x),t,n,ab),(1-ab[:,None]).sqrt()*n,rtol=1e-6,atol=1e-6)


@pytest.mark.t2
def test_per_sample_time(impl):
    """一个 batch 的每行都有自己的时间索引。"""
    x,n = torch.randn(5,2,3),torch.randn(5,2,3)
    ab=torch.linspace(.99,.01,17); t=torch.tensor([0,12,5,16,8])
    got=impl.q_sample(x,t,n,ab)
    for i in range(5):
        close(got[i],ab[t[i]].sqrt()*x[i]+(1-ab[t[i]]).sqrt()*n[i],rtol=1e-6,atol=1e-6)


@pytest.mark.t2
def test_variance_preserving(impl,artifact):
    """独立单位高斯的线性组合，在系数平方和为一时仍有单位方差。"""
    x,n = torch.randn(100000,1),torch.randn(100000,1)
    ab = torch.linspace(.999,.001,11)
    variances=[]
    for j in range(11):
        y=impl.q_sample(x,torch.full((len(x),),j),n,ab)
        variances.append(y.var())
    got=torch.stack(variances)
    artifact('variances',got)
    close(got,torch.ones_like(got),rtol=0,atol=.02)
