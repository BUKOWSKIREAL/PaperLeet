from pathlib import Path
import time
import pytest
import torch
import yaml
from paperleet.common.asserts import assert_close_located as close
from paperleet.common.toydata import GaussianMixture,eight_gaussians
from paperleet.common.oracles import point_velocity_oracle,gaussian_velocity_oracle
from paperleet.common.models import TimeMLP
from paperleet.common.metrics import mmd_rbf,mode_coverage


@pytest.mark.t0
def test_contract(impl):
    """时间向量 [B] 应自动广播到任意样本维度。"""
    for shape in ((4,2),(4,3,2,2)):
        x=torch.randn(shape);got=impl.interpolate(x,x+1,torch.rand(4))
        assert got.shape==x.shape,f'output.shape: expected {x.shape}, got {got.shape}'
    t=impl.sample_t(7,torch.Generator().manual_seed(42))
    assert t.shape==(7,),f't.shape: expected (7,), got {t.shape}'


@pytest.mark.t1
def test_reference(impl,ref):
    """确定性线性速度场固定了插值、损失和积分的数值约定。"""
    x0,x1=torch.randn(2,7,2,dtype=torch.float64);t=torch.rand(7,dtype=torch.float64)
    model=lambda x,t: .3*x+t[:,None]
    close(impl.interpolate(x0,x1,t),ref.interpolate(x0,x1,t),rtol=1e-10,atol=1e-10)
    close(impl.cfm_loss(model,x1,x0,t),ref.cfm_loss(model,x1,x0,t),rtol=1e-10,atol=1e-10)
    close(impl.euler_sample(model,x0,20),ref.euler_sample(model,x0,20),rtol=1e-10,atol=1e-10)


@pytest.mark.t2
def test_point_zero_loss(impl):
    """单点数据下，条件速度恰好等于解析最优速度，回归误差应为零。"""
    x0=torch.randn(100,2,dtype=torch.float64);mu=torch.tensor([1.3,-.7],dtype=torch.float64)
    t=torch.rand(100,dtype=torch.float64)*.99
    loss=impl.cfm_loss(point_velocity_oracle(mu),mu.expand_as(x0),x0,t)
    close(loss,torch.zeros_like(loss),rtol=0,atol=1e-10)
    close(impl.cfm_target(x0,mu.expand_as(x0)),mu-x0,rtol=0,atol=0)


@pytest.mark.t2
def test_point_one_step(impl):
    """单点目标沿直线路径运动，Euler 一步即可到达。"""
    x=torch.randn(100,2,dtype=torch.float64);mu=torch.tensor([1.3,-.7],dtype=torch.float64)
    close(impl.euler_sample(point_velocity_oracle(mu),x,1),mu.expand_as(x),rtol=0,atol=1e-10)


@pytest.mark.t2
def test_gaussian_transport(impl,artifact):
    """高斯解析速度场将标准噪声推到指定均值与标准差的数据分布。"""
    mu=torch.tensor([1.3,-.7]);std=.6;x=torch.randn(20000,2)
    got=impl.euler_sample(gaussian_velocity_oracle(mu,std),x,200)
    artifact('mean',got.mean(0));artifact('std',got.std(0))
    close(got.mean(0),mu,rtol=0,atol=.03)
    close(got.std(0),torch.full((2,),std),rtol=.03,atol=0)


@pytest.mark.t2
def test_no_singular_endpoint(impl):
    """Euler 只在每段左端点求速度，不能在 t=1 的奇点求值。"""
    calls=[]
    def model(x,t):
        calls.extend(t.tolist());return torch.zeros_like(x)
    impl.euler_sample(model,torch.ones(2,2),20)
    assert min(calls)>=0 and max(calls)<1,f'evaluation times: expected [0,1), got [{min(calls)}, {max(calls)}]'
    close(torch.tensor(calls[::2]),torch.arange(20)/20,rtol=1e-6,atol=1e-6)


@pytest.mark.t2
def test_endpoints_and_time_distribution(impl):
    """插值端点固定，logit-normal 时间采样在 logit 空间满足指定高斯参数。"""
    x0,x1=torch.randn(2,4,2)
    close(impl.interpolate(x0,x1,torch.zeros(4)),x0,rtol=0,atol=0)
    close(impl.interpolate(x0,x1,torch.ones(4)),x1,rtol=0,atol=0)
    for dist in ('uniform','logit_normal'):
        t=impl.sample_t(100000,torch.Generator().manual_seed(42),dist,m=.3,s=.7)
        assert ((t>0)&(t<1)).all(),f'{dist} bounds: expected (0,1), got [{t.min()}, {t.max()}]'
        if dist=='logit_normal':
            close(t.logit().mean(),torch.tensor(.3),rtol=0,atol=.01)
            close(t.logit().std(),torch.tensor(.7),rtol=0,atol=.01)


@pytest.mark.t3
def test_learns_eight_modes(impl,artifact):
    """训练速度场必须能从噪声积分到八高斯数据，损失下降本身不足以验证采样方向。"""
    config=yaml.safe_load((Path(__file__).parents[1]/'problem.yaml').read_text())['calibration']
    start=time.perf_counter();model=TimeMLP();optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    generator=torch.Generator().manual_seed(42);curve=[]
    for step in range(config['training_steps']):
        x1=eight_gaussians(256,generator=generator);x0=torch.randn(256,2,generator=generator)
        t=impl.sample_t(256,generator)
        optimizer.zero_grad(set_to_none=True);loss=impl.cfm_loss(model,x1,x0,t);loss.backward();optimizer.step()
        if step%25==0:
            curve.append([step,float(loss.detach())])
    samples=impl.euler_sample(model,torch.randn(2000,2,generator=torch.Generator().manual_seed(999)),50)
    real=eight_gaussians(2000,generator=torch.Generator().manual_seed(123))
    angle=torch.arange(8)*torch.pi/4;centers=2*torch.stack((angle.cos(),angle.sin()),-1)
    metric=mmd_rbf(samples,real);coverage=mode_coverage(samples,centers,.5);elapsed=time.perf_counter()-start
    artifact('samples',samples);artifact('reference_samples',real);artifact('loss_curve',curve);artifact('mmd',metric);artifact('threshold',config['mmd_threshold']);artifact('coverage',coverage);artifact('elapsed_s',elapsed)
    assert metric<config['mmd_threshold'],f'MMD²: expected < {config["mmd_threshold"]}, got {metric}'
    assert coverage['covered_modes']==8,f'covered modes: expected 8, got {coverage["covered_modes"]}'
    assert coverage['outlier_fraction']<config['outlier_threshold'],f'outlier fraction: expected < {config["outlier_threshold"]}, got {coverage["outlier_fraction"]}'
    for i,p in enumerate(coverage['proportions']):
        assert abs(p-.125)<.05,f'mode[{i}] proportion: expected .125 ± .05, got {p}'
    assert elapsed<10,f'T3 runtime: expected < 10s, got {elapsed:.3f}s'
