from pathlib import Path
import pytest
import torch
import yaml
from paperleet.common.asserts import assert_close_located as close
from paperleet.common.toydata import GaussianMixture
from paperleet.common.oracles import gmm_eps_oracle
from paperleet.common.metrics import mmd_rbf,mode_coverage


def data():
    ab=(1-torch.linspace(1e-4,.02,1000,dtype=torch.float64)).cumprod(0)
    a=torch.arange(8,dtype=torch.float64)*torch.pi/4
    g=GaussianMixture(2*torch.stack((a.cos(),a.sin()),-1),.15)
    return ab,g,gmm_eps_oracle(g,ab)


@pytest.mark.t0
def test_shape(impl):
    """DDIM 更新应广播时间系数并保留样本形状。"""
    x=torch.randn(3,2);a=torch.tensor([[.1],[.3],[.5]])
    got=impl.ddim_step(x,x,a,a+.1,0.,x)
    assert got.shape==x.shape,f'output.shape: expected {x.shape}, got {got.shape}'


@pytest.mark.t1
def test_reference(impl,ref):
    """固定输入噪声后，三种 eta 的更新都由 DDIM 公式确定。"""
    x,e,n=torch.randn(3,4,2,dtype=torch.float64);a=torch.tensor(.3);p=torch.tensor(.6)
    for eta in (0.,.5,1.):
        close(impl.ddim_step(x,e,a,p,eta,n),ref.ddim_step(x,e,a,p,eta,n),rtol=1e-8,atol=1e-8)


@pytest.mark.t2
def test_eta_one_posterior(impl):
    """eta=1 时随机项的方差等于相邻时间步的 DDPM 后验方差。"""
    ab,_,_=data();t=torch.tensor([1,20,400,900]);a,p=ab[t,None],ab[t-1,None]
    x,e=torch.randn(2,4,2,dtype=torch.float64)
    difference=impl.ddim_step(x,e,a,p,1.,torch.ones_like(x))-impl.ddim_step(x,e,a,p,1.,torch.zeros_like(x))
    expected=((1-p)/(1-a)*(1-a/p)).expand_as(x)
    close(difference.square(),expected,rtol=1e-9,atol=1e-10)


@pytest.mark.t2
def test_deterministic(impl):
    """eta=0 使随机项完全消失。"""
    x,e,n=torch.randn(3,5,2,dtype=torch.float64);a=torch.tensor(.2);p=torch.tensor(.6)
    close(impl.ddim_step(x,e,a,p,0.,n),impl.ddim_step(x,e,a,p,0.,n*100),rtol=0,atol=0)


@pytest.mark.t2
def test_point_final_jump(impl):
    """单点的 x0 预测完全准确，因此包含干净端点后一步或多步都必须到达该点。"""
    ab,_,_=data();mu=torch.tensor([[1.3,-.7]],dtype=torch.float64);oracle=gmm_eps_oracle(GaussianMixture(mu,0.),ab)
    x=torch.randn(100,2,dtype=torch.float64)
    for n in (1,50):
        close(impl.ddim_sample(oracle,x,ab,n),mu.expand_as(x),rtol=0,atol=1e-6)


@pytest.mark.t2
def test_trailing_times(impl):
    """trailing 子序列从 T-1 开始，按整除步距严格递减。"""
    for T,n in [(1000,50),(17,5),(10,10),(10,1)]:
        expected=T-1-torch.arange(n)*(T//n)
        close(impl.ddim_timesteps(T,n),expected,rtol=0,atol=0)


@pytest.mark.t2
def test_inversion_converges(impl,artifact):
    """确定性离散更新的反演误差应随更密的时间网格下降。"""
    ab,g,oracle=data();x=g.sample(200,torch.Generator().manual_seed(77))
    errors=[]
    for n in (10,20,50,100):
        reconstructed=impl.ddim_sample(oracle,impl.ddim_invert(oracle,x,ab,n),ab,n)
        errors.append(float((reconstructed-x).norm(dim=-1).mean()))
    artifact('roundtrip_errors',{'steps':[10,20,50,100],'errors':errors})
    assert all(a>b for a,b in zip(errors,errors[1:])),f'roundtrip errors: expected strict decrease, got {errors}'
    assert errors[-1]<.06,f'roundtrip[100]: expected < .06, got {errors[-1]}'


@pytest.mark.t2
def test_step_convergence(impl,artifact):
    """同一初始噪声下，增加确定性采样步数应得到相近轨迹终点。"""
    ab,_,oracle=data();x=torch.randn(400,2,dtype=torch.float64)
    distance=float((impl.ddim_sample(oracle,x,ab,50)-impl.ddim_sample(oracle,x,ab,500)).norm(dim=-1).mean())
    artifact('step_convergence_distance',distance)
    assert distance<.08,f'50-vs-500 mean distance: expected < .08, got {distance}'


@pytest.mark.t3
def test_gmm_sampling(impl,artifact):
    """50 步确定性 DDIM 应在解析 oracle 下恢复八个数据模式。"""
    ab,g,oracle=data();x=torch.randn(2000,2,dtype=torch.float64,generator=torch.Generator().manual_seed(42))
    samples=impl.ddim_sample(oracle,x,ab,50);real=g.sample(2000,torch.Generator().manual_seed(123))
    metric=mmd_rbf(samples,real);coverage=mode_coverage(samples,g.means,.5)
    threshold=yaml.safe_load((Path(__file__).parents[1]/'problem.yaml').read_text())['calibration']['mmd_threshold']
    comparison=impl.ddim_sample(oracle,x,ab,50,eta=1.,generator=torch.Generator().manual_seed(42))
    artifact('samples',samples);artifact('reference_samples',real);artifact('mmd',metric);artifact('threshold',threshold);artifact('coverage',coverage);artifact('eta_one_mmd',mmd_rbf(comparison,real))
    assert metric<threshold,f'MMD²: expected < {threshold}, got {metric}'
    for i,p in enumerate(coverage['proportions']):
        assert abs(p-.125)<.03,f'mode[{i}]: expected .125 ± .03, got {p}'
    assert coverage['outlier_fraction']<.05,f'outlier fraction: expected < .05, got {coverage["outlier_fraction"]}'
