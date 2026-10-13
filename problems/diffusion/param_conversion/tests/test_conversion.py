import itertools
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close


def data():
    x0, eps = torch.randn(7, 3, dtype=torch.float64), torch.randn(7, 3, dtype=torch.float64)
    a = torch.linspace(.01, .99, 7, dtype=torch.float64)[:, None]
    s = (1 - a.square()).sqrt()
    return x0, eps, a, s, a * x0 + s * eps


@pytest.mark.t0
def test_shape(impl):
    """可广播的系数应保持每个样本的形状与精度。"""
    x0, eps, a, s, xt = data()
    for src in ('x0', 'eps', 'v'):
        got = impl.convert(x0, src, 'eps', xt, a, s)
        assert got.shape == xt.shape, f'output.shape: expected {xt.shape}, got {got.shape}'
        assert got.dtype == xt.dtype, f'output.dtype: expected {xt.dtype}, got {got.dtype}'


@pytest.mark.t1
def test_reference(impl, ref):
    """全部参数化应满足同一线性变换约定。"""
    x0, eps, a, s, xt = data()
    for src, dst in itertools.product(('x0', 'eps', 'v'), repeat=2):
        close(impl.convert(eps, src, dst, xt, a, s), ref.convert(eps, src, dst, xt, a, s), rtol=1e-10, atol=1e-10)


@pytest.mark.t2
def test_six_roundtrips(impl):
    """六种有向转换互为逆变换，内部计算须保留 float64 精度。"""
    x0, eps, a, s, xt = data()
    predictions = {'x0': x0, 'eps': eps, 'v': a * eps - s * x0}
    for src, dst in itertools.permutations(predictions, 2):
        got = impl.convert(predictions[src], src, dst, xt, a, s)
        close(got, predictions[dst], rtol=0, atol=1e-10)
        close(impl.convert(got, dst, src, xt, a, s), predictions[src], rtol=0, atol=1e-10)


@pytest.mark.t2
def test_velocity_tangent(impl):
    """v 是 x_t 对旋转角的导数，因此应等于中心有限差分。"""
    x0, eps, a, s, _ = data()
    phi, h = a.acos(), 1e-5
    finite = ((phi+h).cos()*x0+(phi+h).sin()*eps-(phi-h).cos()*x0-(phi-h).sin()*eps)/(2*h)
    close(impl.make_v(x0, eps, a, s), finite, rtol=0, atol=1e-6)


@pytest.mark.t2
def test_endpoint_stability(impl):
    """旋转变换在端点仍精确；除法 clamp 阻止零除导致 inf 或 NaN。"""
    x0, eps = torch.randn(2, 3), torch.randn(2, 3)
    a, s = torch.tensor([[0.], [1.]]), torch.tensor([[1.], [0.]])
    xt, v = a*x0+s*eps, a*eps-s*x0
    close(impl.v_to_x0(xt,v,a,s),x0,rtol=0,atol=0)
    close(impl.v_to_eps(xt,v,a,s),eps,rtol=0,atol=0)
    for name, got in [('eps_to_x0',impl.eps_to_x0(xt,eps+1,a,s)),('x0_to_eps',impl.x0_to_eps(xt,x0+1,a,s))]:
        assert torch.isfinite(got).all(), f'{name} endpoint: expected finite, got {got.tolist()}'


@pytest.mark.t2
def test_error_weight_identity(impl):
    """同一错误预测换算后的误差平方仅由 alpha、sigma 决定权重。"""
    x0, eps, a, s, xt = data()
    truth = {'x0':x0,'eps':eps,'v':a*eps-s*x0}
    for src in truth:
        pred = truth[src] + torch.randn_like(x0)
        errors = {dst:(impl.convert(pred,src,dst,xt,a,s)-truth[dst]).square() for dst in truth}
        close(errors['v'],errors['x0']/s.square(),rtol=1e-8,atol=1e-8)
        close(errors['v'],errors['eps']/a.square(),rtol=1e-8,atol=1e-8)


@pytest.mark.t2
def test_conversion_routes(impl):
    """直接换算与先经 x0 的路径应给出相同预测。"""
    x0, eps, a, s, xt = data()
    for src,dst in itertools.product(('x0','eps','v'), repeat=2):
        via = impl.convert(impl.convert(eps,src,'x0',xt,a,s),'x0',dst,xt,a,s)
        close(impl.convert(eps,src,dst,xt,a,s),via,rtol=1e-10,atol=1e-10)
