import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.representation.rotation_trick.experiment import run


def angle_oracle(e,q,g):
    # Independent 2D sin/cos rotation; no Householder expression from the reference.
    a = math.atan2(q[1],q[0]) - math.atan2(e[1],e[0])
    scale = math.hypot(*q) / math.hypot(*e)
    return [scale*(math.cos(a)*g[0]+math.sin(a)*g[1]),
            scale*(-math.sin(a)*g[0]+math.cos(a)*g[1])]


@pytest.mark.t0
def test_shape_dtype_strides_and_inputs(impl):
    """Forward output has q's values with the input shape, dtype and device, including strided tensors."""
    for dtype in (torch.float32,torch.float64):
        for shape in ((3,), (2,3), (2,4,3)):
            e = torch.rand(shape,dtype=dtype)+.2
            q = torch.rand(shape,dtype=dtype)+.2
            if len(shape)>1: e,q = e.transpose(0,1),q.transpose(0,1)
            before=e.clone(),q.clone()
            actual=impl.rotation_quantize(e,q)
            assert actual.shape==e.shape and actual.dtype==dtype and actual.device==e.device
            close(actual,q,atol=2e-6,rtol=2e-6)
            close(e,before[0]);close(q,before[1])


@pytest.mark.t0
def test_supplied_degenerate_domain_validation(impl):
    """Zero, antiparallel and invalid shapes/eps are explicitly outside this rotation's domain."""
    e=torch.tensor([1.,2.],dtype=torch.float64)
    for q in (torch.zeros_like(e),-2*e,e[:1]):
        with pytest.raises(ValueError): impl.rotation_quantize(e,q)
    with pytest.raises(ValueError): impl.rotation_quantize(torch.zeros_like(e),e)
    with pytest.raises(ValueError): impl.rotation_quantize(e[:0],e[:0])
    for eps in (0.,-1.,float('inf'),float('nan')):
        with pytest.raises(ValueError): impl.rotation_quantize(e,e,eps)


@pytest.mark.t1
def test_independent_two_dimensional_rotation_gradient(impl):
    """Analytic angle matrices check rotation direction, scaling and frozen-transform differentiation."""
    e=torch.tensor([[2.,.3],[-.7,2.],[1.,1.]],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([[.4,1.7],[1.,.8],[3.,3.]],dtype=e.dtype,requires_grad=True)
    upstream=torch.tensor([[.3,-2.],[1.2,.7],[-1.,.4]],dtype=e.dtype)
    actual=impl.rotation_quantize(e,q)
    de,dq=torch.autograd.grad((actual*upstream).sum(),(e,q),allow_unused=True)
    expected=e.new_tensor([angle_oracle(a,b,g) for a,b,g in zip(e.detach().tolist(),q.detach().tolist(),upstream.tolist())])
    close(actual,q,atol=1e-12,rtol=1e-12)
    close(de,expected,atol=1e-12,rtol=1e-12)
    assert dq is None or torch.count_nonzero(dq)==0


@pytest.mark.t1
def test_identity_rotation_keeps_scale_in_jacobian(impl):
    """Collinear positive pairs map forward to q, and multiply arbitrary upstream gradients by norm ratio."""
    e=torch.tensor([[1.,-2.,3.],[2.,1.,-.5]],dtype=torch.float64,requires_grad=True)
    scale=e.new_tensor([[2.5],[.3]])
    q=e.detach()*scale
    g=torch.tensor([[.1,.3,2.],[-1.,2.,.4]],dtype=e.dtype)
    out=impl.rotation_quantize(e,q)
    de,=torch.autograd.grad((out*g).sum(),e)
    close(out,q,atol=1e-12,rtol=1e-12)
    close(de,scale*g,atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_high_dimensional_angle_norm_and_orthogonal_subspace(impl):
    """The backward map preserves the angle to its anchor and scales norm; the orthogonal complement stays fixed."""
    e=torch.tensor([2.,0.,0.,0.],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([0.,3.,0.,0.],dtype=e.dtype)
    g=torch.tensor([.4,-.7,1.1,-2.],dtype=e.dtype)
    de,=torch.autograd.grad((impl.rotation_quantize(e,q)*g).sum(),e)
    close(de,e.new_tensor([-1.05,-.6,1.65,-3.]),atol=1e-12,rtol=1e-12)
    close(de.norm(),1.5*g.norm(),atol=1e-12,rtol=1e-12)
    close((de*e).sum()/(de.norm()*e.norm()),(g*q).sum()/(g.norm()*q.norm()),atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_batch_independence_and_near_antiparallel_pair(impl):
    """Distinct samples do not mix; near-opposite valid vectors still use the plane rotation."""
    e=torch.tensor([[1.,0.],[.5,2.]],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([[-1.,.001],[1.,1.]],dtype=e.dtype)
    g=torch.tensor([[1.,2.],[0.,0.]],dtype=e.dtype)
    out=impl.rotation_quantize(e,q)
    de,=torch.autograd.grad((out*g).sum(),e)
    close(out,q,atol=1e-9,rtol=1e-9)
    close(de[0],e.new_tensor(angle_oracle(e[0].detach().tolist(),q[0].tolist(),g[0].tolist())),atol=1e-9,rtol=1e-9)
    close(de[1],torch.zeros_like(de[1]))


@pytest.mark.t3
def test_paper_derived_quadratic_gradient_curve(impl,artifact):
    """The submitted surrogate creates a real autograd curve on the paper's quadratic example."""
    result=run(impl.rotation_quantize)
    artifact('quadratic_gradient_geometry',result)
    for key in ('forward_error','angle_preservation_error','gradient_norm_error'):
        assert result['metrics'][key]<1e-10
    assert result['metrics']['ste_max_angle_error']>.01
