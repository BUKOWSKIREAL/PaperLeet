import pytest
import torch
from torch.testing import assert_close

@pytest.mark.t0
def test_shapes(impl):
    x=torch.randn(3,4,2,dtype=torch.float64);e=torch.randn_like(x);t=torch.tensor([.2,.5,.8])
    z=impl.make_noisy(x,e,t)
    assert z.shape==x.shape and z.dtype==x.dtype
    assert impl.velocity_loss(x,x,e,t).ndim==0

@pytest.mark.t1
def test_prediction_spaces(impl):
    x=torch.tensor([[2.,-1.],[.3,.7]],dtype=torch.float64);e=torch.tensor([[.4,.6],[-.2,1.]],dtype=x.dtype);t=torch.tensor([.2,.7],dtype=x.dtype)
    z=t[:,None]*x+(1-t[:,None])*e
    assert_close(impl.make_noisy(x,e,t),z)
    for pred,kind in ((x,'x'),(e,'eps'),(x-e,'v')):
        assert_close(impl.prediction_to_velocity(pred,z,t,kind),x-e)
        assert impl.velocity_loss(pred,x,e,t,kind).abs()<1e-24

@pytest.mark.t2
def test_v_loss_weight_and_gradient(impl):
    x=torch.randn(2,3);e=torch.randn_like(x);t=torch.tensor([.2,.8]);pred=torch.randn_like(x,requires_grad=True)
    loss=impl.velocity_loss(pred,x,e,t);loss.backward()
    assert_close(loss,((pred-x)/(1-t[:,None])).square().mean())
    assert_close(pred.grad,2*(pred.detach()-x)/(1-t[:,None]).square()/x.numel())

@pytest.mark.t2
def test_endpoints_are_finite(impl):
    x=torch.ones(2,2);e=torch.zeros_like(x);t=torch.tensor([0.,1.])
    assert_close(impl.make_noisy(x,e,t),torch.tensor([[0.,0.],[1.,1.]]))
    for kind in ('x','eps','v'):
        assert torch.isfinite(impl.velocity_loss(x,x,e,t,kind))
    with pytest.raises(ValueError):impl.prediction_to_velocity(x,x,t,'unknown')

@pytest.mark.t2
def test_same_raw_error_has_time_weight(impl):
    x=torch.zeros(2,1);e=torch.ones_like(x);pred=torch.ones_like(x)
    low=impl.velocity_loss(pred,x,e,torch.full((2,),.2))
    high=impl.velocity_loss(pred,x,e,torch.full((2,),.8))
    assert_close(high,16*low)

@pytest.mark.t3
def test_embedded_manifold_denoising(impl,artifact):
    """Train identical-capacity x/epsilon/v networks on an embedded 2D manifold."""
    from problems.diffusion.jit.toy import run
    result=run(impl)
    for k,v in result.items():artifact(k,v)
    assert result['metrics']['projection_orthogonality_error']<1e-5
    assert result['metrics']['x_final_loss']<result['metrics']['x_initial_loss']*.75
    assert result['metrics']['x_final_loss']<3.
