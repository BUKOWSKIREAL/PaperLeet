import pytest
import torch
from torch.testing import assert_close

@pytest.mark.t0
def test_contract(impl):
    z = torch.randn(3,2,dtype=torch.float64)
    p,y=impl.meanflow_target(lambda z,r,t: z*2,z,torch.zeros(3),torch.ones(3),z)
    assert p.shape==z.shape and p.dtype==z.dtype and not y.requires_grad

@pytest.mark.t1
def test_jvp_total_derivative(impl):
    z=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64);v=torch.tensor([[.3,.7],[-.2,.4]],dtype=z.dtype)
    r=torch.tensor([.1,.2],dtype=z.dtype);t=torch.tensor([.6,.8],dtype=z.dtype)
    model=lambda z,r,t: 2*z+3*r[:,None]+5*t[:,None]
    p,y=impl.meanflow_target(model,z,r,t,v)
    assert_close(p,model(z,r,t));assert_close(y,v-(t-r)[:,None]*(2*v+5))

@pytest.mark.t2
def test_target_is_frozen(impl):
    a=torch.tensor(2.,requires_grad=True);z=torch.tensor([[.3,.8]])
    v=torch.tensor([[.5,-.2]]);r=torch.tensor([.1]);t=torch.tensor([.7])
    pred,target=impl.meanflow_target(lambda z,r,t:a*z,z,r,t,v)
    assert not target.requires_grad
    loss=(pred-target).square().mean();loss.backward()
    assert_close(a.grad, (2*(pred.detach()-target)*z).mean())

@pytest.mark.t2
def test_diagonal_is_fm_target(impl):
    z=torch.randn(4,2);t=torch.rand(4);v=torch.randn_like(z)
    p,y=impl.meanflow_target(lambda z,r,t:z.square()+t[:,None],z,t,t,v)
    assert_close(y,v)
    # The target reduces to v; an untrained prediction is not forced to equal v.
    assert not torch.allclose(p,v)

@pytest.mark.t2
def test_single_call_and_direction(impl):
    calls=[];noise=torch.tensor([[2.,-3.]])
    def model(z,r,t):
        calls.append((r.clone(),t.clone()));return torch.ones_like(z)
    assert_close(impl.one_step(model,noise),noise-1)
    assert len(calls)==1 and calls[0][0].eq(0).all() and calls[0][1].eq(1).all()

@pytest.mark.t3
def test_point_distribution_training(impl,artifact):
    """Learn a point-data MeanFlow model with real JVP targets and one-pass inference."""
    from problems.diffusion.meanflow.toy import run
    result=run(impl)
    for k,v in result.items():artifact(k,v)
    assert result['metrics']['endpoint_mse'] < 1e-4
