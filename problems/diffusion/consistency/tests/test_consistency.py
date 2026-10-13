import pytest
import torch
from torch.testing import assert_close

@pytest.mark.t0
def test_contract(impl):
    x=torch.randn(3,2,dtype=torch.float64);s=torch.full((3,),.2,dtype=x.dtype)
    y=impl.consistency_function(lambda x,s:x,x,s)
    assert y.shape==x.shape and y.dtype==x.dtype

@pytest.mark.t1
def test_coefficients_and_network_input(impl):
    x=torch.tensor([[1.,2.],[-.7,.4]],dtype=torch.float64);s=torch.tensor([.1,.7],dtype=x.dtype);calls=[]
    def model(z,t):calls.append(z);return 2*z
    y=impl.consistency_function(model,x,s,.01,.5)
    skip=.25/((s-.01).square()+.25);out=(s-.01)*.5/(s.square()+.25).sqrt();scaled=x/(s[:,None].square()+.25).sqrt()
    assert_close(calls[0],scaled);assert_close(y,skip[:,None]*x+out[:,None]*2*scaled)

@pytest.mark.t2
def test_exact_boundary(impl):
    x=torch.randn(4,2);s=torch.full((4,),.002)
    assert_close(impl.consistency_function(lambda z,s:z*123,x,s),x,rtol=0,atol=0)

@pytest.mark.t2
def test_stopgrad_target(impl):
    p=torch.tensor([[1.,2.]],requires_grad=True);target=torch.tensor([[.2,-.1]],requires_grad=True)
    loss=impl.consistency_loss(p,target);loss.backward()
    assert_close(p.grad,p.detach()-target.detach());assert target.grad is None

@pytest.mark.t2
def test_ema_and_teacher_direction(impl):
    online=torch.nn.Linear(2,1,bias=False);target=torch.nn.Linear(2,1,bias=False)
    with torch.no_grad():online.weight.fill_(2);target.weight.fill_(-1)
    impl.update_ema(online,target,.8)
    assert_close(target.weight,torch.full_like(target.weight,-.4));assert target.weight.grad is None
    x=torch.tensor([[2.,-3.]]);clean=torch.tensor([[1.,1.]])
    assert_close(impl.teacher_euler(x,clean,torch.ones(1),torch.zeros(1)),clean)

@pytest.mark.t3
def test_point_consistency_distillation(impl,artifact):
    """Distill an exact point-data teacher and learn a one-evaluation consistency model."""
    from problems.diffusion.consistency.toy import run
    result=run(impl)
    for k,v in result.items():artifact(k,v)
    assert result['metrics']['endpoint_mse']<.04
    assert result['metrics']['boundary_error']==0

@pytest.mark.t2
def test_adjacent_target_depends_on_ema_but_is_frozen(impl):
    x=torch.tensor([[.4,-.2]],requires_grad=True);sigma=torch.tensor([.4]);weight=torch.tensor(.2,requires_grad=True)
    lower=impl.consistency_function(lambda z,s:weight*z,x,sigma)
    changed=impl.consistency_function(lambda z,s:(weight+1)*z,x,sigma)
    assert not torch.allclose(lower,changed), 'interior-time target must use the EMA network'
    prediction=torch.zeros_like(x,requires_grad=True)
    impl.consistency_loss(prediction,lower).backward()
    assert weight.grad is None and x.grad is None
