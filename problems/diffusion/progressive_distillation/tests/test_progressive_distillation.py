import pytest
import torch
from torch.testing import assert_close

@pytest.mark.t0
def test_contract(impl):
    x=torch.randn(3,2,dtype=torch.float64);at=torch.full((3,),.6);st=torch.full((3,),.8);a=torch.ones(3);s=torch.zeros(3)
    y=impl.distilled_target(x,torch.ones_like(x),at,st,a,s)
    assert y.shape==x.shape and y.dtype==x.dtype and not y.requires_grad

@pytest.mark.t1
def test_algebraic_two_step_endpoint(impl):
    x=torch.tensor([[.2,-.7],[1.,.3]],dtype=torch.float64);t=torch.tensor([.8,.6],dtype=x.dtype);m=t*.7;e=t*.4
    a=lambda v:(v*torch.pi/2).cos();s=lambda v:(v*torch.pi/2).sin()
    teacher=lambda x,t: .4*x+torch.tensor([.5,-.2],dtype=x.dtype)
    def oracle_step(x,p,t,e):return a(e)[:,None]*p+s(e)[:,None]*(x-a(t)[:,None]*p)/s(t)[:,None]
    midpoint=oracle_step(x,teacher(x,t),t,m);endpoint=oracle_step(midpoint,teacher(midpoint,m),m,e)
    assert_close(impl.teacher_pair(teacher,x,t,m,e),endpoint)
    target=impl.distilled_target(x,endpoint,a(t),s(t),a(e),s(e))
    assert_close(impl.ddim_step(x,target,a(t),s(t),a(e),s(e)),endpoint)

@pytest.mark.t2
def test_target_stopgrad_and_weight(impl):
    x=torch.randn(2,2,requires_grad=True);end=torch.randn(2,2,requires_grad=True);at=torch.tensor([.6,.8]);st=torch.tensor([.8,.6]);a=torch.ones(2);s=torch.zeros(2)
    target=impl.distilled_target(x,end,at,st,a,s)
    assert not target.requires_grad
    pred=torch.ones(2,2,requires_grad=True);other=torch.zeros(2,2,requires_grad=True);w=torch.tensor([1.,3.])
    impl.distillation_loss(pred,other,w).backward()
    assert_close(pred.grad,w[:,None].expand_as(pred)/2);assert other.grad is None

@pytest.mark.t2
def test_clean_endpoint_and_two_calls(impl):
    x=torch.randn(5,2);calls=[];t=torch.ones(5)*.8;m=torch.ones(5)*.4;e=torch.zeros(5)
    def teacher(x,t):calls.append(t.clone());return torch.ones_like(x)*.7
    endpoint=impl.teacher_pair(teacher,x,t,m,e)
    assert len(calls)==2;assert_close(calls[0],t);assert_close(calls[1],m)
    assert_close(endpoint,torch.full_like(x,.7))

@pytest.mark.t2
def test_target_is_not_first_prediction(impl):
    x=torch.tensor([[2.,-1.]]);t=torch.tensor([.8]);m=torch.tensor([.4]);e=torch.tensor([.2]);a=lambda v:(v*torch.pi/2).cos();s=lambda v:(v*torch.pi/2).sin()
    teacher=lambda x,t:.3*x+.2
    end=impl.teacher_pair(teacher,x,t,m,e);target=impl.distilled_target(x,end,a(t),s(t),a(e),s(e))
    assert not torch.allclose(target,teacher(x,t))

@pytest.mark.t3
def test_student_matches_two_step_teacher(impl,artifact):
    """A learned one-step linear student matches a separately evaluated two-call teacher."""
    from problems.diffusion.progressive_distillation.toy import run
    result=run(impl)
    for k,v in result.items():artifact(k,v)
    assert result['metrics']['endpoint_mse']<1e-5
    assert result['metrics']['endpoint_mse']<result['metrics']['initial_endpoint_mse']*.01
