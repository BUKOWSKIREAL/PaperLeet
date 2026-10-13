import pytest
import torch
from torch.testing import assert_close

@pytest.mark.t0
def test_shapes(impl):
    a=torch.randn(3,2,dtype=torch.float64);w=torch.ones(3,dtype=a.dtype)
    g=impl.score_difference(a,a,w)
    assert g.shape==a.shape and g.dtype==a.dtype and not g.requires_grad
    assert impl.dmd_loss(a,g,a).ndim==0

@pytest.mark.t1
def test_gradient_sign_and_scaling(impl):
    real=torch.tensor([[1.,2.],[-1.,.3]],requires_grad=True);fake=torch.tensor([[2.,-1.],[.2,.7]],requires_grad=True)
    w=torch.tensor([.5,2.]);g=impl.score_difference(real,fake,w)
    assert_close(g,w[:,None]*(fake-real));assert not g.requires_grad
    x=torch.ones(2,2,requires_grad=True);paired=torch.zeros_like(x,requires_grad=True)
    loss=impl.dmd_loss(x,g,paired,.2);loss.backward()
    assert_close(x.grad,g/4+.4*x.detach()/4);assert paired.grad is None

@pytest.mark.t2
def test_frozen_matching_gradient(impl):
    x=torch.tensor([[1.,2.]],requires_grad=True);grad=2*x
    impl.dmd_loss(x,grad,torch.zeros_like(x),0).backward()
    assert_close(x.grad,torch.tensor([[1.,2.]]))

@pytest.mark.t2
def test_fake_model_only_receives_gradient(impl):
    x=torch.ones(3,2,requires_grad=True);noise=torch.ones_like(x);sigma=torch.tensor([.2,.5,1.])
    a=torch.tensor(.3,requires_grad=True)
    model=lambda noisy,s:a*noisy
    loss=impl.fake_score_loss(model,x,noise,sigma);loss.backward()
    assert a.grad is not None and a.grad.abs()>0
    assert x.grad is None or x.grad.eq(0).all()
    expected=((sigma[:,None]*(.3*(x.detach()+sigma[:,None]*noise)))+noise).square().mean()
    assert_close(loss,expected)

@pytest.mark.t2
def test_exact_conditional_score_zero(impl):
    x=torch.zeros(4,2);noise=torch.randn_like(x);sigma=torch.tensor([.1,.2,.4,.8])
    assert impl.fake_score_loss(lambda noisy,s:-noisy/s[:,None].square(),x,noise,sigma)<1e-12

@pytest.mark.t3
def test_gaussian_matching_with_fake_score_training(impl,artifact):
    """Student and fake score are updated separately in a fixed-noise Gaussian toy."""
    from problems.diffusion.dmd.toy import run
    result=run(impl)
    for k,v in result.items():artifact(k,v)
    assert result['metrics']['mean_error']<.15
    assert result['metrics']['fake_mean_tracking_error']<.15

@pytest.mark.t3
def test_bimodal_mode_coverage(impl,artifact):
    """Figure-inspired analytic mixtures expose real-score-only collapse and matching coverage."""
    from problems.diffusion.dmd.toy import run_bimodal
    result=run_bimodal(impl)
    for k,v in result.items():artifact(k,v)
    for strategy in ('matching','matching_regression'):
        assert result['metrics'][strategy+'_covered_modes']==2
        assert result['metrics'][strategy+'_center_error']<.2
