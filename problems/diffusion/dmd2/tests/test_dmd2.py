import pytest
import torch
from torch.testing import assert_close

@pytest.mark.t0
def test_contract(impl):
    d,g=impl.adversarial_losses(torch.zeros(3),torch.zeros(4))
    assert d.ndim==g.ndim==0
    assert isinstance(impl.generator_due(5),bool)

@pytest.mark.t1
def test_stable_logistic_losses(impl):
    real=torch.tensor([-1000.,0.,1000.],dtype=torch.float64);fake=torch.tensor([-2.,3.],dtype=real.dtype)
    d,g=impl.adversarial_losses(real,fake)
    expected=torch.logaddexp(torch.zeros_like(real),-real).mean()+torch.logaddexp(torch.zeros_like(fake),fake).mean()
    assert_close(d,expected);assert_close(g,torch.logaddexp(torch.zeros_like(fake),-fake).mean())
    assert torch.isfinite(d) and torch.isfinite(g)

@pytest.mark.t2
def test_two_time_schedule(impl):
    assert [i for i in range(16) if impl.generator_due(i,5)]==[5,10,15]
    assert [i for i in range(6) if impl.generator_due(i,2)]==[2,4]
    with pytest.raises(ValueError):impl.generator_due(1,0)

@pytest.mark.t2
def test_generator_gradient_routes(impl):
    x=torch.tensor([[.2,.7]],requires_grad=True);gradient=torch.tensor([[.5,-.3]],requires_grad=True)
    logits=x.sum(1);loss=impl.generator_loss(x,gradient,logits,.2);loss.backward()
    expected=gradient.detach()/x.numel()-.2*torch.sigmoid(-logits.detach())[:,None]
    assert_close(x.grad,expected)
    assert gradient.grad is None

@pytest.mark.t2
def test_gan_gradients_and_no_regression(impl):
    real=torch.tensor([-.2,.4],requires_grad=True);fake=torch.tensor([.1,-.8],requires_grad=True)
    d,g=impl.adversarial_losses(real,fake)
    dr,df=torch.autograd.grad(d,(real,fake),retain_graph=True)
    assert (dr<0).all() and (df>0).all()
    assert (torch.autograd.grad(g,fake)[0]<0).all()
    x=torch.tensor([[20.,-10.]],requires_grad=True)
    impl.generator_loss(x,torch.zeros_like(x),torch.zeros(1),0).backward()
    assert_close(x.grad,torch.zeros_like(x))

@pytest.mark.t3
def test_unpaired_adversarial_training(impl,artifact):
    """Actual 5:1 critic/generator updates move an unpaired Gaussian student toward real data."""
    from problems.diffusion.dmd2.toy import run
    result=run(impl)
    for k,v in result.items():artifact(k,v)
    assert result['metrics']['critic_updates']==1000
    assert result['metrics']['generator_updates']==200
    assert result['metrics']['mean_error']<.3
