import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.finetuning.distillm2.experiment import run


def oracle(a, b, alpha, reverse=False):
    def probabilities(z):
        values = [math.exp(float(v)-max(z)) for v in z]
        return [v/sum(values) for v in values]
    p, q = probabilities(a), probabilities(b)
    source = q if reverse else p
    mix = [(1-alpha)*x+alpha*y if reverse else alpha*x+(1-alpha)*y for x,y in zip(p,q)]
    return sum(x*math.log(x/y) for x,y in zip(source,mix))


def fixture():
    g=torch.Generator().manual_seed(19)
    return [torch.randn(2,3,4,generator=g,dtype=torch.float64) for _ in range(4)]


@pytest.mark.t0
def test_shapes_dtypes_and_strides(impl):
    for dtype in (torch.float32,torch.float64):
        p,q=[x.to(dtype).transpose(0,1) for x in fixture()[:2]]
        before=p.clone(),q.clone()
        for fn in (impl.skew_kl,impl.skew_reverse_kl):
            out=fn(p,q,.3)
            assert out.shape==p.shape[:-1] and out.dtype==dtype and out.device==p.device
        close(p,before[0]);close(q,before[1])


@pytest.mark.t0
def test_supplied_validation(impl):
    p,q,a,b=fixture(); mask=torch.ones(2,3,dtype=torch.bool)
    for v in (-.1,1.1,float('nan')):
        with pytest.raises(ValueError): impl.skew_kl(p,q,v)
        with pytest.raises(ValueError): impl.skew_reverse_kl(p,q,v)
        with pytest.raises(ValueError): impl.distillm2_loss(p,q,mask,a,b,mask,beta=v)
    with pytest.raises(ValueError): impl.distillm2_loss(p,q,mask.float(),a,b,mask)


@pytest.mark.t1
def test_scalar_divergence_oracle(impl):
    p,q=fixture()[:2]
    for alpha in (0.,.13,.8,1.):
        for fn,rev in ((impl.skew_kl,False),(impl.skew_reverse_kl,True)):
            expected=torch.tensor([oracle(a,b,alpha,rev) for a,b in zip(p.reshape(-1,4).tolist(),q.reshape(-1,4).tolist())],dtype=p.dtype).reshape(2,3)
            close(fn(p,q,alpha),expected,atol=1e-11,rtol=1e-11)


@pytest.mark.t1
def test_eq2_uses_distinct_sources_and_half_batch_reduction(impl):
    p,q,a,b=fixture(); mt=torch.tensor([[1,0,1],[1,0,0]],dtype=torch.bool); ms=~mt
    for beta in (0.,.3,1.):
        t=sum(oracle(p[i,j].tolist(),q[i,j].tolist(),.2) for i in range(2) for j in range(3) if mt[i,j])
        s=sum(oracle(a[i,j].tolist(),b[i,j].tolist(),.7,True) for i in range(2) for j in range(3) if ms[i,j])
        expected=torch.tensor(((1-beta)*t+beta*s)/4,dtype=p.dtype)
        close(impl.distillm2_loss(p,q,mt,a,b,ms,.2,.7,beta),expected,atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_student_gradients_finite_difference_teacher_frozen(impl):
    for fn,rev in ((impl.skew_kl,False),(impl.skew_reverse_kl,True)):
        p=torch.tensor([1.,-.8,2.],dtype=torch.float64,requires_grad=True)
        q=torch.tensor([-.6,1.5,.4],dtype=p.dtype,requires_grad=True)
        gt,gq=torch.autograd.grad(fn(p,q,.27),(p,q),allow_unused=True)
        assert gt is None or gt.abs().max()==0
        eps=1e-6; expected=[]
        for i in range(3):
            lo=q.detach().tolist();hi=lo.copy();lo[i]-=eps;hi[i]+=eps
            expected.append((oracle(p.tolist(),hi,.27,rev)-oracle(p.tolist(),lo,.27,rev))/(2*eps))
        close(gq,torch.tensor(expected,dtype=p.dtype),atol=1e-8,rtol=1e-7)


@pytest.mark.t2
def test_padding_and_sequence_sum_contract(impl):
    p,q,a,b=[x.requires_grad_() for x in fixture()];mask=torch.tensor([[1,0,0],[1,1,0]],dtype=torch.bool)
    out=impl.distillm2_loss(p,q,mask,a,b,mask)
    grads=torch.autograd.grad(out,(q,b))
    for grad in grads: assert grad[~mask].abs().max()==0
    def pad(x):return torch.cat([x,torch.full((2,4,4),100.,dtype=x.dtype)],dim=1)
    padded=torch.cat([mask,torch.zeros(2,4,dtype=torch.bool)],dim=1)
    close(impl.distillm2_loss(pad(p),pad(q),padded,pad(a),pad(b),padded),out)
    close(impl.distillm2_loss(p.repeat(1,2,1),q.repeat(1,2,1),mask.repeat(1,2),a.repeat(1,2,1),b.repeat(1,2,1),mask.repeat(1,2)),2*out)


@pytest.mark.t2
def test_extreme_logits_and_logit_shift_invariance(impl):
    p=torch.tensor([[1000.,-1000.,0.]],dtype=torch.float64);q=-p
    for fn in (impl.skew_kl,impl.skew_reverse_kl):
        for alpha in (0.,.2,1.):
            value=fn(p,q,alpha)
            assert torch.isfinite(value).all()
            close(fn(p+777,q-999,alpha),value,atol=1e-10,rtol=1e-10)


@pytest.mark.t2
def test_zero_and_empty_response_tokens(impl):
    p,q,a,b=fixture();off=torch.zeros(2,3,dtype=torch.bool)
    close(impl.distillm2_loss(p,q,off,a,b,off),torch.tensor(0.,dtype=p.dtype))
    for fn in (impl.skew_kl,impl.skew_reverse_kl):
        close(fn(p,p,.3),torch.zeros(2,3,dtype=p.dtype),atol=1e-15,rtol=0)
        student=q.detach().clone().requires_grad_()
        endpoint=fn(p,student,1.).sum()
        close(torch.autograd.grad(endpoint,student)[0],torch.zeros_like(student))
    empty=torch.empty(2,0,4,dtype=p.dtype); mask=torch.empty(2,0,dtype=torch.bool)
    close(impl.distillm2_loss(empty,empty,mask,a,b,off),torch.tensor(0.,dtype=p.dtype))


@pytest.mark.t3
def test_two_source_toy(impl,artifact):
    result=run(impl.skew_kl,impl.skew_reverse_kl,impl.distillm2_loss)
    artifact('two_source_distillation',result)
    assert result['metrics']['initial_loss']>.05
    assert 0 <= result['metrics']['loss_ratio'] < .08
