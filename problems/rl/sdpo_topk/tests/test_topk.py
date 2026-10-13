import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.rl.sdpo_topk.experiment import run


def oracle(p,q,k):
    indices = sorted(range(len(p)),key=lambda i:-p[i])[:k]
    value = sum(p[i]*math.log(p[i]/q[i]) for i in indices)
    tail = [i for i in range(len(p)) if i not in indices]
    if tail:
        pp,qq = sum(p[i] for i in tail),sum(q[i] for i in tail)
        value += pp*math.log(pp/qq)
    return value


@pytest.mark.t0
def test_shape_dtype_validation_and_single_token(impl):
    """Drop only the vocabulary axis; validate k, including V=1."""
    for dtype in (torch.float32,torch.float64):
        s = torch.zeros(2,3,4,dtype=dtype)
        value = impl.topk_tail_reverse_kl(s,s,2)
        assert value.shape == (2,3) and value.dtype == dtype and value.device == s.device
        for k in (0,5,1.5,True):
            with pytest.raises(ValueError): impl.topk_tail_reverse_kl(s,s,k)
        assert impl.topk_tail_reverse_kl(torch.ones(1,dtype=dtype),torch.zeros(1,dtype=dtype),1).item() == 0


@pytest.mark.t1
def test_student_selection_and_shared_indices_scalar_oracle(impl):
    """Teacher ranking differs from student ranking; both masses use student-selected indices."""
    p,q = [.07,.5,.13,.3],[.6,.03,.3,.07]
    s,t = torch.tensor(p,dtype=torch.float64).log(),torch.tensor(q,dtype=torch.float64).log()
    for k in range(1,5):
        close(impl.topk_tail_reverse_kl(s,t,k),torch.tensor(oracle(p,q,k),dtype=s.dtype),atol=1e-12,rtol=1e-12)


@pytest.mark.t1
def test_tail_retains_mass_without_topk_renormalization(impl):
    """At k=1 the two-bin KL is not zero, unlike independently normalized singleton supports."""
    p,q = [.6,.25,.15],[.3,.2,.5]
    value = impl.topk_tail_reverse_kl(torch.tensor(p,dtype=torch.float64).log(),torch.tensor(q,dtype=torch.float64).log(),1)
    expected = .6*math.log(2)+.4*math.log(.4/.7)
    close(value,torch.tensor(expected,dtype=value.dtype),atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_refining_partition_and_full_kl(impl):
    """Splitting a tail bin cannot decrease KL; k=V agrees with independent full-vocabulary KL."""
    gen = torch.Generator().manual_seed(123)
    s,t = torch.randn(4,7,generator=gen,dtype=torch.float64),torch.randn(4,7,generator=gen,dtype=torch.float64)
    values = torch.stack([impl.topk_tail_reverse_kl(s,t,k) for k in range(1,8)])
    assert values.min() >= -1e-12 and (values[1:]-values[:-1]).min() >= -1e-12
    expected = torch.tensor([sum(p*math.log(p/q) for p,q in zip(a,b)) for a,b in zip(s.softmax(-1).tolist(),t.softmax(-1).tolist())],dtype=s.dtype)
    close(values[-1],expected,atol=1e-12,rtol=1e-12)


@pytest.mark.t2
def test_finite_difference_gradients_and_teacher_detach(impl):
    """Away from top-k ties, gradients include the student's tail mass and exclude the teacher."""
    s = torch.tensor([.7,-.3,1.4,-1.2],dtype=torch.float64,requires_grad=True)
    t = torch.tensor([-.5,.2,-.1,1.],dtype=s.dtype,requires_grad=True)
    value = impl.topk_tail_reverse_kl(s,t,2)
    gs,gt = torch.autograd.grad(value,(s,t),allow_unused=True)
    assert gt is None
    finite = []
    for i in range(4):
        plus,minus = s.detach().clone(),s.detach().clone(); plus[i] += 1e-5; minus[i] -= 1e-5
        finite.append((oracle(plus.softmax(-1).tolist(),t.detach().softmax(-1).tolist(),2)-oracle(minus.softmax(-1).tolist(),t.detach().softmax(-1).tolist(),2))/2e-5)
    close(gs,torch.tensor(finite,dtype=s.dtype),atol=1e-9,rtol=1e-7)


@pytest.mark.t2
def test_extreme_logits_shift_and_strides(impl):
    """Complement logsumexp stays finite when top mass rounds to one; strided inputs are supported."""
    s = torch.tensor([[1000.,0.,-1000.,-1500.]],dtype=torch.float64)
    t = -s; before = s.clone(),t.clone()
    value = impl.topk_tail_reverse_kl(s,t,1)
    assert torch.isfinite(value).all()
    close(value,impl.topk_tail_reverse_kl(s+900,t-700,1))
    close(s,before[0]);close(t,before[1])
    close(impl.topk_tail_reverse_kl(s[:,::2],t[:,::2],1),impl.topk_tail_reverse_kl(s[:,::2].contiguous(),t[:,::2].contiguous(),1))


@pytest.mark.t3
def test_vocabulary_information_curve(impl, artifact):
    """Submitted top-k/tail KL traces the independently computed five-category curve."""
    result = run(impl.topk_tail_reverse_kl)
    artifact('topk_information_curve',result)
    assert result['metrics']['maximum_formula_error'] < 1e-12
    assert 0 < result['metrics']['top1_kl'] < result['metrics']['full_kl']
