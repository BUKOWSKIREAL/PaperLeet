import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.rl.sdpo_feedback.experiment import run


@pytest.mark.t0
def test_interface_and_single_attempt(impl):
    """Table 2 uses a successful attempt as its own solution; failed attempts require other feedback."""
    for successful in (False, True):
        for rich in (False, True):
            solution, available = impl.select_feedback(torch.tensor([42]), torch.tensor([successful]), torch.tensor([rich]))
            assert solution.tolist() == ([0] if successful else [-1])
            assert available.tolist() == [successful or rich]
            assert solution.dtype == torch.int64 and available.dtype == torch.bool
    for dtype in (torch.float32,torch.float64):
        x = torch.zeros(1, 2, 3, dtype=dtype)
        y = impl.feedback_reverse_kl(x, x, torch.ones(1,2,dtype=torch.bool), torch.tensor([True]))
        assert y.shape == () and y.dtype == dtype and y.device == x.device


@pytest.mark.t1
def test_self_then_same_question_solution_before_rich_feedback(impl):
    """Own success wins; otherwise use the first same-question success, both before rich feedback."""
    ids = torch.tensor([7, 8, 7, 7, 8, 9])
    success = torch.tensor([False, True, True, True, False, False])
    rich = torch.tensor([True, False, False, True, True, True])
    peer, available = impl.select_feedback(ids, success, rich)
    assert peer.tolist() == [2, 1, 2, 3, 1, -1]
    assert available.tolist() == [True, True, True, True, True, True]


@pytest.mark.t1
def test_reverse_kl_and_available_token_mean(impl):
    """A scalar probability oracle checks KL direction and the valid-token denominator."""
    p = torch.tensor([[[.8,.2],[.1,.9]],[[.3,.7],[.4,.6]],[[.6,.4],[.2,.8]]],dtype=torch.float64)
    q = torch.tensor([.4,.6],dtype=p.dtype).expand_as(p)
    mask = torch.tensor([[True,True],[True,False],[True,True]])
    available = torch.tensor([True,True,False])
    expected = sum(sum(v*math.log(v/w) for v,w in zip(row,[.4,.6])) for row in ([.8,.2],[.1,.9],[.3,.7]))/3
    close(impl.feedback_reverse_kl(p.log(),q.log(),mask,available),torch.tensor(expected,dtype=p.dtype))


@pytest.mark.t2
def test_no_feedback_zero_and_no_cross_question_leakage(impl):
    """Success on another question never makes an attempt eligible; no-feedback loss has zero gradient."""
    peer, available = impl.select_feedback(torch.tensor([0,1,2]),torch.tensor([True,False,False]),torch.zeros(3,dtype=torch.bool))
    assert peer.tolist() == [0,-1,-1] and available.tolist() == [True,False,False]
    _, available = impl.select_feedback(torch.tensor([0,1,2]),torch.zeros(3,dtype=torch.bool),torch.zeros(3,dtype=torch.bool))
    assert not available.any()
    s = torch.randn(3,2,4,dtype=torch.float64,requires_grad=True)
    loss = impl.feedback_reverse_kl(s,-s.detach(),torch.ones(3,2,dtype=torch.bool),available)
    assert loss.item() == 0
    close(torch.autograd.grad(loss,s)[0],torch.zeros_like(s))


@pytest.mark.t2
def test_analytic_reverse_kl_gradient_and_teacher_stop(impl):
    """For valid positions dKL/dlogit = p*(log(p/q)-KL); other positions get no gradient."""
    s = torch.tensor([[[.2,-.5,1.]],[[.1,.8,-.3]]],dtype=torch.float64,requires_grad=True)
    t = torch.tensor([[[1.,.2,-.5]],[[.7,-.1,.4]]],dtype=s.dtype,requires_grad=True)
    loss = impl.feedback_reverse_kl(s,t,torch.ones(2,1,dtype=torch.bool),torch.tensor([True,False]))
    gs,gt = torch.autograd.grad(loss,(s,t),allow_unused=True)
    p,q = s.detach()[0,0].softmax(-1),t.detach()[0,0].softmax(-1)
    ratio = (p/q).log(); expected = torch.zeros_like(s); expected[0,0] = p*(ratio-(p*ratio).sum())
    close(gs,expected,atol=1e-12,rtol=1e-12); assert gt is None


@pytest.mark.t2
def test_inputs_preserved_and_masked_values_irrelevant(impl):
    """Masked teacher/student positions cannot affect either the loss or input values."""
    s = torch.randn(2,3,4,dtype=torch.float64); t = torch.randn_like(s)
    mask = torch.tensor([[True,False,False],[False,False,False]]); available = torch.tensor([True,False])
    before = s.clone(),t.clone()
    value = impl.feedback_reverse_kl(s,t,mask,available)
    close(s,before[0]); close(t,before[1])
    s[~mask] = 700.; t[~mask] = -900.
    close(value,impl.feedback_reverse_kl(s,t,mask,available))


@pytest.mark.t3
def test_tiny_program_feedback_training(impl, artifact):
    """Unit-test feedback guides only eligible program attempts using the submitted functions."""
    result = run(impl.select_feedback,impl.feedback_reverse_kl)
    artifact('program_feedback_training',result)
    assert result['metrics']['feedback_count'] == 5
    assert result['metrics']['final_correct_probability'] > .86
    assert result['metrics']['unavailable_logit_change'] == 0
