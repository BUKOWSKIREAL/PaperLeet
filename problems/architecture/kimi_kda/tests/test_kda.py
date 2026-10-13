import math
import pytest
import torch
from paperleet.common.asserts import assert_close_located as close
from problems.architecture.kimi_kda.experiment import run


def inputs(gradients=False):
    rng = torch.Generator().manual_seed(319)
    shapes = [(2,5,2,3), (2,5,2,3), (2,5,2,4), (2,5,2,3), (2,5,2), (2,2,3,4)]
    tensors = [torch.randn(shape,generator=rng,dtype=torch.float64) for shape in shapes]
    tensors[1] = torch.nn.functional.normalize(tensors[1],dim=-1)
    tensors[3] = tensors[3].sigmoid()
    tensors[4] = tensors[4].sigmoid()
    return [x.requires_grad_(gradients) for x in tensors]


def matrix_oracle(q,k,v,alpha,beta,state):
    # Literal matrix Eq.1 per batch/head; independent of vector residual implementation.
    batches, finals = [], []
    for b in range(q.shape[0]):
        per_head, head_states = [], []
        for h in range(q.shape[2]):
            s, outputs = state[b,h], []
            for t in range(q.shape[1]):
                key = k[b,t,h]
                transition = (torch.eye(key.numel(),dtype=q.dtype)-beta[b,t,h]*torch.outer(key,key)) @ torch.diag(alpha[b,t,h])
                s = transition @ s + beta[b,t,h]*torch.outer(key,v[b,t,h])
                outputs.append(q[b,t,h] @ s)
            per_head.append(torch.stack(outputs))
            head_states.append(s)
        batches.append(torch.stack(per_head,dim=1))
        finals.append(torch.stack(head_states))
    return torch.stack(batches),torch.stack(finals)


@pytest.mark.t0
def test_contract_empty_and_input_preservation(impl):
    """K!=V, empty sequence continuation, dtype/device and caller inputs are preserved."""
    tensors=inputs()
    before=[x.clone() for x in tensors]
    out,state=impl.recurrent_kda(*tensors)
    assert out.shape==(2,5,2,4) and state.shape==(2,2,3,4)
    assert out.dtype==state.dtype==torch.float64 and out.device==tensors[0].device
    for x,y in zip(tensors,before): close(x,y)
    empty=[x[:,:0] for x in tensors[:5]]
    out,state=impl.recurrent_kda(*empty,tensors[-1])
    assert out.shape==(2,0,2,4)
    close(state,tensors[-1])
    out,state=impl.recurrent_kda(*empty)
    close(state,torch.zeros_like(tensors[-1]))


@pytest.mark.t1
def test_matrix_equation_oracle(impl):
    """Channel-wise decay precedes delta correction and the output reads the updated state."""
    tensors=inputs()
    actual=impl.recurrent_kda(*tensors)
    expected=matrix_oracle(*tensors)
    for x,y in zip(actual,expected): close(x,y,atol=1e-10,rtol=1e-10)


@pytest.mark.t2
def test_bounded_gate_and_gradients(impl):
    """K3 bounds log-decay and places exp(A) inside sigmoid; both gate inputs differentiate."""
    z=torch.tensor([[[[-100.,-2.,0.,3.,100.],[4.,1.,-1.,-4.,0.]]]],dtype=torch.float64,requires_grad=True)
    a=torch.tensor([-.7,.8],dtype=z.dtype,requires_grad=True)
    out=impl.bounded_decay(z,a,lower_bound=-3.)
    expected=torch.tensor([[[[math.exp(-3/(1+math.exp(-math.exp(float(a[h]))*float(value)))) for value in z[0,0,h]] for h in range(2)]]],dtype=z.dtype)
    close(out,expected,atol=1e-12,rtol=1e-12)
    assert (out>=math.exp(-3.)-1e-15).all() and (out<=1).all()
    assert out.shape==z.shape and out.dtype==z.dtype
    small_z=z.detach().clamp(-3,3).requires_grad_()
    assert torch.autograd.gradcheck(lambda zz,aa:impl.bounded_decay(zz,aa), (small_z,a), eps=1e-6,atol=1e-5)


@pytest.mark.t2
def test_all_state_and_sequence_gradients(impl):
    """Autograd reaches q/k/v, channel retention, write gates and initial state."""
    tensors=inputs(gradients=True)
    independent=[x.detach().clone().requires_grad_() for x in tensors]
    out,state=impl.recurrent_kda(*tensors)
    expected,final=matrix_oracle(*independent)
    objective=out.square().sum()+.3*state.square().sum()
    oracle_objective=expected.square().sum()+.3*final.square().sum()
    for x,y in zip(torch.autograd.grad(objective,tensors),torch.autograd.grad(oracle_objective,independent)):
        close(x,y,atol=1e-8,rtol=1e-8)


@pytest.mark.t2
def test_segmented_continuation_and_causality(impl):
    """Resumed decoding equals one scan and future tokens never affect the prefix."""
    tensors=inputs()
    whole,final=impl.recurrent_kda(*tensors)
    left,memory=impl.recurrent_kda(*[x[:,:2] for x in tensors[:5]],tensors[-1])
    right,resumed=impl.recurrent_kda(*[x[:,2:] for x in tensors[:5]],memory)
    close(torch.cat([left,right],dim=1),whole)
    close(resumed,final)
    changed=[x.clone() for x in tensors]
    changed[2][:,2:]+=100
    altered,_=impl.recurrent_kda(*changed)
    close(altered[:,:2],whole[:,:2])


@pytest.mark.t2
def test_zero_write_and_noncontiguous_inputs(impl):
    """Zero write strength only decays memory, including strided tensors and unequal channels."""
    tensors=[x.transpose(-1,-2).contiguous().transpose(-1,-2) for x in inputs()]
    tensors[4]=torch.zeros_like(tensors[4])
    out,state=impl.recurrent_kda(*tensors)
    expected,final=matrix_oracle(*tensors)
    close(out,expected,atol=1e-10,rtol=1e-10)
    close(state,final,atol=1e-10,rtol=1e-10)
    expected_final=tensors[-1]*tensors[3].prod(dim=1)[...,None]
    close(state,expected_final)


@pytest.mark.t3
def test_gate_curves_and_memory_overwrite(impl,artifact):
    """Figure3 gate curves and the separate memory toy execute the submitted functions."""
    result=run(impl.bounded_decay,impl.recurrent_kda)
    artifact('bounded_decay_and_memory',result)
    assert result['metrics']['gate_maximum_error']<1e-12
    assert result['metrics']['memory_maximum_error']<1e-12
    assert result['metrics']['final_state_error']<1e-12
