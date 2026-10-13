"""Teaching adaptation: measure cancellation of a shared value offset."""
import math
import torch


def run(differential_lambda,differential_attention):
    dtype=torch.float64
    q1=torch.tensor([[[[2.,0.]]*4]],dtype=dtype)
    k1=torch.tensor([[[[1.,0.],[0.,0.],[-1.,0.],[0.,0.]]]],dtype=dtype)
    q2,k2=torch.zeros_like(q1),torch.zeros_like(k1)
    value=torch.tensor([[[[2.,1.,0.,0.],[0.,1.,0.,0.],[0.,1.,0.,0.],[0.,1.,0.,0.]]]],dtype=dtype)
    offset=torch.tensor([10.,0.,0.,0.],dtype=dtype)
    coefficients=torch.linspace(0.,1.,21,dtype=dtype)
    signal,noise,expected=[],[],[]
    with torch.no_grad():
        for coefficient in coefficients:
            base=differential_attention(q1,k1,q2,k2,value,coefficient,False)
            perturbed=differential_attention(q1,k1,q2,k2,value+offset,coefficient,False)
            signal.append(float(base[0,0,0,0]))
            noise.append(float((perturbed-base)[0,0,0,0]))
            expected.append(10*(1-float(coefficient)))
        z=torch.zeros(2,dtype=dtype)
        initialized=float(differential_lambda(z,z,z,z,.2))
    return {'kind':'toy_experiment','title':'Common-mode cancellation in differential attention',
            'problem':'architecture/differential_attention',
            'provenance':{'kind':'teaching_adaptation','source':'https://arxiv.org/html/2410.05258v1#S2.SS1',
                          'anchor':'Section 2.1, Eqs. (1)–(2)',
                          'adaptation':'A constructed four-token sequence measures the exact response to a constant value offset; not a reproduction of language-model experiments or a universal noise-reduction claim.'},
            'metrics':{'common_mode_formula_error':max(abs(a-b) for a,b in zip(noise,expected)),
                       'residual_at_lambda_one':abs(noise[-1]),'signal_at_lambda_one':signal[-1],
                       'lambda_initialization_error':abs(initialized-.2)},
            'charts':[{'kind':'line','title':'Response to shared value noise','x_label':'Lambda','x':coefficients.tolist(),
                       'series':[{'name':'Submitted attention response','values':noise},{'name':'10 × (1 − lambda)','values':expected}]},
                      {'kind':'line','title':'Signal output before normalization','x_label':'Lambda','x':coefficients.tolist(),
                       'series':[{'name':'Signal retained by differing maps','values':signal}]}]}
