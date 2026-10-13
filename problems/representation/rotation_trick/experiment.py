"""Paper-derived quadratic illustration with a stated sampling adaptation."""
import math
import torch


def run(rotation_quantize):
    axis = torch.linspace(.07, 2*math.pi+.07, 65, dtype=torch.float64)
    e = torch.stack((1.8*axis.cos(), 1.8*axis.sin()),dim=-1).requires_grad_()
    # Sixteen vectors as in the paper's illustrative setup; our grid and circle are chosen here.
    coords = torch.tensor([-1.5,-.5,.5,1.5],dtype=e.dtype)
    codebook = torch.cartesian_prod(coords,coords)
    q = codebook[(e.detach()[:,None,:]-codebook[None,:,:]).square().sum(-1).argmin(-1)]
    output = rotation_quantize(e,q)
    grad, = torch.autograd.grad(output.square().sum(),e)
    ste_grad = 2*q
    grad_norm, ste_norm = grad.norm(dim=-1), ste_grad.norm(dim=-1)
    cosine = (grad*e).sum(-1)/(grad_norm*e.norm(dim=-1))
    ste_cos = (ste_grad*e).sum(-1)/(ste_norm*e.norm(dim=-1))
    expected_norm = 2*q.square().sum(-1)/e.detach().norm(dim=-1)
    return {'kind':'toy_experiment','title':'Quadratic gradients through quantization',
            'problem':'representation/rotation_trick',
            'provenance':{'kind':'paper_derived','source':'https://arxiv.org/html/2410.06424v1',
                          'anchor':'Figures 2–3, Algorithm 1 and Section 4.2',
                          'adaptation':'Use the original quadratic f(x,y)=x²+y² and 16 codes, with an independently chosen 4×4 grid and circular sampling. Plot gradient angle and norm; no VQ-VAE training claim.'},
            'metrics':{'forward_error':float((output-q).abs().max().detach()),
                       'angle_preservation_error':float((cosine-1).abs().max().detach()),
                       'gradient_norm_error':float((grad_norm-expected_norm).abs().max().detach()),
                       'ste_max_angle_error':float((ste_cos-1).abs().max().detach())},
            'charts':[{'kind':'line','title':'Alignment of encoder gradient with encoder vector','x_label':'Input angle (radians)',
                       'x':axis.tolist(),'series':[{'name':'Rotation trick cosine','values':cosine.detach().tolist()},
                                                 {'name':'STE cosine','values':ste_cos.detach().tolist()}]},
                      {'kind':'line','title':'Encoder gradient magnitude','x_label':'Input angle (radians)',
                       'x':axis.tolist(),'series':[{'name':'Rotation trick','values':grad_norm.detach().tolist()},
                                                 {'name':'STE','values':ste_norm.detach().tolist()},
                                                 {'name':'Analytic scaled norm','values':expected_norm.tolist()}]}]}
