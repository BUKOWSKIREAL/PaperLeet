"""Controlled outlier experiment; CPU fake quantization, no speed claims."""
import torch


def _q(x):
    step=x.abs().max()/7
    return torch.zeros_like(x) if step==0 else torch.round(x/step).clamp(-7,7)*step


def run(smooth_inputs, split_weight, svdquant_linear):
    g=torch.Generator().manual_seed(42)
    scale=torch.tensor([30.,1.,1.,1.,1.,1.,1.,1.],dtype=torch.float64)
    x=torch.randn(64,8,generator=g,dtype=torch.float64)*scale
    weight=torch.randn(8,6,generator=g,dtype=torch.float64)
    exact=x@weight
    xs,ws=smooth_inputs(x,weight,scale)
    ranks=list(range(7));errors=[]
    for rank in ranks:
        actual=svdquant_linear(x,weight,scale,rank)
        errors.append(float((actual-exact).norm()/exact.norm()))
    left,right,residual=split_weight(ws,1)
    naive=float((_q(x)@_q(weight)-exact).norm()/exact.norm())
    smoothing=float((_q(xs)@_q(ws)-exact).norm()/exact.norm())
    return dict(kind='toy_experiment',title='Move channel outliers into a low-rank branch',problem='finetuning/svdquant',
                provenance=dict(kind='teaching_adaptation',source='https://arxiv.org/html/2411.05007v1',anchor='Section 4.2, Eq.5 / Figures 4-5',
                                adaptation='Synthetic 64x8 activations with one 30x channel and 8x6 weights; supplied smoothing, one-shot SVD and per-tensor INT4 fake quantization. No PixArt/FLUX checkpoint, calibration search, iterative refinement, FP4 or Nunchaku kernels.'),
                metrics=dict(naive_relative_error=naive,smoothing_only_relative_error=smoothing,rank1_relative_error=errors[1],
                             full_rank_relative_error=errors[-1],smoothing_identity_error=float((xs@ws-exact).abs().max()),
                             weight_absmax=float(ws.abs().max()),residual_absmax=float(residual.abs().max())),
                charts=[dict(kind='line',title='Relative output error versus retained rank',x_label='Rank',x=ranks,
                             series=[dict(name='SVDQuant numerical core',values=errors),dict(name='Naive W4A4',values=[naive]*7),dict(name='Smoothing only W4A4',values=[smoothing]*7)]),
                        dict(kind='line',title='Smoothed weight and rank-one residual singular values',x_label='Singular value index',x=list(range(1,7)),
                             series=[dict(name='Smoothed weight',values=torch.linalg.svdvals(ws).tolist()),dict(name='Residual',values=torch.linalg.svdvals(residual).tolist())])])
