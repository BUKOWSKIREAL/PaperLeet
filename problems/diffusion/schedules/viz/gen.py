"""Seed data for the two schedule views; coefficients recompute in the browser."""
import torch
from paperleet.common.toydata import eight_gaussians


def generate():
    generator=torch.Generator().manual_seed(42)
    points=eight_gaussians(2000,generator=generator)
    noise=torch.randn(points.shape,generator=generator)
    angles=torch.arange(8)*torch.pi/4
    def rounded(x):return (x*1000).round().div(1000).tolist()
    return {'id':'diffusion/schedules','seed':42,'description':'拖动时间，比较线性和 cosine 调度的信号与点云结构。',
            'points':rounded(points),'noise':rounded(noise),'centers':rounded(2*torch.stack((angles.cos(),angles.sin()),-1)),
            'defaults':{'T':1000,'s':.008,'max_beta':.999}}
