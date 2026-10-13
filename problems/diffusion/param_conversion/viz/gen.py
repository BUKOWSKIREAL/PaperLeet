"""Small geometric anchors; all interactive coefficients recompute in TypeScript."""
from pathlib import Path
import importlib.util
import torch


def generate():
    path=Path(__file__).parents[1]/'solution.py'
    spec=importlib.util.spec_from_file_location('conversion_viz_reference',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    phi=torch.linspace(0,torch.pi/2,65,dtype=torch.float64)
    x0=torch.tensor([1.,0.],dtype=torch.float64);eps=torch.tensor([0.,1.],dtype=torch.float64)
    a,s=phi.cos()[:,None],phi.sin()[:,None]
    return {'id':'diffusion/param_conversion','seed':42,'description':'拖动 φ，观察圆上的位置与 v 切向量；系数是 sqrt(alpha_bar)。',
            'angles':phi.tolist(),'positions':(a*x0+s*eps).tolist(),'velocities':module.make_v(x0,eps,a,s).tolist()}
