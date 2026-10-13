"""Two full 500-particle / 60-frame trajectories in compact signed-int16 form."""
from pathlib import Path
import base64
import importlib.util
import torch
from paperleet.common.toydata import GaussianMixture
from paperleet.common.oracles import gmm_eps_oracle


def _pack(tensor):
    arr=(tensor.detach().cpu()*1000).round().clamp(-32768,32767).to(torch.int16).numpy().astype('<i2')
    return {'encoding':'i16-base64','shape':list(arr.shape),'scale':1000,'data':base64.b64encode(arr.tobytes()).decode('ascii')}


def generate():
    spec=importlib.util.spec_from_file_location('ddpm_viz_reference',Path(__file__).parents[1]/'solution.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(2)
    b=torch.linspace(1e-4,.02,1000,dtype=torch.float64);ab=(1-b).cumprod(0)
    a=torch.arange(8,dtype=torch.float64)*torch.pi/4;centers=2*torch.stack((a.cos(),a.sin()),-1)
    g=GaussianMixture(centers,.15);oracle=gmm_eps_oracle(g,ab)
    indices=torch.linspace(0,1000,60).round().long();results={}
    for mode in ('posterior','beta'):
        _,trajectory=module.ddpm_sample(oracle,(500,2),b,torch.Generator().manual_seed(42),mode,True)
        results[mode]={'frames':_pack(trajectory[indices]),'times':(1000-indices).tolist()}
    prev=torch.cat((torch.ones_like(ab[:1]),ab[:-1]));variance=b*(1-prev)/(1-ab)
    return {'id':'diffusion/ddpm_sampling','seed':42,'description':'播放或拖动反向扩散时间；切换两种方差。轨迹量化误差≤0.0005。',
            'reverse_particles':results,'variance_compare':{'betas':b.tolist(),'posterior_variance':variance.tolist()},
            'centers':centers.tolist(),'data_std':.15,'alpha_bars':ab.tolist()}
