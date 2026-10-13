"""A real eta-by-step experiment, inversion and same-noise trajectories."""
from pathlib import Path
import base64
import importlib.util
import torch
from paperleet.common.toydata import GaussianMixture
from paperleet.common.oracles import gmm_eps_oracle
from paperleet.common.metrics import mmd_rbf


def _pack(tensor):
    arr=(tensor.detach().cpu()*1000).round().clamp(-32768,32767).to(torch.int16).numpy().astype('<i2')
    return {'encoding':'i16-base64','shape':list(arr.shape),'scale':1000,'data':base64.b64encode(arr.tobytes()).decode('ascii')}


def rounded(x):return (x*1000).round().div(1000).tolist()


def generate():
    spec=importlib.util.spec_from_file_location('ddim_viz_reference',Path(__file__).parents[1]/'solution.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(2)
    ab=(1-torch.linspace(1e-4,.02,1000,dtype=torch.float64)).cumprod(0)
    a=torch.arange(8,dtype=torch.float64)*torch.pi/4;centers=2*torch.stack((a.cos(),a.sin()),-1)
    g=GaussianMixture(centers,.15);oracle=gmm_eps_oracle(g,ab)
    x=torch.randn(200,2,dtype=torch.float64,generator=torch.Generator().manual_seed(42))
    real=g.sample(200,torch.Generator().manual_seed(123));grid=[];traces={}
    for eta in (0.,.5,1.):
        for steps in (10,20,50,100):
            sample,trajectory=module.ddim_sample(oracle,x,ab,steps,eta,torch.Generator().manual_seed(42),True)
            indices=torch.linspace(0,steps,min(16,steps+1)).round().long()
            grid.append({'eta':eta,'steps':steps,'mmd':mmd_rbf(sample,real),'samples':rounded(sample),'frames':_pack(trajectory[indices])})
            if steps==50 and eta in (0.,1.):traces['deterministic' if eta==0 else 'stochastic']=_pack(trajectory[:,:30])
    original=g.sample(24,torch.Generator().manual_seed(77));inversions=[];errors=[]
    for steps in (10,20,50,100):
        latent=module.ddim_invert(oracle,original,ab,steps);reconstructed=module.ddim_sample(oracle,latent,ab,steps)
        errors.append(float((reconstructed-original).norm(dim=-1).mean()))
        inversions.append({'steps':steps,'original':rounded(original),'latent':rounded(latent),'reconstructed':rounded(reconstructed)})
    return {'id':'diffusion/ddim','seed':42,'description':'点击 eta×步数组合播放对应真实轨迹；反演面板比较不同网格的往返误差。',
            'centers':centers.tolist(),'eta_steps_grid':grid,'trajectories':traces,
            'inversion_demo':{'steps':[10,20,50,100],'errors':errors,'points':inversions}}
