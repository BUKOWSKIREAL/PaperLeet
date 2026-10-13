"""Train the reference model, then record its actual flow and vector field."""
from pathlib import Path
import base64
import importlib.util
import torch
import yaml
from paperleet.common.models import TimeMLP
from paperleet.common.toydata import eight_gaussians
from paperleet.common.oracles import point_velocity_oracle,gaussian_velocity_oracle


def _pack(tensor):
    arr=(tensor.detach().cpu()*1000).round().clamp(-32768,32767).to(torch.int16).numpy().astype('<i2')
    return {'encoding':'i16-base64','shape':list(arr.shape),'scale':1000,'data':base64.b64encode(arr.tobytes()).decode('ascii')}


def rounded(x):return (x.detach()*1000).round().div(1000).tolist()


def generate():
    spec=importlib.util.spec_from_file_location('flow_viz_reference',Path(__file__).parents[1]/'solution.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.manual_seed(42);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    config=yaml.safe_load((Path(__file__).parents[1]/'problem.yaml').read_text())['calibration']
    model=TimeMLP();opt=torch.optim.Adam(model.parameters(),lr=.001);gen=torch.Generator().manual_seed(42);curve=[]
    for step in range(config['training_steps']):
        x1=eight_gaussians(256,generator=gen);x0=torch.randn(256,2,generator=gen);t=module.sample_t(256,gen)
        opt.zero_grad(set_to_none=True);loss=module.cfm_loss(model,x1,x0,t);loss.backward();opt.step()
        if step%25==0:curve.append([step,round(float(loss.detach()),5)])
    initial=torch.randn(1000,2,generator=torch.Generator().manual_seed(999))
    samples,trajectory=module.euler_sample(model,initial,59,True)
    side=torch.linspace(-3,3,20);grid=torch.cartesian_prod(side,side);fields=[]
    with torch.no_grad():
        for t in torch.linspace(0,.98,8):
            fields.append({'t':float(t),'vectors':_pack(model(grid,torch.full((400,),float(t))))})
        wrong=initial.clone()
        for k in range(50):
            wrong=wrong-model(wrong,torch.full((len(wrong),),1-k/50))/50
    angle=torch.arange(8)*torch.pi/4
    return {'id':'diffusion/flow_matching','seed':42,'description':'真实2000步训练；播放1000个粒子的流与20×20速度场。loss的非零终值来自条件目标方差。',
            'centers':rounded(2*torch.stack((angle.cos(),angle.sin()),-1)),
            'flow_particles':{'frames':_pack(trajectory),'positions':rounded(grid),'fields':fields},
            'steps_compare':[{'steps':n,'samples':rounded(module.euler_sample(model,initial[:200],n))} for n in (1,2,4,8,50)],
            'convention_bug':{'correct':rounded(samples[:500]),'reversed':rounded(wrong[:500])},
            'oracle_trajectories':{'point':_pack(module.euler_sample(point_velocity_oracle(torch.tensor([2.,0.])),initial[:20],50,True)[1]),
                                   'gaussian':_pack(module.euler_sample(gaussian_velocity_oracle(torch.tensor([2.,0.]),.4),initial[:20],50,True)[1])},
            'loss_curve':curve,'training_steps':config['training_steps']}
