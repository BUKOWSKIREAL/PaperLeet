import torch,copy


def run(impl,steps=350):
    torch.manual_seed(42)
    net=torch.nn.Sequential(torch.nn.Linear(3,48),torch.nn.SiLU(),torch.nn.Linear(48,48),torch.nn.SiLU(),torch.nn.Linear(48,2))
    target=copy.deepcopy(net);target.requires_grad_(False);opt=torch.optim.Adam(net.parameters(),lr=.004);goal=torch.tensor([.7,-.5]);curve=[]
    online=lambda x,s:net(torch.cat((x,s[:,None]),1))
    ema=lambda x,s:target(torch.cat((x,s[:,None]),1))
    for step in range(steps):
        sigma=.1+.9*torch.rand(64);low=sigma*.5;low[:32]=.002;x=goal+sigma[:,None]*torch.randn(64,2)
        # Point-data probability-flow ODE is exactly linear; Euler reaches the same path at low.
        lower=impl.teacher_euler(x,goal.expand_as(x),sigma,low)
        pred=impl.consistency_function(online,x,sigma)
        with torch.no_grad():teacher=impl.consistency_function(ema,lower,low)
        opt.zero_grad();loss=impl.consistency_loss(pred,teacher);loss.backward();opt.step();impl.update_ema(net,target,.95)
        if step%20==0:curve.append([step,float(loss.detach())])
    x=goal+torch.randn(128,2)
    with torch.no_grad():
        samples=impl.consistency_function(ema,x,torch.ones(128));boundary=impl.consistency_function(online,x,torch.full((128,),.002))
    return {'samples':samples.tolist(),'reference_samples':goal.expand_as(samples).tolist(),'loss_curve':curve,
            'metrics':{'endpoint_mse':float((samples-goal).square().mean()),'boundary_error':float((boundary-x).abs().max()),'nonboundary_target_fraction':.5}}
