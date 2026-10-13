import torch


def run(impl,steps=180):
    torch.manual_seed(42)
    net=torch.nn.Linear(2,2);optimizer=torch.optim.SGD(net.parameters(),lr=.12)
    teacher=lambda x,t:.35*x+torch.tensor([.6,-.3])
    a=lambda v:(v*torch.pi/2).cos();s=lambda v:(v*torch.pi/2).sin()
    def target_for(x):
        t=x.new_full((len(x),),.8);m=t*.5;e=t*.25
        endpoint=impl.teacher_pair(teacher,x,t,m,e)
        target=impl.distilled_target(x,endpoint,a(t),s(t),a(e),s(e))
        return target,endpoint,t,e
    evalx=torch.randn(128,2);t=evalx.new_full((len(evalx),),.8);m=t*.5;e=t*.25
    # Independent held-out oracle never invokes learner stepping/target functions.
    def teacher_step(x,high,low):
        pred=teacher(x,high)
        epsilon=(x-a(high)[:,None]*pred)/s(high)[:,None]
        return a(low)[:,None]*pred+s(low)[:,None]*epsilon
    endpoint=teacher_step(teacher_step(evalx,t,m),m,e)
    with torch.no_grad():initial=float((impl.ddim_step(evalx,net(evalx),a(t),s(t),a(e),s(e))-endpoint).square().mean())
    curve=[]
    for step in range(steps):
        x=torch.randn(64,2);target,_,_,_=target_for(x)
        optimizer.zero_grad();loss=impl.distillation_loss(net(x),target,torch.ones(len(x)));loss.backward();optimizer.step()
        if step%10==0:curve.append([step,float(loss.detach())])
    with torch.no_grad():samples=impl.ddim_step(evalx,net(evalx),a(t),s(t),a(e),s(e))
    return {'samples':samples.tolist(),'reference_samples':endpoint.tolist(),'loss_curve':curve,
            'metrics':{'initial_endpoint_mse':initial,'endpoint_mse':float((samples-endpoint).square().mean()),'teacher_calls_per_sample':2,'student_calls_per_sample':1}}
