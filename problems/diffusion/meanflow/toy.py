import torch


def run(impl, steps=160):
    torch.manual_seed(42)
    center=torch.nn.Parameter(torch.zeros(2));target=torch.tensor([1.2,-.7])
    optimizer=torch.optim.SGD([center],lr=.04)
    def model(z,r,t):return (z-center)/t[:,None]
    curve=[]
    for step in range(steps):
        noise=torch.randn(64,2);clean=target.expand_as(noise)
        t=.4+.6*torch.rand(64);r=t*torch.rand(64)
        optimizer.zero_grad();loss=impl.meanflow_loss(model,clean,noise,r,t);loss.backward();optimizer.step()
        if step%10==0:curve.append([step,float(loss.detach())])
    noise=torch.randn(100,2);samples=impl.one_step(model,noise)
    return {'samples':samples.tolist(),'reference_samples':target.expand_as(samples).tolist(),'loss_curve':curve,
            'metrics':{'endpoint_mse':float((samples-target).square().mean())}}
