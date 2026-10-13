import torch,math


def sample(n,kind):
    if kind=='swiss_roll':
        a=math.pi*(.5+4*torch.rand(n));x=torch.stack((a.cos(),a.sin()),1)*a[:,None]/(4.5*math.pi)
        return x+.03*torch.randn(n,2)
    parity=torch.randint(2,(n,1));cells=2*torch.randint(2,(n,2))+parity
    return (cells+torch.rand(n,2)-2)/2+.05*torch.randn(n,2)


def mmd(x,y):
    k=lambda a,b:torch.exp(-torch.cdist(a,b).square()/(2*.3**2)).mean()
    return float(k(x,x)+k(y,y)-2*k(x,y))


def run(impl,steps=260):
    metrics={};comparison={};curve=[];references={}
    for kind in ('swiss_roll','checkerboard'):
        torch.manual_seed(42)
        net=torch.nn.Sequential(torch.nn.Linear(8,48),torch.nn.SiLU(),torch.nn.Linear(48,48),torch.nn.SiLU(),torch.nn.Linear(48,2))
        opt=torch.optim.Adam(net.parameters(),lr=.002)
        evaluation_noise=torch.randn(256,8);real=sample(256,kind)
        metrics[kind+'_initial_mmd']=mmd(net(evaluation_noise).detach(),real)
        for step in range(steps):
            generated=net(torch.randn(96,8));positive=sample(128,kind)
            opt.zero_grad();loss=impl.drifting_loss(generated,positive,.2);loss.backward();opt.step()
            if kind=='swiss_roll' and step%20==0:curve.append([step,float(loss.detach())])
        with torch.no_grad():final=net(evaluation_noise)
        metrics[kind+'_final_mmd']=mmd(final,real);comparison[kind]=final.tolist();references[kind]=real.tolist()
    return {'samples':comparison['swiss_roll'],'reference_samples':references['swiss_roll'],'comparison_samples':comparison,
            'comparison_reference_samples':references,'loss_curve':curve,'metrics':metrics}
