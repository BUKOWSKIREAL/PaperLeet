import torch


def run(impl,steps=220):
    torch.manual_seed(42)
    mean=torch.nn.Parameter(torch.zeros(2));fake_mean=torch.nn.Parameter(torch.zeros(2));goal=torch.tensor([1.2,-.8])
    optg=torch.optim.SGD([mean],lr=.08);optf=torch.optim.SGD([fake_mean],lr=.2)
    curve=[]
    for step in range(steps):
        z=torch.randn(128,2);eps=torch.randn_like(z);sigma=torch.full((len(z),),.5)
        generated=z+mean
        model=lambda noisy,s:-(noisy-fake_mean)/(1+s[:,None].square())
        optf.zero_grad();floss=impl.fake_score_loss(model,generated,eps,sigma);floss.backward();optf.step()
        generated=z+mean;noisy=generated.detach()+sigma[:,None]*eps
        real_score=-(noisy-goal)/1.25;fake_score=model(noisy,sigma)
        gradient=impl.score_difference(real_score,fake_score,torch.ones(len(z)))
        optg.zero_grad();loss=impl.dmd_loss(generated,gradient,z+goal,.1);loss.backward();optg.step()
        if step%10==0:curve.append([step,float(loss.detach())])
    z=torch.randn(128,2)
    return {'samples':(z+mean.detach()).tolist(),'reference_samples':(z+goal).tolist(),'loss_curve':curve,
            'metrics':{'mean_error':float((mean.detach()-goal).norm()),'fake_mean_tracking_error':float((fake_mean.detach()-mean.detach()).norm())}}


def run_bimodal(impl, steps=600):
    """Figure-3-inspired adaptation: real-score-only versus DM versus DM+reg.

    Both distributions are analytic two-component Gaussian mixtures. This is an
    illustrative distribution-level experiment, not the original image pipeline.
    """
    def mixture_score(points, centers, variance):
        delta=points[:,None,:]-centers[None,:,:]
        responsibility=(-delta.square().sum(-1)/(2*variance)).softmax(1)
        return -(responsibility[:,:,None]*delta).sum(1)/variance
    centers=torch.tensor([[-1.8,0.],[1.8,0.]])
    comparison={};curves={};metrics={}
    for strategy in ('real_score_only','matching','matching_regression'):
        torch.manual_seed(42)
        means=torch.nn.Parameter(torch.tensor([[.2,-.1],[.4,.1]]))
        optimizer=torch.optim.SGD([means],lr=.15);curve=[]
        for step in range(steps):
            label=torch.arange(128)%2;z=.25*torch.randn(128,2);generated=means[label]+z
            sigma=1.5-.9*step/max(steps-1,1)
            noisy=generated.detach()+sigma*torch.randn_like(generated)
            real=mixture_score(noisy,centers,.25**2+sigma**2)
            fake=mixture_score(noisy,means.detach(),.25**2+sigma**2)
            gradient=-real if strategy=='real_score_only' else impl.score_difference(real,fake,torch.ones(len(z)))
            regression=.1 if strategy=='matching_regression' else 0.
            optimizer.zero_grad();loss=impl.dmd_loss(generated,gradient,centers[label]+z,regression);loss.backward();optimizer.step()
            if step%40==0:curve.append([step,float(loss.detach())])
        with torch.no_grad():
            label=torch.arange(256)%2;samples=means[label]+.25*torch.randn(256,2)
            distance=torch.cdist(samples,centers);nearest=distance.argmin(1);inside=distance.min(1).values<.75
            proportions=[float(((nearest==i)&inside).float().mean()) for i in range(2)]
            coverage=sum(value>.1 for value in proportions)
        comparison[strategy]=samples.tolist();curves[strategy]=curve
        metrics[strategy+'_covered_modes']=coverage
        metrics[strategy+'_left_fraction']=proportions[0];metrics[strategy+'_right_fraction']=proportions[1]
        metrics[strategy+'_center_error']=float(torch.cdist(means.detach(),centers).min(0).values.mean())
    torch.manual_seed(123);reference=centers[torch.arange(256)%2]+.25*torch.randn(256,2)
    return {'samples':comparison['matching_regression'],'reference_samples':reference.tolist(),'comparison_samples':comparison,
            'loss_curve':curves['matching_regression'],'comparison_loss_curves':curves,'metrics':metrics}
