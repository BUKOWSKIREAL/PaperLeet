import torch


def run(impl,steps=1000):
    torch.manual_seed(42)
    mean=torch.nn.Parameter(torch.zeros(2));fake_mean=torch.nn.Parameter(torch.zeros(2));goal=torch.tensor([1.2,-.8])
    disc=torch.nn.Linear(2,1);optc=torch.optim.SGD([fake_mean,*disc.parameters()],lr=.08);optg=torch.optim.SGD([mean],lr=.12)
    curve=[];updates=0
    for count in range(1,steps+1):
        real=torch.randn(64,2)+goal;fake=torch.randn(64,2)+mean.detach();noise=torch.randn_like(fake);noisy=fake+.5*noise
        fake_score=-(noisy-fake_mean)/1.25
        score_loss=(.5*fake_score+noise).square().mean()
        dloss,_=impl.adversarial_losses(disc(real).flatten(),disc(fake).flatten())
        optc.zero_grad();(score_loss+.1*dloss).backward();optc.step()
        if impl.generator_due(count,5):
            generated=torch.randn(64,2)+mean
            with torch.no_grad():
                y=generated+.5*torch.randn_like(generated)
                gradient=(-(y-fake_mean)/1.25+(y-goal)/1.25)
            disc.requires_grad_(False)
            optg.zero_grad();loss=impl.generator_loss(generated,gradient,disc(generated).flatten(),.1);loss.backward();optg.step()
            disc.requires_grad_(True);updates+=1
            if updates%10==0:curve.append([updates,float(loss.detach())])
    z=torch.randn(128,2)
    return {'samples':(z+mean.detach()).tolist(),'reference_samples':(torch.randn(128,2)+goal).tolist(),'loss_curve':curve,
            'metrics':{'mean_error':float((mean.detach()-goal).norm()),'critic_updates':steps,'generator_updates':updates}}
