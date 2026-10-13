import torch


def run(impl, steps=160, dimension=16, width=32, depth=3):
    torch.manual_seed(42)
    projection=torch.linalg.qr(torch.randn(dimension,2)).Q
    angle=torch.rand(512)*4*torch.pi
    data=torch.stack((angle.cos(),angle.sin()),1)*(angle[:,None]/(4*torch.pi)+.2)
    embedded=data@projection.T
    curves=[];metrics={'projection_orthogonality_error':float((projection.T@projection-torch.eye(2)).abs().max())};all_samples={}
    eval_noise=torch.randn(512,dimension);eval_t=torch.full((512,),.5)
    for kind in ('x','eps','v'):
        torch.manual_seed(123)
        layers=[torch.nn.Linear(dimension+1,width),torch.nn.ReLU()]
        for _ in range(depth-2):layers.extend((torch.nn.Linear(width,width),torch.nn.ReLU()))
        layers.append(torch.nn.Linear(width,dimension));net=torch.nn.Sequential(*layers)
        optimizer=torch.optim.Adam(net.parameters(),lr=.004)
        z=impl.make_noisy(embedded,eval_noise,eval_t)
        with torch.no_grad():initial=float(impl.velocity_loss(net(torch.cat((z,eval_t[:,None]),1)),embedded,eval_noise,eval_t,kind))
        for step in range(steps):
            x=embedded[torch.randint(len(embedded),(64,))];e=torch.randn_like(x);t=.05+.9*torch.rand(64)
            z=impl.make_noisy(x,e,t);pred=net(torch.cat((z,t[:,None]),1))
            optimizer.zero_grad();loss=impl.velocity_loss(pred,x,e,t,kind);loss.backward();optimizer.step()
            if kind=='x' and step%10==0:curves.append([step,float(loss.detach())])
        with torch.no_grad():
            z=impl.make_noisy(embedded,eval_noise,eval_t);pred=net(torch.cat((z,eval_t[:,None]),1))
            final=float(impl.velocity_loss(pred,embedded,eval_noise,eval_t,kind))
            generated=torch.randn(128,dimension)
            for k in range(40):
                t=torch.full((len(generated),),k/40)
                p=net(torch.cat((generated,t[:,None]),1))
                generated=generated+impl.prediction_to_velocity(p,generated,t,kind)/40
            all_samples[kind]=(generated@projection).tolist()
        metrics[kind+'_initial_loss']=initial;metrics[kind+'_final_loss']=final
    return {'samples':all_samples['x'],'reference_samples':data[:128].tolist(),'loss_curve':curves,'comparison_samples':all_samples,'metrics':metrics}


if __name__=='__main__':
    import argparse,json,importlib.util
    p=argparse.ArgumentParser();p.add_argument('--impl',required=True);p.add_argument('--full',action='store_true');p.add_argument('--steps',type=int,default=None);args=p.parse_args()
    s=importlib.util.spec_from_file_location('learner',args.impl);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    torch.set_num_threads(2)
    dimensions=(2,8,16,512) if args.full else (16,)
    print(json.dumps({str(d):run(m,steps=args.steps or (5000 if args.full else 160),dimension=d,width=256 if args.full else 32,depth=5 if args.full else 3) for d in dimensions}))
