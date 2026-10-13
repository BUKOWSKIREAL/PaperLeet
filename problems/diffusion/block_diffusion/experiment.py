"""Recompute the paper's six-token attention example and probe information flow."""
import torch


def run(block_diffusion_mask,block_diffusion_attention):
    length,block=6,2
    dtype=torch.float64
    index=torch.arange(24,dtype=dtype).reshape(1,1,12,2)
    q,k=torch.sin(index*.3),torch.cos(index*.4)
    value=torch.arange(12,dtype=dtype).reshape(1,1,12,1)/10
    with torch.no_grad():
        allowed=block_diffusion_mask(length,block)
        baseline=block_diffusion_attention(q,k,value,block)
        influences=[]
        for token in range(12):
            changed=value.clone();changed[:,:,token]+=1
            influences.append((block_diffusion_attention(q,k,changed,block)-baseline)[0,0,:,0])
        influence=torch.stack(influences,dim=-1)
        # Explicit original L=6, block=2 allowed-key lists, independent of submitted mask.
        keys=[[0,1],[0,1],[2,3,6,7],[2,3,6,7],[4,5,6,7,8,9],[4,5,6,7,8,9],
              [6,7],[6,7],[6,7,8,9],[6,7,8,9],[6,7,8,9,10,11],[6,7,8,9,10,11]]
        expected=torch.zeros(12,12,dtype=torch.bool)
        for row,columns in enumerate(keys):expected[row,columns]=True
    return {'kind':'toy_experiment','title':'Block diffusion: six-token information flow',
            'problem':'diffusion/block_diffusion',
            'provenance':{'kind':'paper_derived','source':'https://arxiv.org/html/2503.09573v2#A2.SS6',
                          'anchor':'Appendix B.6, Figure 3: L=6 and block size 2',
                          'adaptation':'Recompute the original mask dimensions; add deterministic tiny attention and individual-value perturbations to measure leakage. This is an attention dependency experiment, not diffusion model training.'},
            'metrics':{'mask_mismatches':int((allowed!=expected).sum()),
                       'maximum_forbidden_influence':float(influence[~expected].abs().max()),
                       'minimum_allowed_influence':float(influence[expected].min()),
                       'attention_row_sum_error':float((influence.sum(-1)-1).abs().max())},
            'charts':[{'kind':'line','title':'Allowed keys per query','x_label':'Concatenated query index: noisy 0–5, clean 6–11',
                       'x':list(range(12)),'series':[{'name':'Submitted mask count','values':allowed.sum(-1).tolist()},
                                                   {'name':'Original block dependency count','values':[len(x) for x in keys]}]},
                      {'kind':'line','title':'Influence of clean-token perturbations','x_label':'Concatenated query index',
                       'x':list(range(12)),'series':[{'name':'First clean token (index 6)','values':influence[:,6].tolist()},
                                                   {'name':'Last clean token (index 11)','values':influence[:,11].tolist()}]}]}
