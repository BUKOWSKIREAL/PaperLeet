"""Teaching adaptation: a learned pseudo-query selects a useful depth representation."""
import math
import torch


def run(assemble_sources, depth_attention):
    dtype = torch.float64
    embedding = torch.tensor([[[1., 0.]]], dtype=dtype)
    completed = [torch.tensor([[[-2., 0.]]], dtype=dtype)]
    partial = torch.tensor([[[0., 3.]]], dtype=dtype)
    sources = assemble_sources(embedding, completed, partial)
    strengths = torch.linspace(0, 6, 25, dtype=dtype)
    weights, errors, max_error = [], [], 0.
    for strength in strengths.tolist():
        output, attention = depth_attention(sources, torch.tensor([strength, 0.], dtype=dtype), torch.ones(2,dtype=dtype))
        # Independent scalar softmax over the explicitly constructed 3 sources.
        scores = [strength / math.sqrt(.5 + 1e-6), -2*strength / math.sqrt(2 + 1e-6), 0.]
        unnormalized = [math.exp(score-max(scores)) for score in scores]
        probabilities = [p/sum(unnormalized) for p in unnormalized]
        expected = torch.tensor([probabilities[0]-2*probabilities[1], 3*probabilities[2]],dtype=dtype)
        max_error = max(max_error, (output.flatten()-expected).abs().max().item(),
                        (attention.flatten()-torch.tensor(probabilities,dtype=dtype)).abs().max().item())
        weights.append(attention.flatten().detach().tolist())
        errors.append((output-embedding).square().mean().item())
    return {
        'kind': 'toy_experiment', 'problem': 'architecture/block_attention_residual', 'title': 'Kimi K3: Block Attention Residuals',
        'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/html/2607.24653v2', 'anchor': 'Section 2.2, Eqs.8–10; official Attention-Residuals README pseudocode', 'adaptation': 'Three synthetic depth sources and a query-strength sweep demonstrate the operator; this is not a reported paper experiment or trained model.'},
        'metrics': {'maximum_error': max_error, 'uniform_mse': errors[0], 'selected_mse': errors[-1], 'selected_source_weight': weights[-1][0]},
        'charts': [
            {'title': 'Which depth source does the pseudo-query select?', 'kind': 'line', 'x': strengths.tolist(), 'x_label': 'Query strength',
             'series': [{'name': name, 'values': [row[i] for row in weights]} for i,name in enumerate(['Embedding','Completed block','Current partial block'])]},
            {'title': 'Distance to useful embedding', 'kind': 'line', 'x': strengths.tolist(), 'x_label': 'Query strength',
             'series': [{'name': 'Implementation MSE', 'values': errors}]},
        ],
    }
