"""Paper Figure 3 gate curves plus a separate educational memory-overwrite example."""
import math
import torch


def run(bounded_decay, recurrent_kda):
    x = torch.linspace(-10, 10, 81, dtype=torch.float64)
    retention = bounded_decay(x[None, :, None, None], torch.zeros(1, dtype=x.dtype)).flatten()
    expected_retention = torch.tensor([math.exp(-5 / (1 + math.exp(-z))) for z in x.tolist()], dtype=x.dtype)
    old_log_decay = -torch.nn.functional.softplus(x)
    q = torch.tensor([[[[1., 0.]], [[0., 1.]], [[1., 0.]], [[1., 0.]]]], dtype=x.dtype)
    k = q.clone()
    v = torch.tensor([[[[2.]], [[7.]], [[5.]], [[100.]]]], dtype=x.dtype)
    alpha = torch.ones_like(q)
    alpha[:, 3, :, 0] = .25
    beta = torch.tensor([[[1.], [1.], [1.], [0.]]], dtype=x.dtype)
    output, state = recurrent_kda(q, k, v, alpha, beta)
    expected = torch.tensor([2., 7., 5., 1.25], dtype=x.dtype)
    return {
        'kind': 'toy_experiment', 'problem': 'architecture/kimi_kda', 'title': 'Kimi K3: Bounded Decay and Delta Memory',
        'provenance': {'kind': 'paper_derived', 'source': 'https://arxiv.org/html/2607.24653v2', 'anchor': 'Section 2.1.1, Eqs.1/5 and Figure 3(a)', 'adaptation': 'Redraw Figure 3(a) with A=0, g_min=-5 and logits [-10,10]; separately add a four-token memory-overwrite teaching example, not an original paper experiment.'},
        'metrics': {'gate_maximum_error': (retention-expected_retention).abs().max().item(),
                    'memory_maximum_error': (output.flatten()-expected).abs().max().item(),
                    'final_state_error': (state.flatten()-torch.tensor([1.25,7.],dtype=x.dtype)).abs().max().item()},
        'charts': [
            {'title': 'Figure 3(a): bounded and unbounded log-decay', 'kind': 'line', 'x': x.tolist(), 'x_label': 'Decay logit',
             'series': [{'name': 'K3 implementation', 'values': retention.log().detach().tolist()},
                        {'name': 'Kimi Linear: negative softplus', 'values': old_log_decay.tolist()},
                        {'name': 'Lower bound', 'values': [-5.] * len(x)}]},
            {'title': 'Write, read another key, overwrite, then forget', 'kind': 'bar', 'x': [0,1,2,3], 'x_label': 'Token',
             'series': [{'name': 'Implementation', 'values': output.detach().flatten().tolist()},
                        {'name': 'Analytic target', 'values': expected.tolist()}]},
        ],
    }
