"""Teaching adaptation: residual coding of a tiny synthetic latent map."""
import torch


def run(multiscale_quantize, decode_multiscale):
    axis = torch.arange(8)
    smooth = .55 + .3 * (axis[:, None] >= 4).double() - .25 * (axis[None, :] >= 4).double()
    checker = .08 * ((axis[:, None] + axis[None, :]) % 2 * 2 - 1).double()
    features = (smooth + checker)[None, None]
    codebook = torch.tensor([[-1.], [-.5], [-.25], [-.125], [0.], [.125], [.25], [.5], [1.]], dtype=torch.float64)
    scales = [1, 2, 4, 8]
    projections = torch.ones(4, 1, 1, dtype=torch.float64)
    tokens, reconstructed = multiscale_quantize(features, codebook, scales, projections)
    decoded = decode_multiscale(tokens, codebook, projections)
    errors = [float((features - image).square().mean()) for image in reconstructed]
    first_indices = tokens[0].flatten().tolist()
    return {'kind': 'toy_experiment', 'title': 'Coarse structure followed by residual detail',
            'problem': 'representation/var_tokenizer',
            'provenance': {'kind': 'teaching_adaptation', 'source': 'https://arxiv.org/html/2404.02905v2',
                           'anchor': 'Algorithms 1–2; official f_to_idxBl_or_fhat and embed_to_fhat',
                           'adaptation': 'An 8×8 synthetic latent and nine fixed scalar codes illustrate the residual loop. Identity channel projections replace learned spatial Phi; no encoder, image decoder or ImageNet model is trained.'},
            'metrics': {'initial_mse': float(features.square().mean()), 'coarse_mse': errors[0], 'final_mse': errors[-1],
                        'encode_decode_error': float((decoded - reconstructed[-1]).abs().max()),
                        'first_scale_code': first_indices[0], 'number_of_scales': len(tokens)},
            'charts': [{'kind': 'line', 'title': 'Reconstruction MSE after each residual scale', 'x_label': 'Scale width', 'x': scales,
                        'series': [{'name': 'Residual VQ', 'values': errors}]},
                       {'kind': 'line', 'title': 'A row of latent features', 'x_label': 'Column', 'x': axis.tolist(),
                        'series': [{'name': 'Target latent', 'values': features[0, 0, 2].tolist()},
                                   {'name': 'After 1×1 code', 'values': reconstructed[0][0, 0, 2].tolist()},
                                   {'name': 'After all scales', 'values': decoded[0, 0, 2].tolist()}]}]}
