"""Teaching adaptation: ROE recovery under a high-leverage corrupted coordinate."""
import torch


def run(optimal_scale_1d, roe_align):
    dtype = torch.float64
    p = torch.tensor([[-1., 0., 1.], [0., 1., 2.], [1., -1., 3.],
                      [2., 1., 4.], [-2., -1., 5.], [40., 2., 6.]], dtype=dtype)
    truth = 1.5 * p + p.new_tensor([0., 0., 2.])
    corrupted = truth.clone()
    corrupted[-1, 0] = -100.
    mask = torch.ones(len(p), dtype=torch.bool)
    scale, shift = roe_align(p, corrupted, mask, trunc=.4)
    plain_scale, plain_shift = roe_align(p, corrupted, mask)
    recovered = scale * p + p.new_tensor([0., 0., shift])
    plain = plain_scale * p + p.new_tensor([0., 0., plain_shift])
    weights = (1 / corrupted[:, 2])[:, None].expand_as(p).reshape(-1)
    axis = torch.linspace(-3, 3, 121, dtype=dtype)
    x = p.clone(); x[:, 2] -= p[0, 2]
    y = corrupted.clone(); y[:, 2] -= corrupted[0, 2]
    best, objective = optimal_scale_1d(x.flatten(), y.flatten(), weights, .4)
    curves = weights * (axis[:, None] * x.flatten() - y.flatten()).abs()
    return {
        "kind": "toy_experiment", "title": "ROE: truncated weighted L1 rejects a leverage outlier",
        "problem": "representation/moge_alignment",
        "provenance": {"kind": "teaching_adaptation", "source": "https://arxiv.org/abs/2410.19115",
                       "anchor": "Section 3, optimal alignment and robust optimization",
                       "adaptation": "Six generated 3D points, known scale/z-shift, one corrupted x coordinate; not a paper benchmark."},
        "metrics": {"scale": scale.item(), "z_shift": shift.item(),
                    "inlier_max_error": (recovered[:-1] - truth[:-1]).abs().max().item(),
                    "untruncated_inlier_error": (plain[:-1] - truth[:-1]).abs().mean().item(),
                    "anchor_scale": best.item(), "anchor_objective": objective.item()},
        "charts": [
            {"kind": "line", "title": "One anchor's scale objective", "x_label": "Scale",
             "x": axis.tolist(), "series": [
                 {"name": "Truncated weighted L1", "values": curves.clamp_max(.4).sum(-1).tolist()},
                 {"name": "Untruncated weighted L1", "values": curves.sum(-1).tolist()}]},
            {"kind": "line", "title": "Aligned depth on the inlier points", "x_label": "Point index",
             "x": list(range(5)), "series": [
                 {"name": "Ground truth", "values": truth[:-1, 2].tolist()},
                 {"name": "Submitted ROE", "values": recovered[:-1, 2].tolist()},
                 {"name": "Untruncated", "values": plain[:-1, 2].tolist()}]}],
    }
