"""Recompute the scalar experiment in Kimi K3 Figure 4; no copied figure data."""
import math
import torch


def scalar_oracle(gate, up, beta1=4.0, beta2=25.0):
    """Independent scalar arithmetic for the chart's numerical error check."""
    sigmoid = 1 / (1 + math.exp(-gate))
    return beta1 * math.tanh(gate / beta1) * sigmoid * beta2 * math.tanh(up / beta2)


def run(softcap, situ_glu):
    x = torch.linspace(-10, 100, 111, dtype=torch.float64)
    near = torch.linspace(-2, 2, 81, dtype=x.dtype)
    with torch.no_grad():
        output = situ_glu(x, x)
        near_output = situ_glu(near, near)
        gate = softcap(x, 4.0) * x.sigmoid()
        up = softcap(x, 25.0)
        local_x = torch.tensor([-.05, .05], dtype=x.dtype)
        local = situ_glu(local_x, local_x)
        local_swiglu = local_x.square() * local_x.sigmoid()
    expected = torch.tensor([scalar_oracle(v, v) for v in x.tolist()], dtype=x.dtype)

    def series(name, values):
        return {"name": name, "values": values.tolist()}

    def chart(title, axis, curves):
        return {"kind": "line", "title": title, "x_label": "Scalar input x",
                "x": axis.tolist(), "series": curves}

    return {
        "kind": "toy_experiment",
        "title": "Figure 4: GLU, SwiGLU and SiTU-GLU scalar responses",
        "problem": "architecture/situ_glu",
        "provenance": {
            "kind": "paper_derived",
            "source": "https://arxiv.org/html/2607.24653v2#S2.SS3.SSS2",
            "anchor": "Figure 4, Eq. (12), and Appendix B",
            "adaptation": "Independently recomputed curves with both branches x on [-10, 100]; 111 full-range samples and 81 near-origin samples. Not copied graphics or a full-model training experiment.",
        },
        "metrics": {
            "maximum_formula_error": (output - expected).abs().max().item(),
            "maximum_absolute_output": output.abs().max().item(),
            "output_at_x_100": output[-1].item(),
            "coordinate_bound": 100.0,
            "local_relative_swiglu_error": ((local - local_swiglu) / local_swiglu).abs().max().item(),
        },
        "charts": [
            chart("Full scalar response (paper domain)", x, [
                series("GLU", x * x.sigmoid()),
                series("SwiGLU", x.square() * x.sigmoid()),
                series("SiTU-GLU implementation", output),
            ]),
            chart("Gate branch", x, [
                series("GLU sigmoid", x.sigmoid()),
                series("SwiGLU Swish", x * x.sigmoid()),
                series("SiTU gate implementation", gate),
            ]),
            chart("Up branch", x, [
                series("GLU / SwiGLU linear", x),
                series("SiTU up implementation", up),
            ]),
            chart("Near-origin scalar response", near, [
                series("GLU", near * near.sigmoid()),
                series("SwiGLU", near.square() * near.sigmoid()),
                series("SiTU-GLU implementation", near_output),
            ]),
        ],
    }
