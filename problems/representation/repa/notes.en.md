# Reference notes

REPA normalizes **each patch** after projection. Pooling first destroys the correspondence that the objective is meant to transfer. The loss uses negative cosine, without a `1 - cosine` offset; the offset would leave gradients unchanged but violate the numeric objective contract.

For a patch above the epsilon floor, the student derivative is

$$-\frac1{BN}\left(\frac z{\lVert p\rVert\lVert z\rVert}-\frac{(p^Tz)p}{\lVert p\rVert^3\lVert z\rVert}\right).$$

Below the floor, treat the clamped norm as constant. Tests avoid the nondifferentiable clamp boundary. Teacher freezing is performed inside this exercise's loss so callers cannot accidentally train the target branch. The author training setup obtains clean teacher features outside the diffusion model.

The toy's cosine curve is an independently measured statistic. It should improve because a small trainable projector is optimized on fixed synthetic data. The experiment does not establish generation quality, FID, or a speedup from aligning pretrained visual features.
