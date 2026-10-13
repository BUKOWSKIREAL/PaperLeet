# Reference notes

Use the two dot products `(e*r).sum(-1)` and `(e*u).sum(-1)` to apply the rotation. Detaching only the final output also detaches the required encoder path; detaching nothing differentiates the changing transformation rather than the intended surrogate.

The 2D oracle computes an angle with `atan2(q_y,q_x)-atan2(e_y,e_x)` and forms its sine/cosine matrix, independently of the efficient Householder formula. Higher-dimensional tests check the orthogonal complement, norm scaling and anchor-angle preservation.

For the toy quadratic, the upstream gradient is `2q`. The transported gradient is radial at `e` and has norm `2||q||²/||e||`; it is generally **not** the unquantized gradient `2e`. STE instead returns `2q` at every point in a quantization region. These are distinct surrogate choices, not claims that one equals the true derivative of nearest-neighbor quantization.

The singular cases are explicitly excluded, and tests do not impose a fictitious unique rotation there. The selected code receives no reconstruction-path gradient; an actual VQ-VAE can update its codebook with a separate objective or EMA.
