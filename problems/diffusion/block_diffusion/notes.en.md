# Reference notes

Build the three small masks from local block indices, then assemble their quadrants in the specified noisy/clean order. A token-causal triangle is wrong inside a block: both noisy and clean representations need bidirectional access within their own block.

The scalar oracle constructs per-block lists of visible keys and computes attention over those local contexts. It never calls the reference mask. Gradient checks explicitly derive a softmax VJP for noisy query 2 and verify zero derivatives at every forbidden key/value. The toy's influence matrix is measured by rerunning the submitted operator with one perturbed value, rather than displayed from metadata.

With block size 1, this training mask still has two streams: each noisy position sees its own noisy token and earlier clean tokens. It is not the ordinary causal mask over the concatenated sequence. With a single full block, the two streams are isolated and internally bidirectional.

Attention dependency correctness does not by itself verify discrete diffusion probabilities or the paper's likelihood bound; those are separate modules.
