# Muon Split notes

Start from the matrix layout, not a global normalization: the output features are consecutive head blocks. A single polar factor over all heads couples their relative scales. Per-head scalar Frobenius normalization also fails to equalize singular directions.

Use reduced SVD and multiply the columns of U by the retained-direction indicators before multiplying by Vh. The cutoff is relative to each head's largest singular value. Its strict comparison makes an all-zero head remain zero. Merge exactly the original row layout.

The exact partial polar factor is our teaching contract, not a specified GLM production kernel. Full-rank, distinct-singular-value local derivatives are useful for validating an intact functional tensor path; actual optimizer updates commonly run without recording an autograd graph. The supplied experiment is an algebraic head-scale diagnostic, so it makes no convergence or model-quality claim.
