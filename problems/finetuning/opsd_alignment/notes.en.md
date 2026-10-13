# Reference notes

Build a causal index for each branch before gathering. Invalid indices may be clamped safely, but masking must zero their outputs and gradients. Detaching the teacher must not detach the student. Both policies rescore the same student trajectory, not the reference solution or a newly generated teacher response.
