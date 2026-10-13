# Reference notes

Each row describes one image choosing among texts; each column describes one text choosing among images. Use the same diagonal labels for both cross-entropies, then average them. This produces symmetric supervision even when the actual logits are not symmetric.

Normalizing features removes their norm as a shortcut for increasing similarities. Learnable inverse temperature supplies a shared scale. Clip `logit_scale` before exponentiation to avoid an overflowing intermediate and a possible NaN gradient; do not detach it below the cap.

The paper describes an upper inverse-temperature bound of 100. The public inference implementation need not show its entire training-time constraint, so this exercise states the convention explicitly. The paired-vector toy is a teaching adaptation, not a recovered author experiment.
