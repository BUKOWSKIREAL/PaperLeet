# Review: DMD2: Two-Time-Scale and Adversarial Updates

The critic/fake diffusion model updates more often than the generator (5:1 here). Counts are completed critic updates, so update 0 does not trigger a generator step. Logistic discriminator and generator losses have opposite signs on fake logits. Caller detaches fake data during critic training and freezes critic parameters, while retaining input gradients, during generator training. Unlike DMD, the generator objective has no paired teacher regression. This exercise covers the stabilization/real-data contribution, not the paper’s multistep backward simulation.

An unpaired shifted-Gaussian toy jointly trains a fake score and discriminator for five updates per student update. It demonstrates scheduling, GAN gradient direction, and removal of paired regression. Uses scalar Gaussian score parameters rather than pretrained diffusion features.

The deliberately incorrect implementations isolate equal update frequency, gan wrong sign, no gan generator, target not frozen. T1 checks formula values, T2 checks invariants/gradient routing, and T3 invokes the learner implementation in the toy.
