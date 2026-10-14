# Reference notes

The real and imaginary parts of the empirical characteristic function are batch means of cosine and sine. Squaring their differences after averaging compares distributions. Averaging squared per-sample differences would define a different objective.

For real-valued samples, the negative-frequency characteristic function is the conjugate of its positive-frequency counterpart. Its squared distance to the real Gaussian target is therefore even. A trapezoid rule on `[0,t_max]` can be doubled; its endpoint weights remain half the interior weights for a uniform grid. Nonuniform grids use adjacent node differences. At zero frequency, both characteristic functions equal one.

The source is [LeJEPA v1](https://arxiv.org/html/2511.08544v1), Sections 4.2.3 and 5.1, with the optimized code convention in [official EppsPulley at c293d291](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/univariate/epps_pulley.py), [slicing.py](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/multivariate/slicing.py) and [MINIMAL.md](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md). The paper snippet uses a full grid; the pinned optimized code uses the positive grid and doubled quadrature weights. This module preserves float32/float64 and accepts supplied unit directions and nodes. It omits distributed reduction and direction resampling.

SIGReg targets a fixed standard normal, unlike a normality test that estimates a mean and scale first. A balanced binary distribution already matches its first two moments but not its characteristic function. Conversely, finite selected frequencies and directions can miss differences: the toy demonstrates a specific distinction, not a universal finite-test guarantee.

The `N` multiplier is part of the statistical test. Duplicating observations preserves the empirical characteristic function but doubles this statistic; increasing the batch size is not an invariant. Four mutants remove the imaginary term, omit `N`, omit positive-half doubling, or standardize the projections first.

At exact zero collapse, sine is zero and cosine has zero derivative, so every feature gradient vanishes even though the loss is positive. T2 tests this fact and descent near a nonconstant perturbation. Toy optimization starts from a distorted nonconstant cloud. The oracle uses a full symmetric grid and complex arithmetic independently of the submitted functions.
