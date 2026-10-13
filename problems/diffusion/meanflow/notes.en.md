# Review: MeanFlow: Average Velocity and JVP Targets

Use paper argument order `(z,r,t)`: JVP tangents `(v,0,1)` hold r fixed. Noise increases with t. The target is `stopgrad(v-(t-r) du/dt)`. We use elementwise mean MSE, omitting the optional adaptive weighting in the official image implementation. At r=t only the target becomes conditional flow matching; the untrained network need not predict v.

An analytic point-data family learns its center using the learner JVP target. This is a mechanism demonstration, not a low-dimensional experiment claimed by the paper.

The deliberately incorrect implementations isolate wrong jvp time, target has gradient, reverse sampling. T1 checks formula values, T2 checks invariants/gradient routing, and T3 invokes the learner implementation in the toy.
