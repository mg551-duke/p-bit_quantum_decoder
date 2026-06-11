# Decoders

All decoders return a correction, logical class, validity flag, timing, and sampler diagnostics.
Logical success is evaluated with `A(e_true xor e_hat) = 0 mod 2`.

## ExactCosetDecoder

For small affine solution spaces, enumerate all `e` satisfying `H e = s`, group by `A e`, compute
coset log weights, and return a representative from the maximum-posterior coset.

## PBitCosetSamplerDecoder

Use a soft syndrome-penalty energy:

```text
E(e | s) = sum_j lambda_j e_j + rho sum_a 1[(H e)_a xor s_a = 1]
lambda_j = log((1 - p_j) / p_j)
rho = max(8.0, 4.0 * max(lambda_j))
```

The sampler uses stochastic single-bit Gibbs/p-bit updates by default:

```text
P(e_j = 1 | e_setminus_j, s) = 1 / (1 + exp(beta Delta_j))
Delta_j = E(e_j = 1) - E(e_j = 0)
```

This is the direct binary-variable form of a p-bit stochastic unit controlled by its local input
bias. The implementation also keeps `update = "metropolis"` available as an explicit comparison
mode. The default chain initialization is random; `initial_state = "affine"` is available as an
experimental syndrome-aware warm start, but it is not used in the v1 artifact because a bounded
calibration check did not improve the published hard-case recovery score. Syndrome residuals are
maintained incrementally. Syndrome-valid samples are grouped by logical class, and the most
frequently sampled class is selected.

## PBitAnnealedOptimizerDecoder

Use the same soft energy but anneal from low to high beta with the same Gibbs/p-bit conditional
update and return the lowest-prior-energy valid correction found. This is the optimization-style
Ising baseline.

## NullspaceCosetSamplerDecoder

Solve `H e = s`, sample inside the affine solution space using a GF(2) nullspace basis, and group
valid states by logical class. This diagnostic separates validity search from coset-mass
estimation.
