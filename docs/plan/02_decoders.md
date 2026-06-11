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

## PBitNullspaceCosetSamplerDecoder

Solve `H e = s`, write valid corrections as `e = x0 + z B` over GF(2), and treat the binary
nullspace coordinates `z_i` as p-bits. Each update uses the same logistic conditional:

```text
P(z_i = 1 | z_setminus_i, s) = 1 / (1 + exp(beta Delta_i))
Delta_i = E(z_i = 1) - E(z_i = 0)
```

Because every state in the affine parameterization is syndrome-valid, this decoder separates the
hardware-style stochastic update question from the validity-search problem. It is the strongest
non-exact hard-case recovery decoder in the current artifact. The v1 hard-case calibration selected
`beta = 0.75` and `updates_per_sample = 4`. For broad topological and MWPM-comparison rows, the
current artifact uses the same `beta = 0.75` but increases to `updates_per_sample = 32` with
`update_schedule = "sweep"` so each recorded sample follows more local stochastic p-bit relaxation.
The BB144 sweep uses a separate particular-start setting, `beta = 0.5`, `updates_per_sample = 78`,
four chains, and random-sweep coordinate updates, because random single-chain starts mixed poorly in
the 78-dimensional BB144 affine solution space.

## NullspaceCosetSamplerDecoder

Solve `H e = s`, sample inside the affine solution space using a GF(2) nullspace basis, and group
valid states by logical class. This diagnostic separates validity search from coset-mass
estimation.

## MWPMDecoder

Use PyMatching for graphlike topological syndrome matrices with column weight at most two. This is
the topological-code baseline requested after the initial prototype. It is deliberately not applied
to BB144, whose syndrome matrix is not graphlike in this representation.

## BPOSDDecoder

Use the LDPC package's BP-OSD implementation for qLDPC-style syndrome matrices. This is the strong
BB144 baseline. In the committed artifact it uses minimum-sum BP, 100 maximum iterations, OSD-CS,
and OSD order 10. BP-OSD is not a p-bit decoder; it is included to make the performance comparison
honest.
