# Decoders

All decoders return a correction, logical class, validity flag, timing, and sampler diagnostics.
Logical success is evaluated with `A(e_true xor e_hat) = 0 mod 2`.

## ExactCosetDecoder

For small affine solution spaces, enumerate all `e` satisfying `H e = s`, group by `A e`, compute
coset log weights, and return a representative from the maximum-posterior coset.

## PBitSyndromeDecoder

Start from Bernoulli-sampled physical error bits and search the full error space directly, without
calling `gf2.solve_affine`, BP, BP-OSD, MWPM, or a nullspace sampler. The decoder uses the same soft
syndrome-penalty energy and local Gibbs/p-bit conditional as the physical-bit sampler, but it is
instrumented around syndrome convergence: incremental residual updates, first valid sweep,
valid-hit count, final residual weight, best residual weight, and `used_linear_solve = false`.

The first committed mode relies only on Bernoulli randomness and asynchronous update order for
symmetry breaking. `stop_on_first_valid = true` returns the first syndrome-valid correction, while
`collect_valid_samples = true` continues after convergence and groups valid states by logical
class. `selection = "best_energy"`, `selection = "counts"`, `selection = "min_weight"`, and
`selection = "ml_coset"` are all available. Persistent disorder is deliberately left off by
default and should be added only as a follow-up comparison if the Bernoulli-only baseline cannot
find valid corrections reliably.

### Syndrome penalty `rho`

The penalty weight `rho` multiplies the residual-syndrome term in the energy. Resolution
precedence is: an explicit `rho`, then a p-adaptive `rho_scale * max(lambda_j)` when `rho_scale`
is set, then the shared `default_rho(lambda) = max(8.0, 4.0 * max(lambda_j))` heuristic. The
production large-code configs set `rho_scale = 1.0` (`rho ~ max(lambda_j)`, the per-qubit
log-likelihood). The much larger `default_rho` over-penalizes residuals and traps chains away from
the true low-weight coset; `rho_scale = 1.0` restores full coverage (`valid_rate = 1.0`) at far
lower chain counts.

### Selection contract

`min_weight`, `counts`, and `ml_coset` aggregate over the full ensemble of valid samples, so they
require exploring past the first valid hit. `stop_on_first_valid = true` returns the first valid
sample in chain order and is therefore only compatible with `selection = "best_energy"`; combining
it with an aggregation selection raises `ValueError`. Use `stop_on_first_valid = false` (optionally
with `target_valid_solutions` to cap exploration) for those modes. `min_weight` returns the
lowest-Hamming-weight valid correction with prior energy as the tie-break.

### Weight reduction

With `weight_reduction = true`, each per-coset representative is post-processed by greedily XORing
stabilizer generators (the nullspace of `[H; A]`, or the low-weight complementary CSS checks
`metadata['h_x']` when available) to lower its Hamming weight while preserving both syndrome and
logical coset. The incoming global best correction and its own reduction are always folded into the
candidate pool, so the returned correction can never regress above what plain selection would have
returned.

### Vectorized batched path

`vectorized = true` (default) advances all chains together as rows of a `(chains, n)` state matrix.
Bits are still updated sequentially within a sweep (asynchronous Gibbs/Metropolis) so per-chain
dynamics match the scalar fallback; only the per-bit work is vectorized across chains. Results are
deterministic given the seed but draw from a different random stream than the scalar path.

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
the 78-dimensional BB144 affine solution space. Since `beta = 0.5` samples a tempered distribution,
BB144 coset selection uses importance-weighted scores targeting `target_beta = 1.0` instead of raw
tempered sample counts.

The `bp_guided_pbit_sampler` artifact row uses the same nullspace p-bit sampler but first asks
plain BP, without OSD, for a valid correction. If BP converges to a syndrome-valid correction, that
correction becomes the affine origin and all p-bit chains start from the corresponding
zero-coordinate state. If BP does not return a valid correction, the sampler falls back to the
ordinary GF(2) particular solution. This tests whether BP can supply a good region of the affine
solution space while p-bit updates still perform the coset exploration.

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
