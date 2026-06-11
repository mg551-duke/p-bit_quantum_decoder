# v1 Large, Precision, MWPM, And BB144 Results

This file summarizes the generated `results/v1_large` artifacts from
`experiments/configs/v1_large.json` and `experiments/configs/v1_hard_cases.json`, plus the
focused precision artifacts from `experiments/configs/v1_precision_focus.json` and
`experiments/configs/v1_hard_cases_precision.json`, the topological MWPM comparison from
`experiments/configs/v1_mwpm_comparison.json`, the [[144,12,12]] bivariate-bicycle p-bit sweep
from `experiments/configs/v1_bb144_pbit.json`, and the p-bit-native syndrome probe from
`experiments/configs/v1_pbit_syndrome.json`.

## Artifact Contents

- `docs/results/index.html`: static HTML report containing all generated plots and current
  aggregate results.
- `trials.csv`: 147,600 decoder-trial rows.
- `summary.csv`: 234 grouped summary rows.
- `coset_distributions.csv`: 309,418 sampled/exact coset rows.
- `hard_cases.csv`: 59 small-code syndromes where the exact maximum-coset class differs from
  the minimum-energy class.
- `hard_case_recovery.csv`: 9,440 recovery attempts on mined hard cases.
- `hard_case_summary.csv`: grouped hard-case recovery rates with Wilson confidence intervals.
- `figures/v1_large/`: six PNG figures for logical error, exact agreement, valid sampling,
  logical-class diversity, coset entropy, and hard-case recovery.
- `results/v1_precision_focus/`: 28,000 focused decoder-trial rows, 70 grouped summary rows, and
  58,165 coset-distribution rows with 400 trials in every configured group.
- `figures/v1_precision_focus/`: five PNG figures for the focused logical-error, exact-agreement,
  valid-sampling, logical-class diversity, and coset-entropy curves.
- `results/v1_hard_cases_precision/`: 37,760 high-repeat hard-case recovery attempts and grouped
  Wilson intervals.
- `results/v1_hard_cases_precision/pbit_nullspace_calibration.csv`: tuning grid used to select
  `beta = 0.75` and `updates_per_sample = 4` for `pbit_nullspace_sampler`.
- `figures/v1_hard_cases_precision/hard_case_recovery.png`: high-repeat hard-case recovery plot
  with interval bars.
- `results/v1_mwpm_comparison/`: 32,000 topological decoder-trial rows, 70 grouped summary rows,
  and 69,599 coset-distribution rows comparing exact, MWPM, nullspace, and nullspace-coordinate
  p-bit decoders.
- `figures/v1_mwpm_comparison/`: five PNG figures for logical error, exact-coset agreement,
  valid-sampling, logical-class diversity, and coset entropy in the MWPM comparison.
- `results/v1_bb144_pbit/`: 9,600 BB144 decoder-trial rows, 24 grouped summary rows, and 558,415
  coset-distribution rows for BP-OSD, the BP-guided p-bit sampler, the unguided p-bit sampler,
  and the plain nullspace sampler.
- `figures/v1_bb144_pbit/`: five PNG figures for the BB144 p-bit sweep.
- `results/v1_pbit_syndrome/`: 5,400 decoder-trial rows, 108 grouped summary rows, and 35,372
  coset-distribution rows for the no-linear-solve p-bit syndrome pilot.
- `figures/v1_pbit_syndrome/`: eleven PNG figures including first-valid, residual, distinct
  valid-solution, and selected-weight diagnostics.

The generated artifact directories are small enough for review, with no generated file near the
50 MB limit.

## Cautious Readout

These PR artifacts are reproducible benchmarks, not final publication-strength statistical claims.
The p-bit decoders use direct Gibbs/p-bit conditional updates with
`P(e_j = 1 | rest, s) = 1 / (1 + exp(beta Delta_j))`. Averaged over the configured grouped
summaries, exact coset decoding is the best available small-code reference. The larger run improves
the soft p-bit sampler's logical error rate, exact-coset agreement, and valid-sample rate relative
to the earlier artifact. The `pbit_nullspace_sampler` uses the same Gibbs/p-bit conditional over
affine nullspace coordinates, keeping every sampled correction syndrome-valid by construction. The
stronger topological setting with 32 sweep-style local updates per collected sample now has the
best non-exact exact-coset agreement in the v1-large sweep, while the separately calibrated
hard-case setting still gives the strongest mined hard-case recovery.

Mean exact-coset agreement where exact decoding is available:

- `exact_coset`: 1.000
- `pbit_nullspace_sampler`: 0.941
- `nullspace_coset_sampler`: 0.864
- `pbit_annealed_optimizer`: 0.520
- `pbit_coset_sampler`: 0.504

The broad v1-large sweep now uses 1000 trials for exact-feasible small toric/surface groups and
400 trials for medium toric/surface groups. The worst v1-large logical-error Wilson half-width is
0.0488, and every v1-large summary row reports zero additional trials needed for a 5% Wilson
half-width target.

Overall hard-case exact-coset recovery with 95% Wilson intervals:

- `pbit_nullspace_sampler`: 0.537 [0.517, 0.557]
- `pbit_annealed_optimizer`: 0.445 [0.425, 0.465]
- `nullspace_coset_sampler`: 0.440 [0.420, 0.460]
- `pbit_coset_sampler`: 0.419 [0.400, 0.440]

The precision focus run reduces mean logical-error Wilson half-widths on matching groups from about
0.112 to 0.046 for `pbit_coset_sampler`, a 58.6% reduction. Focused mean logical error rates are:

- `exact_coset`: 0.247, mean half-width 0.038
- `nullspace_coset_sampler`: 0.330, mean half-width 0.042
- `pbit_coset_sampler`: 0.486, mean half-width 0.046
- `pbit_annealed_optimizer`: 0.498, mean half-width 0.046

The worst precision-focus logical-error Wilson half-width is now 0.0488, and every
precision-focus row reports zero additional trials needed for a 5% Wilson half-width target.

The high-repeat hard-case run changes the hard-case readout:

- `pbit_nullspace_sampler`: 5414/9440 = 0.574 [0.564, 0.583]
- `nullspace_coset_sampler`: 4430/9440 = 0.469 [0.459, 0.479]
- `pbit_annealed_optimizer`: 4338/9440 = 0.460 [0.449, 0.470]
- `pbit_coset_sampler`: 4074/9440 = 0.432 [0.422, 0.442]

The current evidence is therefore not strong enough to claim a hard-case advantage for soft
physical-bit p-bit coset sampling alone. It does support a hard-case advantage for
nullspace-coordinate p-bit sampling, which separates syndrome validity from posterior coset-mass
estimation. The high-repeat hard-case overall Wilson half-width is now about 0.010, and the
`pbit_nullspace_sampler` lower bound is above the other decoders' upper bounds. The focused
precision run still supports a small logical-error advantage for the soft physical-bit sampler over
annealed optimization in those configured groups.

## Topological MWPM Comparison

The MWPM comparison uses PyMatching on graphlike toric and surface-code syndrome matrices. MWPM is
not run for BB144 because the BB syndrome matrix is not graphlike.

Mean logical error across configured topological groups:

- `exact_coset`: 0.244, mean half-width 0.034
- `mwpm`: 0.270, mean half-width 0.035
- `nullspace_coset_sampler`: 0.336, mean half-width 0.039
- `pbit_nullspace_sampler`: 0.302, mean half-width 0.039

Logical error by code family and size:

- surface 3: exact 0.204, MWPM 0.226, nullspace 0.232, p-bit nullspace 0.211
- surface 7: MWPM 0.241, nullspace 0.364, p-bit nullspace 0.286
- toric 3: exact 0.284, MWPM 0.283, nullspace 0.314, p-bit nullspace 0.293
- toric 5: MWPM 0.329, nullspace 0.432, p-bit nullspace 0.419

Mean exact-coset agreement on exact-feasible small groups:

- `exact_coset`: 1.000
- `mwpm`: 0.910
- `pbit_nullspace_sampler`: 0.948
- `nullspace_coset_sampler`: 0.867

This is now a mixed topological-code comparison rather than a simple negative result for the
current stochastic p-bit implementation. MWPM is still faster and has lower mean logical error
overall, but the better-mixed p-bit nullspace sampler reduces mean logical error from the previous
0.367 artifact to 0.302, beats the plain nullspace sampler in the configured topological
comparison, and has higher exact-coset agreement than MWPM on the exact-feasible small groups.

The medium topological rows now use 400 trials per physical error rate. The worst logical-error
Wilson half-width in `results/v1_mwpm_comparison/summary.csv` is 0.0488, and every medium
topological row reports zero additional trials needed for a 5% Wilson half-width target.

## BB144 P-bit Sweep

The BB144 artifact tests the X-error sector of the [[144,12,12]] bivariate-bicycle gross code with
`l = 12`, `m = 6`, `A = x^3 + y + y^2`, and `B = y^3 + x + x^2`. It now includes LDPC BP-OSD as
the strong qLDPC baseline. The p-bit sampler uses particular-solution initialization, `beta = 0.5`,
`updates_per_sample = 78`, four chains, and a random-sweep schedule, matching one full randomized
pass over the 78-dimensional affine nullspace per collected sample. It samples at the tempered
mixing beta but selects the decoded coset with importance-weighted scores targeting the physical
posterior at `target_beta = 1`. The BP-guided p-bit row uses plain BP, not OSD, as a valid affine
origin when BP converges and falls back to the GF(2) particular solution otherwise.

BB144 logical error by physical error rate:

- p=0.01: BP-OSD 0.000, BP-guided p-bit 0.000, p-bit nullspace 0.085, nullspace 0.315
- p=0.02: BP-OSD 0.000, BP-guided p-bit 0.000, p-bit nullspace 0.212, nullspace 0.528
- p=0.04: BP-OSD 0.005, BP-guided p-bit 0.042, p-bit nullspace 0.458, nullspace 0.808
- p=0.06: BP-OSD 0.083, BP-guided p-bit 0.208, p-bit nullspace 0.680, nullspace 0.898
- p=0.08: BP-OSD 0.273, BP-guided p-bit 0.545, p-bit nullspace 0.900, nullspace 0.978
- p=0.10: BP-OSD 0.518, BP-guided p-bit 0.808, p-bit nullspace 0.955, nullspace 0.995

Aggregate BB144 readout:

- `bp_osd`: mean logical error 0.146, mean half-width 0.023, mean runtime 0.9 ms
- `bp_guided_pbit_sampler`: mean logical error 0.267, mean half-width 0.026, mean runtime 224 ms,
  mean coset entropy 1.860 bits
- `pbit_nullspace_sampler`: mean logical error 0.548, mean half-width 0.035, mean runtime 226 ms,
  mean coset entropy 3.615 bits
- `nullspace_coset_sampler`: mean logical error 0.753, mean half-width 0.031, mean runtime 13 ms,
  mean coset entropy 0.628 bits

This is still negative for p-bit competitiveness against a strong BB144 decoder: BP-OSD is much
faster and lower-error. The BP-guided p-bit row is scientifically useful because it shows where BP
information helps: BP finds a much better valid affine origin, and p-bit sampling then explores
nearby nullspace cosets. It closes much of the gap between unguided p-bit sampling and BP-OSD, but
it is not a state-of-the-art BB144 decoder. Exact coset enumeration is not feasible in this
artifact.

The BB144 rows now use 400 trials per physical error rate. The worst logical-error Wilson
half-width in `results/v1_bb144_pbit/summary.csv` is 0.0488, and every BB144 row reports zero
additional trials needed for a 5% Wilson half-width target.

## P-Bit-Native Syndrome Probe

The `v1_pbit_syndrome` artifact tests whether Bernoulli p-bit randomness and asynchronous local
updates can find `H e = s` directly, without `gf2.solve_affine`, BP, BP-OSD, MWPM, or nullspace
sampling inside `PBitSyndromeDecoder`. It is a 50-trial-per-group pilot, so the Wilson intervals
are wider than the precision artifacts.

The p-bit syndrome row now uses minimum-weight selection over up to ten distinct valid physical
corrections. It has 1.000 valid and first-valid rate on the configured toric and surface groups.
On BB144, it has mean valid rate 0.927, mean first-valid sweep 20.9, and mean best residual weight
0.147. This supports the narrow claim that p-bit-native stochastic dynamics can often find
syndrome-valid corrections without deterministic syndrome solving.

It does not support a strong BB144 decoding claim. Mean BB144 logical error is 0.583 for
`pbit_syndrome`, compared with 0.167 for BP-OSD, 0.297 for BP-guided p-bit sampling, and 0.587 for
the unguided p-bit nullspace row in this pilot. The ten-solution minimum-weight rule is also not
exercised well on BB144: the BB144 rows average 1.91 distinct valid physical corrections and never
hit the target of ten within this budget. The main BB144 bottleneck is therefore not just first
validity; it is finding a diverse enough set of valid corrections, then selecting the right logical
class from that set.

Required interpretation constraints:

- logical success is coset/logical success, not exact physical-error recovery;
- invalid corrections are failures;
- exact-coset agreement is available only where exact enumeration is configured and feasible;
- hard-case recovery is the key test of sampling versus optimization.
