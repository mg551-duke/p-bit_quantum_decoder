# v1 Large, Precision, MWPM, And BB144 Results

This file summarizes the generated `results/v1_large` artifacts from
`experiments/configs/v1_large.json` and `experiments/configs/v1_hard_cases.json`, plus the
focused precision artifacts from `experiments/configs/v1_precision_focus.json` and
`experiments/configs/v1_hard_cases_precision.json`, the topological MWPM comparison from
`experiments/configs/v1_mwpm_comparison.json`, and the [[144,12,12]] bivariate-bicycle p-bit sweep
from `experiments/configs/v1_bb144_pbit.json`.

## Artifact Contents

- `docs/results/index.html`: static HTML report containing all generated plots and current
  aggregate results.
- `trials.csv`: 18,000 decoder-trial rows.
- `summary.csv`: 234 grouped summary rows.
- `coset_distributions.csv`: 37,474 sampled/exact coset rows.
- `hard_cases.csv`: 59 small-code syndromes where the exact maximum-coset class differs from
  the minimum-energy class.
- `hard_case_recovery.csv`: 9,440 recovery attempts on mined hard cases.
- `hard_case_summary.csv`: grouped hard-case recovery rates with Wilson confidence intervals.
- `figures/v1_large/`: six PNG figures for logical error, exact agreement, valid sampling,
  logical-class diversity, coset entropy, and hard-case recovery.
- `results/v1_precision_focus/`: 18,200 focused decoder-trial rows, 70 grouped summary rows, and
  37,792 coset-distribution rows with 260 trials in every configured group.
- `figures/v1_precision_focus/`: five PNG figures for the focused logical-error, exact-agreement,
  valid-sampling, logical-class diversity, and coset-entropy curves.
- `results/v1_hard_cases_precision/`: 37,760 high-repeat hard-case recovery attempts and grouped
  Wilson intervals.
- `results/v1_hard_cases_precision/pbit_nullspace_calibration.csv`: tuning grid used to select
  `beta = 0.75` and `updates_per_sample = 4` for `pbit_nullspace_sampler`.
- `figures/v1_hard_cases_precision/hard_case_recovery.png`: high-repeat hard-case recovery plot
  with interval bars.
- `results/v1_mwpm_comparison/`: 29,000 topological decoder-trial rows, 70 grouped summary rows,
  and 63,267 coset-distribution rows comparing exact, MWPM, nullspace, and nullspace-coordinate
  p-bit decoders.
- `figures/v1_mwpm_comparison/`: five PNG figures for logical error, exact-coset agreement,
  valid-sampling, logical-class diversity, and coset entropy in the MWPM comparison.
- `results/v1_bb144_pbit/`: 3,600 BB144 decoder-trial rows, 12 grouped summary rows, and 202,491
  coset-distribution rows for the nullspace and particular-start nullspace-coordinate p-bit
  samplers.
- `figures/v1_bb144_pbit/`: five PNG figures for the BB144 p-bit sweep.

The generated artifact directories are small enough for review, with no generated file near the
50 MB limit.

## Cautious Readout

These PR artifacts are reproducible benchmarks, not final publication-strength statistical claims.
The p-bit decoders use direct Gibbs/p-bit conditional updates with
`P(e_j = 1 | rest, s) = 1 / (1 + exp(beta Delta_j))`. Averaged over the configured grouped
summaries, exact coset decoding is the best available small-code reference. The larger run improves
the soft p-bit sampler's logical error rate, exact-coset agreement, and valid-sample rate relative
to the earlier artifact. The new `pbit_nullspace_sampler` uses the same Gibbs/p-bit conditional over
affine nullspace coordinates, keeping every sampled correction syndrome-valid by construction. The
tuned setting has comparable non-exact exact-coset agreement in the v1-large sweep and the strongest
mined hard-case recovery.

Mean exact-coset agreement where exact decoding is available:

- `exact_coset`: 1.000
- `nullspace_coset_sampler`: 0.863
- `pbit_nullspace_sampler`: 0.861
- `pbit_annealed_optimizer`: 0.529
- `pbit_coset_sampler`: 0.514

Overall hard-case exact-coset recovery with 95% Wilson intervals:

- `pbit_nullspace_sampler`: 0.537 [0.517, 0.557]
- `pbit_annealed_optimizer`: 0.445 [0.425, 0.465]
- `nullspace_coset_sampler`: 0.440 [0.420, 0.460]
- `pbit_coset_sampler`: 0.419 [0.400, 0.440]

The precision focus run reduces mean logical-error Wilson half-widths on matching groups from about
0.112 to 0.057 for `pbit_coset_sampler`, a 48.8% reduction. Focused mean logical error rates are:

- `exact_coset`: 0.251, mean half-width 0.047
- `nullspace_coset_sampler`: 0.335, mean half-width 0.052
- `pbit_coset_sampler`: 0.485, mean half-width 0.057
- `pbit_annealed_optimizer`: 0.498, mean half-width 0.057

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
- `mwpm`: 0.271, mean half-width 0.038
- `nullspace_coset_sampler`: 0.336, mean half-width 0.042
- `pbit_nullspace_sampler`: 0.364, mean half-width 0.045

Logical error by code family and size:

- surface 3: exact 0.204, MWPM 0.226, nullspace 0.232, p-bit nullspace 0.219
- surface 7: MWPM 0.240, nullspace 0.355, p-bit nullspace 0.368
- toric 3: exact 0.284, MWPM 0.283, nullspace 0.314, p-bit nullspace 0.323
- toric 5: MWPM 0.334, nullspace 0.441, p-bit nullspace 0.547

Mean exact-coset agreement on exact-feasible small groups:

- `exact_coset`: 1.000
- `mwpm`: 0.910
- `pbit_nullspace_sampler`: 0.891
- `nullspace_coset_sampler`: 0.867

This is a negative topological-code comparison for the current stochastic p-bit implementation:
MWPM is faster and has lower logical error overall, especially on the medium toric and surface-code
sectors. The p-bit nullspace sampler remains useful as a coset-mass diagnostic and is competitive
only on the small surface-3 aggregate in this sweep.

## BB144 P-bit Sweep

The BB144 artifact tests the X-error sector of the [[144,12,12]] bivariate-bicycle gross code with
`l = 12`, `m = 6`, `A = x^3 + y + y^2`, and `B = y^3 + x + x^2`. The p-bit sampler uses a
particular-solution initialization, `beta = 0.5`, and `updates_per_sample = 78`, matching one
expected pass over the 78-dimensional affine nullspace per collected sample.

BB144 logical error by physical error rate:

- p=0.01: nullspace 0.297, p-bit nullspace 0.217
- p=0.02: nullspace 0.500, p-bit nullspace 0.390
- p=0.04: nullspace 0.817, p-bit nullspace 0.670
- p=0.06: nullspace 0.930, p-bit nullspace 0.803
- p=0.08: nullspace 0.970, p-bit nullspace 0.940
- p=0.10: nullspace 0.997, p-bit nullspace 0.993

Aggregate BB144 readout:

- `pbit_nullspace_sampler`: mean logical error 0.669, mean half-width 0.040, mean runtime 247 ms,
  mean coset entropy 2.849 bits
- `nullspace_coset_sampler`: mean logical error 0.752, mean half-width 0.035, mean runtime 14 ms,
  mean coset entropy 0.609 bits

This is cautiously positive for the particular-start p-bit sampler relative to the plain nullspace
baseline, but it is not a strong BB144 decoding result. The high-p logical-error rates remain large,
and exact coset enumeration is not feasible in this artifact.

Required interpretation constraints:

- logical success is coset/logical success, not exact physical-error recovery;
- invalid corrections are failures;
- exact-coset agreement is available only where exact enumeration is configured and feasible;
- hard-case recovery is the key test of sampling versus optimization.
