# v1 Large And Precision Results

This file summarizes the generated `results/v1_large` artifacts from
`experiments/configs/v1_large.json` and `experiments/configs/v1_hard_cases.json`, plus the
focused precision artifacts from `experiments/configs/v1_precision_focus.json` and
`experiments/configs/v1_hard_cases_precision.json`.

## Artifact Contents

- `docs/results/index.html`: static HTML report containing all generated plots and current
  aggregate results.
- `trials.csv`: 14,040 decoder-trial rows.
- `summary.csv`: 180 grouped summary rows.
- `coset_distributions.csv`: 27,207 sampled/exact coset rows.
- `hard_cases.csv`: 59 small-code syndromes where the exact maximum-coset class differs from
  the minimum-energy class.
- `hard_case_recovery.csv`: 3,540 recovery attempts on mined hard cases.
- `hard_case_summary.csv`: grouped hard-case recovery rates with Wilson confidence intervals.
- `figures/v1_large/`: six PNG figures for logical error, exact agreement, valid sampling,
  logical-class diversity, coset entropy, and hard-case recovery.
- `results/v1_precision_focus/`: 18,200 focused decoder-trial rows, 70 grouped summary rows, and
  37,792 coset-distribution rows with 260 trials in every configured group.
- `figures/v1_precision_focus/`: five PNG figures for the focused logical-error, exact-agreement,
  valid-sampling, logical-class diversity, and coset-entropy curves.
- `results/v1_hard_cases_precision/`: 11,328 high-repeat hard-case recovery attempts and grouped
  Wilson intervals.
- `figures/v1_hard_cases_precision/hard_case_recovery.png`: high-repeat hard-case recovery plot
  with interval bars.

The generated artifact directories are small enough for review, with no generated file near the
50 MB limit.

## Cautious Readout

These PR artifacts are reproducible benchmarks, not final publication-strength statistical claims.
The p-bit decoders use direct Gibbs/p-bit conditional updates with
`P(e_j = 1 | rest, s) = 1 / (1 + exp(beta Delta_j))`. Averaged over the configured grouped
summaries, exact coset decoding is the best available small-code reference. The larger run improves
the soft p-bit sampler's logical error rate, exact-coset agreement, and valid-sample rate relative
to the earlier artifact. The hard-constrained nullspace sampler remains the strongest exact-
agreement diagnostic baseline and is slightly ahead on mined hard-case recovery.

Mean exact-coset agreement where exact decoding is available:

- `exact_coset`: 1.000
- `nullspace_coset_sampler`: 0.862
- `pbit_annealed_optimizer`: 0.519
- `pbit_coset_sampler`: 0.514

Overall hard-case exact-coset recovery with 95% Wilson intervals:

- `nullspace_coset_sampler`: 0.445 [0.417, 0.473]
- `pbit_coset_sampler`: 0.435 [0.407, 0.463]
- `pbit_annealed_optimizer`: 0.426 [0.398, 0.455]

The precision focus run reduces mean logical-error Wilson half-widths on matching groups from about
0.112 to 0.057 for `pbit_coset_sampler`, a 48.8% reduction. Focused mean logical error rates are:

- `exact_coset`: 0.251, mean half-width 0.047
- `nullspace_coset_sampler`: 0.335, mean half-width 0.052
- `pbit_coset_sampler`: 0.485, mean half-width 0.057
- `pbit_annealed_optimizer`: 0.498, mean half-width 0.057

The high-repeat hard-case run changes the hard-case readout:

- `nullspace_coset_sampler`: 1747/3776 = 0.463 [0.447, 0.479]
- `pbit_annealed_optimizer`: 1731/3776 = 0.458 [0.443, 0.474]
- `pbit_coset_sampler`: 1639/3776 = 0.434 [0.418, 0.450]

The current evidence is therefore not strong enough to claim a hard-case advantage for soft p-bit
coset sampling. It supports a small focused logical-error advantage over annealed optimization in
the configured groups, while the higher-repeat hard-case artifact is a negative result for p-bit
hard-case recovery.

Required interpretation constraints:

- logical success is coset/logical success, not exact physical-error recovery;
- invalid corrections are failures;
- exact-coset agreement is available only where exact enumeration is configured and feasible;
- hard-case recovery is the key test of sampling versus optimization.
