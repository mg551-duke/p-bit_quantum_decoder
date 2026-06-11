# Experiments

The repository provides three CLIs:

```bash
uv run python scripts/run_experiments.py --config experiments/configs/v1_large.json
uv run python scripts/find_hard_cases.py --config experiments/configs/v1_hard_cases.json
uv run python scripts/make_plots.py --results results/v1_large --figures figures/v1_large
```

Additional committed comparison configs are:

```bash
uv run python scripts/run_experiments.py --config experiments/configs/v1_mwpm_comparison.json
uv run python scripts/run_experiments.py --config experiments/configs/v1_bb144_pbit.json
uv run python scripts/make_plots.py --results results/v1_mwpm_comparison --figures figures/v1_mwpm_comparison
uv run python scripts/make_plots.py --results results/v1_bb144_pbit --figures figures/v1_bb144_pbit
```

## Exact Small-Code Validation

Toric `L=3` and surface `d=3` runs include exact coset decoding. The key metric is agreement with
the exact maximum-posterior coset.

## Sampling Versus Optimization

Finite-temperature coset sampling is compared against annealed minimum-energy decoding over
`p = 0.02, 0.04, ..., 0.18`.

## Degeneracy Stress Test

The hard-case miner enumerates small-code syndromes and records cases where the exact
maximum-posterior coset differs from the minimum-energy coset. These are the central scientific
cases for the project.

## Medium Scaling

Medium toric and surface-code runs omit exact decoding and compare logical error rate,
valid-sample rate, logical classes visited, empirical coset entropy, and runtime.

## Topological MWPM Baseline

The `v1_mwpm_comparison` config compares exact decoding where feasible, PyMatching MWPM,
hard-constrained nullspace sampling, and nullspace-coordinate p-bit sampling on toric and surface
sectors. MWPM is the expected strong baseline on these graphlike topological checks.

## BB144 Code-Capacity Probe

The `v1_bb144_pbit` config tests the [[144,12,12]] bivariate-bicycle X-error sector. It compares
particular-start nullspace-coordinate p-bits against the plain hard-constrained nullspace sampler.
The goal is diagnostic: record whether p-bit coset sampling improves over a validity-only affine
baseline, without claiming exact maximum-likelihood performance.
