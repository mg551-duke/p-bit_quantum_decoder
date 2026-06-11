# Experiments

The repository provides three CLIs:

```bash
uv run python scripts/run_experiments.py --config experiments/configs/v1_large.json
uv run python scripts/find_hard_cases.py --config experiments/configs/v1_hard_cases.json
uv run python scripts/make_plots.py --results results/v1_large --figures figures/v1_large
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
