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
uv run python scripts/run_experiments.py --config experiments/configs/v1_pbit_syndrome.json
uv run python scripts/make_plots.py --results results/v1_mwpm_comparison --figures figures/v1_mwpm_comparison
uv run python scripts/make_plots.py --results results/v1_bb144_pbit --figures figures/v1_bb144_pbit
uv run python scripts/make_plots.py --results results/v1_pbit_syndrome --figures figures/v1_pbit_syndrome
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
valid-sample rate, logical classes visited, empirical coset entropy, and runtime. The committed
`v1_large` artifact uses 1000 trials for exact-feasible small groups and 400 trials for medium
groups, bringing every broad-sweep logical-error Wilson half-width below 0.049.

## Precision Focus

The `v1_precision_focus` config uses 400 trials in every configured small and medium topological
group. It is the higher-sample companion to the broader v1-large sweep and brings the worst focused
logical-error Wilson half-width below 0.049.

## Topological MWPM Baseline

The `v1_mwpm_comparison` config compares exact decoding where feasible, PyMatching MWPM,
hard-constrained nullspace sampling, and nullspace-coordinate p-bit sampling on toric and surface
sectors. MWPM is the expected strong baseline on these graphlike topological checks. Medium
topological groups use 400 trials per physical error rate in the committed precision artifact,
which brings the worst logical-error Wilson half-width below 0.049. The topological
nullspace-coordinate p-bit rows use `beta = 0.75`, `updates_per_sample = 32`, and
`update_schedule = "sweep"` after a pilot showed better exact-coset agreement than the older
four-update random schedule.

## BB144 Code-Capacity Probe

The `v1_bb144_pbit` config tests the [[144,12,12]] bivariate-bicycle X-error sector. It compares
particular-start chained-sweep nullspace-coordinate p-bits and BP-guided nullspace-coordinate
p-bits against the plain hard-constrained nullspace sampler and LDPC BP-OSD. The goal is
diagnostic: record whether p-bit coset sampling improves over a validity-only affine baseline,
whether BP guidance closes the gap, and how both compare against a strong qLDPC decoder. It uses
400 trials per physical error rate, which brings the worst BB144 logical-error Wilson half-width
below 0.049. The p-bit BB144 rows sample at `beta = 0.5` for mixing but select the reported coset
with importance-weighted posterior scores targeting `target_beta = 1.0`.

## P-Bit-Native Syndrome Probe

The `v1_pbit_syndrome` config is the first no-linear-solve p-bit experiment. It uses sweep-level
decoder lists so MWPM appears only on graphlike toric/surface sweeps and BP-OSD appears only on the
BB144 sweep. The new `pbit_syndrome` row starts from Bernoulli physical error bits, updates local
checks incrementally, and records whether plain p-bit randomness can find `H e = s` before any
nullspace, BP, or OSD aid is introduced. This first artifact is a 50-trial-per-group pilot with
8 chains, 240 sweeps, annealing from `beta = 0.3` to `1.2`, valid-coset collection enabled, and
minimum-weight selection after up to ten distinct valid physical corrections.

This probe should be interpreted through valid correction rate, first-valid sweep, final residual
weight, logical error, runtime, and valid coset counts. It is intentionally the
no-persistent-disorder baseline; persistent bit/check/edge disorder should be added only after this
baseline shows whether intrinsic Bernoulli firing is enough to break degeneracy.
