# p-bit QEC Decoder Prototype

This repository is a research-grade prototype for testing whether finite-temperature
probabilistic-Ising dynamics can help degenerate quantum error correction decoding.

The central hypothesis is:

> QEC decoding should choose the most likely logical/coset class, not necessarily the single
> lowest-energy correction. Sampling many syndrome-compatible corrections may therefore beat
> pure Ising-style optimization on degenerate instances.

For one Pauli sector, the object of interest is:

```text
P(l | s) proportional to sum_{e: H e = s, A e = l} exp[-E_prior(e)]
```

The prototype implements small toric and surface-code sectors, exact small-code coset
enumeration, finite-temperature p-bit/Gibbs conditional sampling, a hard-constrained
nullspace-coordinate p-bit sampler, annealed optimization baselines, and reproducible CSV/plot
generation.

## Install

```bash
uv sync --extra dev
```

## Reproduce

Smoke run used by CI:

```bash
uv run python scripts/run_experiments.py --config experiments/configs/v1_smoke.json
uv run python scripts/find_hard_cases.py --config experiments/configs/v1_hard_cases.json
uv run python scripts/make_plots.py --results results/v1_large --figures figures/v1_large
```

PR artifact run:

```bash
uv run python scripts/run_experiments.py --config experiments/configs/v1_large.json
uv run python scripts/find_hard_cases.py --config experiments/configs/v1_hard_cases.json
uv run python scripts/make_plots.py --results results/v1_large --figures figures/v1_large
```

Focused precision artifacts:

```bash
uv run python scripts/run_experiments.py --config experiments/configs/v1_precision_focus.json
uv run python scripts/find_hard_cases.py --config experiments/configs/v1_hard_cases_precision.json
uv run python scripts/tune_pbit_nullspace.py --config experiments/configs/v1_pbit_nullspace_calibration.json
uv run python scripts/make_plots.py --results results/v1_precision_focus --figures figures/v1_precision_focus
uv run python scripts/make_plots.py --results results/v1_hard_cases_precision --figures figures/v1_hard_cases_precision
```

The heavier target from the original study brief is encoded in
`experiments/configs/v1_full_target.json`; it keeps the 1000/300 trial target and the default
16-chain/1000-sweep sampler settings, but is not used by CI.

## Outputs

HTML report:

- `docs/results/index.html`

Generated CSVs live in `results/v1_large/`:

- `summary.csv`
- `trials.csv`
- `coset_distributions.csv`
- `hard_cases.csv`
- `hard_case_recovery.csv`
- `hard_case_summary.csv`

Focused precision CSVs live in `results/v1_precision_focus/` and
`results/v1_hard_cases_precision/`, including `pbit_nullspace_calibration.csv` for the tuned
nullspace-coordinate p-bit setting.

Generated figures live in `figures/v1_large/`:

- logical error rate
- exact coset agreement
- syndrome-valid sample rate
- distinct logical classes visited
- empirical coset entropy
- hard-case recovery behavior

Focused precision figures live in `figures/v1_precision_focus/` and
`figures/v1_hard_cases_precision/`.

## Current Readout

- The v1-large run now includes `pbit_nullspace_sampler`, giving 18,000 decoder-trial rows and
  9,440 mined hard-case recovery attempts.
- The precision focus run adds 18,200 decoder-trial rows with 260 trials in every configured
  group.
- The p-bit sampler's focused logical-error confidence half-width drops from about 0.112 to 0.057
  on matched groups.
- Focused mean logical error is 0.485 for `pbit_coset_sampler` and 0.498 for
  `pbit_annealed_optimizer`.
- The high-repeat hard-case run has 37,760 recovery attempts. It supports a hard-case advantage for
  syndrome-valid nullspace-coordinate p-bit sampling: `pbit_nullspace_sampler` recovers
  0.574 [0.564, 0.583], while `nullspace_coset_sampler` recovers 0.469 [0.459, 0.479],
  `pbit_annealed_optimizer` recovers 0.460 [0.449, 0.470], and `pbit_coset_sampler` recovers
  0.432 [0.422, 0.442].
- The tuned nullspace-coordinate p-bit setting is `beta = 0.75` with four updates per collected
  sample, selected from `results/v1_hard_cases_precision/pbit_nullspace_calibration.csv`.

## Interpretation Rules

- A decoder succeeds only when `A(e_true xor e_hat) = 0 mod 2`.
- Equality to the sampled physical error is not used as a success criterion.
- The soft physical-bit p-bit sampler updates each binary error bit with
  `P(e_j = 1 | rest, s) = 1 / (1 + exp(beta * Delta_j))`.
- The hard-constrained p-bit sampler uses the same logistic conditional on affine nullspace
  coordinates, so every collected correction is syndrome-valid.
- Invalid decoder outputs count as logical failures.
- Positive claims require exact-coset agreement or logical-success evidence in the CSVs.
- Negative outcomes should be interpreted through valid-sample rate, mixing, coset entropy,
  runtime, and hard-case recovery, not only final logical error rate.

## Literature Chain

- [Roffe et al.](https://arxiv.org/abs/1903.10254): QEC-to-Ising mapping and maximum-entropy framing.
- [Hutter, Wootton, and Loss](https://arxiv.org/abs/1302.2669): MCMC decoding for the surface code.
- [Bravyi, Suchara, and Vargo](https://arxiv.org/abs/1405.4883): maximum-likelihood surface-code decoding.
- [Fujisaki et al.](https://arxiv.org/abs/2308.00369): practical QEC decoding as Ising optimization.
- [Poulin and Chung](https://authors.library.caltech.edu/records/fgska-n5w47): BP failures from short cycles and degeneracy.
- [Panteleev and Kalachev](https://quantum-journal.org/papers/q-2021-11-22-585/): BP+OSD improvements for degenerate QLDPC codes.
- [Kaiser and Datta](https://pubs.aip.org/aip/apl/article/119/15/150503/40486/Probabilistic-computing-with-p-bits): p-computers for randomized algorithms.
- [Camsari, Sutton, and Datta](https://arxiv.org/abs/1809.04028): p-bits for probabilistic spin logic.
- [Aadit et al.](https://arxiv.org/abs/2110.02481): massively parallel sparse p-bit Ising-machine sampling.

## Development

```bash
uv run ruff check .
uv run pytest
```
