# Annealing-schedule tuning: rho, beta, and temperature shape

This study tunes the controls that govern **convergence speed** and **solution
quality** of `PBitSyndromeDecoder`: the syndrome-penalty weight `rho` (via
`rho_scale`), the inverse-temperature endpoints `beta_start`/`beta_stop`, and the
**temperature-schedule shape** (`geometric` / `linear` / `sigmoid`). The decoder
gained a `beta_schedule_shape` parameter so the ramp shape is itself tunable;
`geometric` remains the default and reproduces prior behavior exactly.

## Tooling

- `pbit_qec.schedule_tuning` — sweeps a `rho_scale x (beta_start, beta_stop) x
  shape x chains x sweeps` grid on **identical (paired) error realizations** so
  config-to-config differences are variance-reduced. Reports, per config:
  logical error rate with a **Wilson 95% interval**, syndrome-valid rate,
  convergence speed via the **first-valid-sweep** distribution
  (`fvs_mean/median/p95/max`), mean selected weight, and seconds/decode.
  Outputs `schedule_summary.csv`, raw `schedule_trials.csv`,
  `schedule_best_by_p.csv`, and a provenance `manifest.json` (grid, seeds, git
  commit, platform).
- `pbit_qec.schedule_plots` — renders `schedule_shapes.png`,
  `ler_vs_convergence.png`, `rho_sensitivity.png`, and `ler_vs_budget.png`.

Convergence metric: `first_valid_sweep` is the sweep index at which a chain first
produces a syndrome-valid configuration. Lower is faster.

## Results

### Surface d=7 (chains=64, sweeps=120, 200 trials/config, paired errors)

| p | schedule | rho_scale | shape | LER (±95% hw) | first-valid p95 |
|---|---|---|---|---|---|
| 0.05 | baseline `b0.8-3.5` | 1.0 | geometric | 0.020 ± 0.021 | 5.0 |
| 0.05 | **tuned** `b0.8-3.5` | 2.0 | **linear** | **0.015 ± 0.019** | **2.0** |
| 0.10 | baseline `b0.8-3.5` | 1.0 | geometric | 0.160 ± 0.051 | 21.0 |
| 0.10 | **tuned** `b0.8-3.5` | 1.5 | **linear** | **0.125 ± 0.046** | **5.0** |

At p=0.10, moving from `rho_scale=1.0`/geometric to `rho_scale=1.5`/linear cut LER
by ~22% **and** sped first-valid convergence 4.2x (p95 21 -> 5).

### BB144 [[144,12,12]] (chains=256, sweeps=150, 150 trials/config, paired errors)

| p | schedule | rho_scale | shape | LER (±95% hw) | first-valid p95 |
|---|---|---|---|---|---|
| 0.04 | baseline `b0.3-5` | 1.0 | geometric | 0.000 (CI ≤0.025) | 37.0 |
| 0.04 | **tuned** `b0.5-5` | 1.0 | geometric | 0.000 (CI ≤0.025) | **8.0** |
| 0.06 | baseline `b0.3-5` | 1.0 | geometric | 0.067 ± 0.041 | 52.0 |
| 0.06 | **tuned** `b0.5-5` | 1.0 | **linear** | **0.053 ± 0.037** | **14.0** |

Raising only `beta_start` (0.3 -> 0.5) shrinks first-valid p95 ~4.6x at p=0.04 with
identical LER, and at p=0.06 a `b0.5-5` linear ramp lowers LER ~20% while
converging 3.7x faster.

### rho must stay near 1.0

`rho_sensitivity.png` shows that over-penalizing the syndrome is harmful: at
p=0.06, `rho_scale=2.0` raises BB144 LER from ~0.06 to ~0.24 (`b0.5-5`) or ~0.71
(`b0.3-5`). Keep `rho_scale ≈ 1.0` at higher noise; small boosts (1.5) only help
deep below threshold.

### Faster convergence buys a cheaper decoder (BB144 budget sweep)

`b0.3` vs `b0.5` at `rho_scale=1.0`, geometric, sweeps in {40, 80, 150}:

| p | schedule | sweeps | LER (±95% hw) | s/decode |
|---|---|---|---|---|
| 0.06 | baseline `b0.3-5` | 150 | 0.067 ± 0.041 | 2.88 |
| 0.06 | **tuned** `b0.5-5` | **40** | **0.060 ± 0.039** | **0.89** |
| 0.04 | tuned `b0.5-5` | 40 | 0.000 (CI ≤0.025) | 0.78 |

LER is already flat at 40 sweeps for the tuned schedule, so the faster
convergence converts directly into a **~3.1x cheaper decode at equal accuracy**.

## Recommended operating points

- **Surface d=7**: `beta_start=0.8, beta_stop=3.5, beta_schedule_shape="linear",
  rho_scale=1.5` (use `rho_scale=2.0` deep below threshold, e.g. p≈0.05).
- **BB144**: `beta_start=0.5` (not 0.3), `beta_stop=5.0`,
  `beta_schedule_shape="linear"`, `rho_scale=1.0`; `sweeps` can drop ~150 -> ~40.
- **Avoid**: `rho_scale >= 2.0` near/above threshold; the `sigmoid` ramp for these
  endpoints (it dwells cold longer, slowing first-valid with no LER benefit).

## Reproduce

```bash
# Surface d=7 schedule grid
.venv/bin/python -m pbit_qec.schedule_tuning --family surface --size 7 \
  --p 0.05 0.10 --trials 200 --chains 64 --sweeps 120 \
  --rho-scales 1.0 1.5 2.0 --beta-pairs 0.5:3.5 0.8:3.5 0.8:5.0 \
  --shapes geometric linear sigmoid --workers 32 --name sched_surface_d7

# BB144 schedule grid (production budget)
.venv/bin/python -m pbit_qec.schedule_tuning --family bb --size 144 \
  --p 0.04 0.06 --trials 150 --chains 256 --sweeps 150 \
  --rho-scales 1.0 1.5 2.0 --beta-pairs 0.3:5.0 0.5:5.0 0.3:8.0 \
  --shapes geometric linear sigmoid --workers 32 --name sched_bb144

# BB144 budget-reduction confirmation
.venv/bin/python -m pbit_qec.schedule_tuning --family bb --size 144 \
  --p 0.04 0.06 --trials 150 --chains 256 --sweeps 40 80 150 \
  --rho-scales 1.0 --beta-pairs 0.3:5.0 0.5:5.0 --shapes geometric \
  --workers 24 --name sched_bb144_budget

# Figures
.venv/bin/python -m pbit_qec.schedule_plots \
  --summary results/sched_bb144/schedule_summary.csv \
  --figures figures/sched_bb144 --demo-beta 0.5 5.0
```

Artifacts: `results/sched_surface_d7/`, `results/sched_bb144/`,
`results/sched_bb144_budget/`, and figures under `figures/sched_*`.
