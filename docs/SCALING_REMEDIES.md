# Scaling Remedies Log

Running log of work on the QLDPC p-bit decoder's circuit-level **scaling
problem**: as code distance grows, the DEM gains more fault mechanisms,
detectors, and residual configurations, so the update budget
`chains x sweeps x legs` needed to hit a valid syndrome-consistent state can
grow too fast. This file records the baseline and each remedy direction.

## 2026-07-02 (PDT) -- Baseline: v0.1 basic first-draft

Tagged `v0.1-basic-draft`. This commit is the **basic, first-draft decoder**
(constraint-annealed chromatic block-Gibbs, replicas + restart legs, min-weight
valid selection) evaluated under code-capacity and circuit-level noise. It is
the branch point from which we start trying scaling remedies; subsequent
directions are explored on separate branches off `main`.

Baseline circuit-level operating point (correlated / full-DEM, `ColoredBlockGibbsDecoder`):
`chains=128, max_legs=24, rho_scale=0.30, beta 0.5->6 (geometric), sweeps=1000`,
selection `min_weight`, target 10 tv5 errors/point.

Reference LERs (shot-level, from the 2026-06/07 campaigns):
- **bb72 correlated**: `2.13e-3 @p=0.003`, `2.22e-4 @p=0.002`, `1.25e-5 @p=0.001`.
- **bb144 correlated**: `7.85e-5 @p=0.003` (~65k GPU-s), `9.43e-6 @p=0.002` (~76k GPU-s); `p=0.001` skipped as infeasible.
- **bb288 correlated**: `~2.1e-7 @p=0.004` per-round (expensive; 206k mechanisms, 299 colors).

### Planned directions
1. **Active-region updates (first).** Concentrate stochastic updates where they
   matter: a mechanism `j` is active iff `e_j=1` or `v_j>0` (`v_j` = adjacent
   violated detectors). Idle bits (`e_j=0, v_j=0`) are updated only with a small
   probability `idle_update_rate` (soft gating), and/or made more stable via
   active/far prior scaling (`gamma_active` near the syndrome, `gamma_far` away).
   Already present in the CPU `PBitSyndromeDecoder`; port to the GPU
   `ColoredBlockGibbsDecoder` and benchmark on bb72 -> bb144 correlated (cost
   down, LER equal-or-better).
2. Controlled disorder / asymmetry across replicas (perturbed priors,
   heterogeneous schedules, randomized color orders).
3. Decoder theory (mixing, first-passage to the valid manifold, logical-sector
   coverage within a finite budget).

## 2026-07-02 (PDT) -- Remedy #1 result: flip-gating is a GPU cost ADDER

Branches `scaling-remedy-{soft,hard}-active-gating` (shared implementation at
`06572d6`). GPU fused==reference validated bit-for-bit for baseline, hard
(`idle_update_rate=0.0`) and soft (`0.05`).

**A/B benchmark** (bb72 correlated full-DEM memory_Z, `p=0.003`, 10000 shots,
seed 12345, baseline config above, one config per GPU 0/1/2):

| config | tv5 LER | ms/shot | sw_exec | legs_max |
|---|---|---|---|---|
| baseline (idle=1.0) | 2.7e-3 (27/10k) | **24.09** | 711 | 1 |
| hard (idle=0.0) | 2.1e-3 (21/10k) | 32.05 (+33%) | 744 | 2 |
| soft (idle=0.05) | 2.3e-3 (23/10k) | 30.44 (+26%) | 713 | 1 |

**LER is neutral** (27/21/23 events overlap within CI at `gamma=1.0`), but
**cost went UP, not down.**

**Root cause.** In the dense vectorized kernel, gating is applied as a *mask
after* every column's flip is computed (`flip &= active | idle_uniforms<rate`),
and idle gating draws an *extra* `(cols x rows)` uniform each color each sweep
(doubles RNG). So flip-suppression strictly *adds* work; it cannot reduce
compute because all columns are still processed. Hard gating can also force
extra restart legs by freezing a bit a shot needed to move (`legs_max=2`).

**Takeaway.** Flip-suppression active-region gating is not a GPU cost remedy
(it is LER-neutral overhead). Real savings require cutting the **column count**
processed: e.g. *static per-shot active-region pruning* -- at init keep only
columns within radius-`r` (1-2 Tanner hops) of a violated detector, freeze the
rest for the whole decode (done once, shrinks the problem at low `p`). The
prior-scaling knobs (`gamma_active`/`gamma_far`) are a separate *accuracy* lever
and have not yet been tested for LER improvement.
