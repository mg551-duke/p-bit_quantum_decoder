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

**Takeaway (superseded below).** The +26-33% was a *masking bug*, not a
fundamental limit -- see the fix in the next section, which makes gating
cost-neutral. The prior-scaling knobs (`gamma_active`/`gamma_far`) are a separate
lever, characterized below.

## 2026-07-02 (PDT) -- Remedy #1 follow-up: masking fix + `gamma_active` sweep

**Masking fix** (`2da5b87`). The extra RNG draw was the culprit. Fold the Gibbs
idle gate into the resample probability using the *same* uniform -- idle bits get
`p1' = (1-rate)*bit + rate*p1` (rate=0 => frozen), which is distribution-
equivalent to gating the whole update with probability `rate` but needs no second
draw. Metropolis keeps its 2-uniform gate. Applied identically to the CUDA
kernel, reference `_sweep`, and the fused launcher; `fused==reference` stays
bit-for-bit (validated baseline/hard/soft/prior-only/soft+prior).

Post-fix isolated gating A/B (bb72 correlated, `p=0.003`, 10k):

| config | ms/shot (before -> after) | tv5 LER |
|---|---|---|
| baseline | 24.09 -> **23.85** | 2.7e-3 |
| hard (idle=0.0) | 32.05 (+33%) -> **24.96 (+4.7%)** | 2.3e-3 |
| soft (idle=0.05) | 30.44 (+26%) -> **23.74 (~0%)** | 2.5e-3 |

=> gating is now **cost-neutral and LER-neutral**; the residual +4.7% for hard is
an occasional extra restart leg.

**`gamma_active` sweep** (prior strength on OFF bits adjacent to a violated
detector -- the candidates to turn ON). bb72 correlated, `p=0.003`:

| `gamma_active` | tv5 LER | ms/shot | note |
|---|---|---|---|
| 0.3 / 0.5 | ~0.99 (meanW 293-746) | 46-54 | **catastrophic** (wrong coset) |
| 0.7 | 3.5e-3 | 20.1 (-24%) | cheaper, worse LER |
| **1.0 (baseline)** | **2.4e-3** | 26.3 | **LER-optimal** |
| 1.1 / 1.2 | ~2.6-2.9e-3 (same) | 36 / 50 | same LER, more cost |
| 1.3 | 3.5e-3 | 88 (legs_max 17) | worse LER AND cost |
| >=1.5 | -- | -- | impractically slow |

**Takeaway.** Active-region (gating + prior scaling) is a **cost/robustness
dial, not an accuracy lever**: `gamma_active=1.0` is LER-optimal (min-weight
selection already yields the sparsest valid correction, so sharpening the prior
only makes convergence harder); `<1` trades LER for speed (cliff below ~0.6);
`>1.2` degrades both. `gamma_far` barely matters (idle bits are already pinned to
0 by the syndrome term). **To improve LER the lever is ensemble budget
(chains/legs), not active-region.** The masking fix stands as a real correctness/
efficiency win regardless.

## 2026-07-02 (PDT) -- Remedy #1 at bb144 scale (d=12)

Throughput comparison, bb144 correlated full-DEM memory_Z (1728 det / 67752 mech
/ 12 obs), `p=0.003` (representative sub-threshold point; mean detector
weight/shot 181), 3000 shots, seed 12345, baseline config. Sub-threshold LER
(~8e-5) needs a full campaign so only cost is read here (0-1 events/3k):

| config | ms/shot | vs base | sw_exec | legs_max | meanW |
|---|---|---|---|---|---|
| baseline | **263.3** | -- | 787 | 24 | 50.1 |
| hard (idle=0.0) | 267.9 | +1.8% | 820 | 24 | 50.0 |
| soft (idle=0.05) | 269.4 | +2.3% | 842 | 24 | 50.0 |
| **ga0.7** | **132.8** | **-49.6%** | 551 | 1 | 52.9 |

- **Masking fix holds at scale**: gating is cost-neutral on bb144 too (+1.8-2.3%,
  was +26-33% before the fix).
- **`ga0.7` halves cost** on bb144 (-49.6%, vs -24% on bb72) for only +5.6% meanW.
  Reason: baseline bb144 hits `legs_max=24` (many shots exhaust all restart legs
  under the strong prior) whereas `ga0.7` converges in a single leg (`legs_max=1`,
  `sw_exec` 551 vs 787). The weaker near-syndrome prior makes the dense bb144
  syndrome far easier to satisfy.

**Bottom line.** Active-region is a **throughput/robustness lever, not an accuracy
lever.** For raw speed on the expensive code, `gamma_active~0.7` is a big win
(~2x on bb144) at a modest accuracy cost (bb72 showed +46% LER; the bb144 LER
cost is unquantified and needs a campaign). For **best LER**, keep
`gamma_active=1.0` and use the masking-fixed gating freely (now free). Neither
gating nor prior scaling lowers LER below baseline.
