# LER-vs-p multi-code campaign wave (2026-07)

Durable archive of the p-bit decoder LER-vs-p campaigns run in the 2026-06/07
session. Originals were written to `/tmp` (ephemeral, wiped on reboot); this
folder is the persisted copy in-repo.

Each leaf holds:
- `RESULT.txt`  -- final pooled LER per p-point, tv=1..5 (score at tv5).
- `progress.txt` -- per-round running error counts + config header.
- `launch.log`  -- driver stdout (where captured).

## Layout

```
surface_d7/{filtered,correlated}     rotated surface code, d=7
surface_d11/{filtered,correlated}    rotated surface code, d=11
bb72/{filtered,correlated}           bicycle-bivariate [[72,12,6]]
bb288/filtered                       bicycle-bivariate [[288,12,18]]
```

## Decoder configs (ColoredBlockGibbsDecoder, geometric beta anneal)

| track | rho | beta | sweeps | chains | max_legs | DEM |
|-------|-----|------|--------|--------|----------|-----|
| surface_d7 filtered   | 0.25 | 0.5->16 | 1000 | 128 | 24 | basis-filtered X |
| surface_d7 correlated | 0.30 | 0.5->6  | 1000 | 128 | 24 | full (basis=none) |
| surface_d11 filtered  | 0.20 | 0.5->12 | 1000 | 128 | 24 | basis-filtered X |
| surface_d11 correlated| 0.30 | 0.5->6  | 1000 | 128 | 24 | full (basis=none) |
| bb72 filtered         | 0.30 | 0.5->6  | 1000 | 128 | 24 | basis-filtered Z (memory_Z) |
| bb72 correlated       | 0.30 | 0.5->6  | 1000 | 128 | 24 | full (memory_Z) |
| bb288 filtered        | 0.30 | 0.5->6  | 1000 | 128 | 24 | basis-filtered Z (memory_Z) |

All campaigns targeted **10 tv5 errors per p-point**, sharded across GPUs 0-5,
resume-safe. Numbers live in each `RESULT.txt`.

## Completeness (as archived)

| track | p-points | status |
|-------|----------|--------|
| surface_d7 filtered   | 0.005-0.001 | complete |
| surface_d7 correlated | 0.005-0.001 | complete |
| surface_d11 filtered  | 0.005-0.001 | complete |
| surface_d11 correlated| 0.005-0.002 | complete |
| bb72 filtered         | 0.005-0.001 | complete |
| bb72 correlated       | 0.005-0.002 | **PARTIAL** (p=0.001 still running at archive time) |
| bb288 filtered        | 0.005-0.002 | complete (p=0.005 anomalous: invalid=77, N=3000) |

### Known gaps
- **bb288 correlated: MISSING** -- never run / not found in `/tmp` or repo. Needs a fresh campaign.
- **bb72 correlated p=0.001**: refresh `RESULT.txt` once the `/tmp/bb72corrX` run reaches `ALLDONE`.

## Provenance (/tmp sources)

| dest | source |
|------|--------|
| surface_d7/filtered    | /tmp/sccamp/d7filtX_g05 |
| surface_d7/correlated  | /tmp/sccamp/d7corr |
| surface_d11/filtered   | /tmp/sccamp/filtX |
| surface_d11/correlated | /tmp/sccamp/corr |
| bb72/filtered          | /tmp/bb72filtZ |
| bb72/correlated        | /tmp/bb72corrX |
| bb288/filtered         | /tmp/zcamp288 |

## Regeneration

Campaign drivers live in `../ler_vs_p_bb144_2026-06/scripts/`
(`sc7_*`, `sc11_*`, `bb72_*`, `filtered_campaign288.sh`). Override `OUT=` to set
the output dir, e.g.:

```bash
OUT=/tmp/sccamp/d7corr setsid bash ../ler_vs_p_bb144_2026-06/scripts/sc7_correlated_campaign.sh &
```
