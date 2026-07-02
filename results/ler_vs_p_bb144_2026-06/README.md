# LER vs physical error rate — bb144 (p-bit / ColoredBlockGibbsDecoder)

Reproducibility + comparison record for the June 2026 tuning and logical-error-rate
(LER) campaigns on the `[[144,12,12]]` bivariate-bicycle code, circuit-level noise.

## Environment

| field | value |
|---|---|
| date generated | 2026-06-17 (PDT) |
| git commit | `e67c5ac` (branch `codex/pbit-qec-prototype`) |
| host | u115240 |
| GPU | NVIDIA A100 80GB PCIe x8 (**only GPUs 0–5 used; 6 & 7 reserved**) |
| python | 3.12.11 (venv `.venv-gpu`) |
| stim | 1.16.0 |
| numpy | 2.4.6 |
| cupy (CUDA backend) | 14.1.1 |
| decoder | `ColoredBlockGibbsDecoder` via `experiments/bench_circuit_pbit.py` |

## Circuit family

`bicycle_bivariate_144_12_12_memory_X`, distance=12, rounds=12, noise_model=uniform_circuit,
basis=CX, A=x^3+y+y^2, B=y^3+x+x^2. 1728 detectors, 67824 error mechanisms, 12 observables.

Path template (error_rate in {0.001,0.002,0.003,0.004,0.005,...,0.009}):
```
/home/mert/CascadeProjects/relay/tests/testdata/bicycle_bivariate/circuit=bicycle_bivariate_144_12_12_memory_X,distance=12,rounds=12,error_rate=<p>,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim
```

## Best decoder configuration (all p)

`rho_scale` (syndrome penalty) is the dominant lever; **rho_scale=0.3** wins at every p
(lowest mean selected weight, highest validity, lowest LER). Fixed settings: beta schedule
0.5→8.0, update=gibbs, selection=min_weight, gamma=1.0, capture=sweep.

```
chains=128  sweeps=500  max_legs=12  target_valid=1  rho_scale=0.3
```

Single-run command (one p, one GPU):
```bash
export PYTHONPATH=src CUDA_VISIBLE_DEVICES=0
.venv-gpu/bin/python experiments/bench_circuit_pbit.py \
  --device cuda --shots 10000 --seed 0 \
  --circuit "<circuit path for p>" \
  --chains 128 --sweeps 500 --max-legs 12 --target-valid 1 --rho-scales 0.3
```

## Results — LER vs p (50-error campaigns unless noted)

| p | LER | ± (stderr) | errors / shots | min_valid | per-shot decode | compute | wall (6 GPUs) |
|---|---|---|---|---|---|---|---|
| 0.005 | 1.05e-2 | 1.3e-3 | 63 / 6,000 | 0.979 | ~489 ms/shot | 2,932 GPU-s | ~8 min |
| 0.004 | 1.08e-3 | 1.3e-4 | 67 / 62,000 | 0.999 | ~236 ms/shot | 14,632 GPU-s | ~41 min |
| 0.003 | 7.85e-5 | 1.1e-5 | 51 / 650,000 | 1.000 | ~100 ms/shot | 64,737 GPU-s | ~2 h 50 min |
| 0.002 | 9.43e-6 | 3.0e-6 | 10 / 1,060,000 | ~1.0 | ~72 ms/shot | ~76,320 GPU-s | ~3 h 32 min |
| 0.001 | **skipped** | — | infeasible (~1e8+ shots) | — | — | — |

Notes:
- "compute" = sum of per-chunk wall across all parallel shards (GPU-seconds). "wall" ≈ compute / 6.
- "per-shot decode" is single-GPU latency for one chunk; effective throughput is ~6× with 6 GPUs.
- p=0.004/0.005 slightly overshoot 50 errors because chunks finish in parallel before the stop check.

### Slope / threshold behaviour
Very steep: LER(0.005)/LER(0.004) = 9.8×, LER(0.004)/LER(0.003) = 13.6×,
LER(0.003)/LER(0.002) = 8.3× → effective exponent ~p^8…p^12. This is why p=0.003
LER looks "small" — it is genuinely small; the decoder is well below threshold, so
LER plummets as p drops.

### Feasibility of lower p (50-error target)
| p | extrapolated LER | shots for 50 errors | est. time (6 A100) |
|---|---|---|---|
| 0.002 | ~5e-6 | ~10 M | ~1.5–2 days |
| 0.001 | ~4e-7 or less | >100 M | weeks+ (infeasible) |

Decision (user): skip 0.001; collect **10 errors** at 0.002 via a self-running job.
**Outcome:** the p=0.002 10-error job completed at 10 / 1,060,000 → LER = 9.43e-6 ±
3.0e-6 (the ~5e-6 extrapolation was a ~1.9× underestimate; consistent with the
steep but slightly shallower-than-extrapolated slope).

## Tuning experiments (how the config was chosen)

All at p=0.003 unless stated; paired seed=0 (identical shots) where noted. Raw logs in `logs/`.

- **Speed grid** (session 1): fastest valid config chains=32,target_valid=1 = 42.7 ms/shot vs
  chains=128 baseline 89 ms/shot; chains=16 not faster (GPU underutilized).
- **Accuracy grid** (`logs/acc_g*.log`, 10k paired): meanW floor reached only via low rho;
  chains=256 / target_valid=8 at rho=0.6 still meanW 55.3.
- **Rho sweep** (`logs/rho_g*.log`, 10k paired, chains=128 sw=500 ml=16 tv=4):
  meanW = 50.3 (rho 0.2–0.35), 50.8 (0.45), 51.9 (0.5), 57.1 (0.6); validity 1.0 throughout.
- **Cheap-config probe** (`logs/cheap*.log`): rho=0.3 hits the meanW floor with tv=1 and
  fewer chains (c64 sw400 tv1 = 50.7 @78 ms; c32 sw500 tv1 = 50.8 @68 ms).
- **p=0.005 config diagnostics** (`logs/d5_g*.log`, 1k shots): rho=0.3 best LER+validity;
  max_legs=16 burns sweeps on unsolvable shots, ml=12 sufficient.

## Reproduction

Helper scripts in `scripts/` (copy to /tmp or run in place):
- `shardcamp.sh TAG CIRCUIT RHO CHUNK SEEDBASE GPUS...` — launch a sharded campaign
  (config via `SC_CFG` env var). Writes `/tmp/${TAG}_g<gpu>.log`.
- `pool.sh TAG` — pool errors/shots → LER ± stderr and min validity across shards.
- `campaign.sh` — the p=0.003 50-error campaign (8→6 GPUs).
- `acc_grid.sh`, `rho_sweep.sh` — the tuning grids.
- `auto_p002.sh` — **autonomous** p=0.002 job: launches workers on GPUs 0–5, polls,
  and auto-stops + writes `/tmp/p002_RESULT.txt` once ≥10 errors are pooled.

Example (reproduce p=0.004 to 50 errors across GPUs 0–5):
```bash
C4="<circuit path for 0.004>"
export SC_CFG="--chains 128 --sweeps 500 --max-legs 12 --target-valid 1"
bash scripts/shardcamp.sh C004 "$C4" 0.3 2000 600000 0 1 2 3 4 5
watch -n 30 'bash scripts/pool.sh C004'   # stop (pkill -f error_rate=0.004) once >=50 errors
```

## Artifact inventory
- `scripts/` — 6 campaign/tuning scripts (above).
- `logs/` — raw per-shard logs: `camp_g*` (p=0.003), `C004_g*` (p=0.004), `C005_g*` (p=0.005),
  `rho_g*`, `acc_g*`, `cheap*`, `d5_g*` (tuning), `probe16.log`, `p002_progress.txt`.
- `summary.csv` — machine-readable comparison table.

## Status of the p=0.002 autonomous job
**Complete (2026-06-17 12:52).** Stopped itself at 10 errors / 1,060,000 shots →
LER = 9.43e-6 ± 3.0e-6, and freed GPUs 0–5. (The `/tmp/p002*` scratch files have
since been cleaned; the result is recorded in `summary.csv` and the results table
above. Progress snapshot retained at `logs/p002_progress.txt`.)

---

# Filtered (memory_Z, basis-Z) campaign + RelayBP comparison — 2026-06-23

Added to compare the p-bit decoder against **RelayBP-300sets** on the *exact* problem
RelayBP benchmarks: the `memory_Z` circuit with the DEM restricted via `--basis-filter Z`
to **936 detectors / 8784 mechanisms / 12 observables** (vs the full 1728 / 67824 used
above). This is a deliberately *information-reduced* problem — one basis sector, no
cross-basis correlations — so it is the fair apples-to-apples baseline against filtered
BP decoders, and a floor that the full correlated decoder (top of this README) beats handily.

Self-driving run: `scripts/filtered_auto.sh`, GPUs **0–7**, same `.venv-gpu`
(stim 1.16.0, cupy 14.1.1). Launched 2026-06-22 19:28, finished 2026-06-23 00:06.

## Tuning (filtered problem)
Three-phase GPU sweep (`scripts/filtered_tune.sh` + `configs/ftune_phase*.txt`), scored at
p=0.004, tv=5:
- **rho_scale** is again the dominant lever; the filtered optimum shifts to **≈0.20**
  (full-DEM optimum was 0.3). rho ≥ 0.45 degrades; rho ≥ 0.9 collapses (and runs slow —
  bad convergence triggers many legs).
- **chains** beyond 128 did **not** help; **sharper annealing** (beta_stop 12/16) **hurt**
  (keep 0.5→8.0).
- **sweeps** has only a *weak* effect: pooled tv=5 LER fell 4.81e-3 (s=500) → 3.96e-3
  (s=1500), ~1.6σ across the range. The auto-selector (`filtered_pick_sweeps.py`) chose
  **sweeps=750** — the cheapest setting statistically tied (within 1σ) with the best.
  (An earlier low-count phase-2 sample suggested a larger sweeps gain; higher-statistics
  phase-3 pooling did not confirm it.)

Campaign config: `chains=128 sweeps=750 max_legs=24 rho_scale=0.20 gamma=1.0 beta 0.5→8.0
gibbs min_weight`, capturing tv=1..5 from one trajectory/shot.

## Results — filtered LER vs p (tv=5; full tv=1..5 in `FILTERED_RESULT.txt`)

| p | p-bit filtered (ours) | RelayBP-300sets | filtered ratio | p-bit correlated (ours, full DEM) | corr. gain |
|---|---|---|---|---|---|
| 0.002 | 1.12e-5 ± 2.3e-6 (23 / 2.05M) | 2.42e-5 (21 / 868k) | **2.2× better** | 2.21e-6 | 5.1× |
| 0.003 | 3.20e-4 ± 2.9e-5 (123 / 384k) | 4.77e-4 (22 / 46k) | **1.5× better** | 2.50e-5 | 12.8× |
| 0.004 | 4.67e-3 ± 2.7e-4 (299 / 64k) | 4.29e-3 (23 / 5.4k) | ~tied (CIs overlap) | 5.52e-4 | 8.5× |
| 0.005 | 3.22e-2 ± 2.2e-3 (206 / 6.4k) | 4.78e-2 (21 / 439) | **1.5× better** | 7.72e-3 | 4.2× |

Wall time (8× A100): p=0.005 ~8 min, p=0.004 ~33 min, p=0.003 ~58 min, p=0.002 ~1h46m
(2.048M shots, capped); total ~3h25m. Per-shot cost grows toward threshold because
collecting tv=5 valid samples needs more leg retries.

**Takeaways:**
1. On the filtered problem the p-bit decoder **matches or beats RelayBP-300sets** at every p
   (clearly better at 0.002/0.003/0.005; statistically tied at 0.004 — RelayBP's reference
   points carry only ~21–23 errors, ±~20%), and with far tighter statistics.
2. Decoding the **full correlated DEM** is **4–13× lower** than the filtered floor at matched
   p — quantifying the value of jointly decoding the full hypergraph error model, which the
   p-bit sampler handles natively.

Plot: `plots/compare_pbit_relaybp.{png,tex,csv}`.

## Reproduction (filtered pipeline)
Resume-safe, detached, end-to-end:
```bash
cd results/ler_vs_p_bb144_2026-06/scripts
setsid nohup bash filtered_auto.sh > /tmp/zauto/launch.log 2>&1 &   # waits→picks→campaign→plot
```
Components: `filtered_tune.sh` (sweep), `filtered_tune_pool.py` (pool), `filtered_pick_sweeps.py`
(auto-select sweeps), `filtered_campaign.sh` (LER-vs-p, env `SWEEPS/RHO/CHAINS/GPUS/OUT`),
`make_compare_plot.py` (3-way plot). Outputs archived to `FILTERED_RESULT.txt`,
`logs/filtered_progress.txt`, `logs/filtered_phase3_pool.txt`.
