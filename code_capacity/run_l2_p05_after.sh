#!/usr/bin/env bash
# Follow-up to run_l1l2_sharpen.sh: once the L1/L2 p=0.04 sharpen finishes, collect
# >=10 pooled CONTENT errors for L2 at p=0.05 (only 2 in the fixed-budget grid).
# Waits on the "sharpen ALL DONE" marker so it never 3-way-contends the GPUs, then
# runs one L2 pool on GPUs 3,4,5 (NEVER 6/7) into the shared b1_lowp_l1l2 dir.
set -uo pipefail

ROOT=/home/mert/p-bit_quantum_decoder
PY="$ROOT/.venv-gpu/bin/python"
POOL="$ROOT/code_capacity/run_lowp_pool.py"
OUT="$ROOT/code_capacity/results/b1_lowp_l1l2"
mkdir -p "$OUT"

log(){ echo "[$(date '+%F %T')] $*" >> "$OUT/CHAIN.log"; }

log "L2 p=0.05 watcher up (pid $$): waiting for 'sharpen ALL DONE' marker"
while ! grep -q "sharpen ALL DONE" "$OUT/CHAIN.log" 2>/dev/null; do
  sleep 30
done
log "sharpen finished; settling 15s then launching L2 p=0.05"
sleep 15

"$PY" "$POOL" --p 0.05 --gpus 3 4 5 --target-errors 10 \
  --name L2_s800_l4 --sweeps 800 --max-legs 4 --target-valid 4 \
  --chains 256 --rho-scale 0.35 --beta-start 0.5 --beta-stop 5.0 --shape linear \
  --gpu-round 2000 --seed-base 40000000000 --output-dir "$OUT"

log "L2 p=0.05 sharpen DONE"
